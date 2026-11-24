import os
import sys
import tempfile
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import models
from pretrain_jepa import compute_vicreg_regularization
from train_downstream import (
    AlphaBalancedFocalGDLLoss,
    feature_distillation_loss,
    build_llrd_optimizer,
    SegmentationGradCAM,
    export_to_onnx,
)


def make_synthetic_rgbdt(batch_size: int, h: int, w: int, device: torch.device) -> torch.Tensor:
    """Generates physical-contract 5-channel tensors: Z-scored RGB + [0, 1] Depth & Thermal."""
    rgb = torch.randn(batch_size, 3, h, w, device=device)
    depth = torch.rand(batch_size, 1, h, w, device=device) * 0.9 + 0.1
    therm = torch.rand(batch_size, 1, h, w, device=device) * 0.9 + 0.1
    return torch.cat([rgb, depth, therm], dim=1)


def make_stride32_mask(batch_size: int, h: int, w: int, mask_ratio: float, device: torch.device) -> torch.Tensor:
    patch = 32
    nh, nw = h // patch, w // patch
    total = nh * nw
    k = max(1, int(total * mask_ratio))
    mask = torch.zeros((batch_size, 1, h, w), device=device)
    rng = np.random.default_rng(42)
    for b in range(batch_size):
        chosen = rng.choice(total, k, replace=False)
        for idx in chosen:
            r, c = (idx // nw) * patch, (idx % nw) * patch
            mask[b, 0, r:r + patch, c:c + patch] = 1.0
    return mask


# ====================================================================================
# TEST 1: STEM PHYSICAL CALIBRATION & MODALITY DROPOUT STABILITY
# ====================================================================================

def test_stem_calibration_and_dropout(device: torch.device):
    orig = nn.Conv2d(3, 64, kernel_size=7, stride=4, padding=3)
    stem_no_scse = models.ModalityIsolatedPatchEmbed(orig, enable_dirac=True, enable_scse=False).to(device)
    stem_scse = models.ModalityIsolatedPatchEmbed(
        nn.Conv2d(3, 64, kernel_size=7, stride=4, padding=3), enable_dirac=True, enable_scse=True
    ).to(device)

    # Verify thermal identity gain at initialization: 2.0 * sigmoid(0.0) == 1.0
    init_therm_gain = float((2.0 * torch.sigmoid(stem_no_scse.therm_scale)).item())
    assert abs(init_therm_gain - 1.0) < 1e-6, f"Expected initial thermal gain 1.0, got {init_therm_gain}"

    x = make_synthetic_rgbdt(2, 128, 128, device)
    rgb_f, dt_f = stem_no_scse.forward_split(x)
    out = stem_no_scse(x)
    assert torch.allclose(rgb_f + dt_f, out, atol=1e-6), "forward_split() diverged from forward()!"

    # Extreme Modality Dropout: 100% zero Depth and 100% zero Thermal
    x_drop = x.clone()
    x_drop[:, 3, :, :] = 0.0
    x_drop[:, 4, :, :] = 0.0
    x_drop.requires_grad_(True)

    out_drop = stem_scse(x_drop)
    assert torch.isfinite(out_drop).all(), "Non-finite activations under 100% Depth+Thermal dropout!"
    out_drop.sum().backward()
    assert torch.isfinite(x_drop.grad).all(), "Non-finite input gradients under 100% Depth+Thermal dropout!"
    for name, p in stem_scse.named_parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), f"Non-finite parameter gradient in stem.{name}"


# ====================================================================================
# TEST 2: 2D PE ASYMMETRY, STRIDE-32 MASK ALIGNMENT & PREDICTOR NON-LEAKAGE
# ====================================================================================

def test_jepa_geometry_and_non_leakage(device: torch.device):
    # 2A. Verify 2D PE is NOT transpose-symmetric: PE(y, x) != PE(x, y)
    pe = models.PositionalEncoding2D(channels=64).to(device)
    emb = pe(torch.zeros(1, 64, 10, 10, device=device))
    assert not torch.allclose(emb[0, :, 2, 7], emb[0, :, 7, 2], atol=1e-5), \
        "PositionalEncoding2D is transpose-symmetric: (2, 7) equals (7, 2)!"

    # 2B. Verify SpatialJEPAPredictor cannot shortcut z_context at masked coordinates
    predictor = models.SpatialJEPAPredictor(embed_dim=64, hidden_dim=128).to(device)
    predictor.eval()
    z_ctx = torch.randn(2, 64, 4, 4, device=device)
    latent_mask = torch.zeros(2, 1, 4, 4, device=device)
    latent_mask[:, :, 1:3, 1:3] = 1.0

    z_ctx_perturbed = z_ctx.clone()
    z_ctx_perturbed[:, :, 1:3, 1:3] += 1000.0  # Massive perturbation strictly inside masked region

    with torch.no_grad():
        pred_a = predictor(z_ctx, latent_mask)
        pred_b = predictor(z_ctx_perturbed, latent_mask)
    max_diff = (pred_a - pred_b).abs().max().item()
    assert max_diff == 0.0, f"Predictor leaked masked z_context! Max diff = {max_diff}"

    # 2C. Verify stride-32 mask downsampling produces exact binary {0, 1} latent masks
    mask_hr = make_stride32_mask(2, 128, 128, 0.60, device)
    pooled = F.adaptive_avg_pool2d(mask_hr, output_size=(4, 4))
    unique_vals = set(torch.unique(pooled).cpu().tolist())
    assert unique_vals.issubset({0.0, 1.0}), f"Non-binary stride-32 pooled mask values: {unique_vals}"


# ====================================================================================
# TEST 3: PHASE 1 MM-JEPA FORWARD/BACKWARD, ENCODER VICREG GRADS & EMA SYNC
# ====================================================================================

def test_phase1_jepa_and_vicreg_gradients(device: torch.device):
    jepa = models.MultimodalJEPA(backbone_name='mit_b1', isolated_stem=True, enable_dirac=True, enable_scse=True).to(device)
    jepa.train()

    x_full = make_synthetic_rgbdt(2, 128, 128, device)
    mask = make_stride32_mask(2, 128, 128, 0.60, device)
    x_vis = x_full * (1.0 - mask)
    x_vis[0, 3, :, :] = 0.0  # Simulate depth modality dropout on sample 0

    # Pure VICReg backward test: must deliver non-zero gradients directly to context_encoder
    jepa.zero_grad(set_to_none=True)
    z_pred, z_target, latent_mask, z_context = jepa(x_vis, x_full, mask)
    assert z_pred.shape == z_target.shape == z_context.shape, "JEPA latent shape mismatch!"
    assert not z_target.requires_grad, "z_target must be detached!"

    loss_vicreg_spatial = compute_vicreg_regularization(z_context, variance_type='spatial', enable_covariance=True)
    loss_vicreg_batch = compute_vicreg_regularization(z_context, variance_type='batch', enable_covariance=True)
    assert torch.isfinite(loss_vicreg_spatial) and torch.isfinite(loss_vicreg_batch), "Non-finite VICReg loss!"

    loss_vicreg_spatial.backward()
    ctx_grad_norm = sum(
        p.grad.abs().sum().item() for p in jepa.context_encoder.parameters() if p.grad is not None
    )
    tgt_has_grad = any(p.grad is not None for p in jepa.target_encoder.parameters())
    assert ctx_grad_norm > 0.0, "VICReg produced ZERO gradients in context_encoder!"
    assert not tgt_has_grad, "target_encoder unexpectedly received gradients!"

    # Full composite Phase 1 backward + EMA target update
    jepa.zero_grad(set_to_none=True)
    z_pred, z_target, latent_mask, z_context = jepa(x_vis, x_full, mask)
    mask_exp = latent_mask.expand(-1, z_pred.shape[1], -1, -1)
    loss_target = F.mse_loss(F.normalize(z_pred, dim=1)[mask_exp == 1], F.normalize(z_target, dim=1)[mask_exp == 1])
    loss_context = F.mse_loss(F.normalize(z_pred, dim=1)[mask_exp == 0], F.normalize(z_target, dim=1)[mask_exp == 0])
    loss_reg = compute_vicreg_regularization(z_context, variance_type='spatial', enable_covariance=True)
    stem = jepa.context_encoder.patch_embed1.proj
    loss_phys = 0.01 * (((stem.depth_scale - 1.0) ** 2).mean() + (stem.therm_scale ** 2).mean())

    total_loss = loss_target + 0.5 * loss_context + 0.1 * loss_reg + loss_phys
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(jepa.parameters(), 1.0)

    # Verify EMA moves target parameters cleanly
    with torch.no_grad():
        for p in jepa.context_encoder.parameters():
            p.add_(0.01)
    before = jepa.target_encoder.patch_embed1.proj.rgb_proj.weight.detach().clone()
    jepa.update_target_network(tau=0.9)
    after = jepa.target_encoder.patch_embed1.proj.rgb_proj.weight.detach().clone()
    assert not torch.equal(before, after), "EMA target update failed to modify target_encoder weights!"


# ====================================================================================
# TEST 4: DOWNSTREAM V1/V2/V3 PARITY, LLRD & LOSS EDGE CASES
# ====================================================================================

def test_downstream_models_llrd_and_losses(device: torch.device):
    num_classes = 6
    x = make_synthetic_rgbdt(2, 128, 128, device)
    seg_mask = torch.randint(0, num_classes, (2, 128, 128), device=device)
    seg_mask[:, :12, :12] = 255  # Simulated rotated ignore_index border

    # Check V1, V2 (LoRA), V3 (full + LoRA) forward/backward parity
    configs = [
        ("v1", models.TMLPN_Downstream_v1(num_classes=num_classes, backbone_name='mit_b1', isolated_stem=True)),
        ("v2_lora", models.TMLPN_Downstream_v2(num_classes=num_classes, backbone_name='mit_b1', use_lora=True, lora_r=4)),
        ("v3_full", models.TMLPN_Downstream_v3(num_classes=num_classes, backbone_name='mit_b1', enable_scse=True, use_lora=False)),
        ("v3_lora", models.TMLPN_Downstream_v3(num_classes=num_classes, backbone_name='mit_b1', enable_scse=True, use_lora=True, lora_r=4)),
    ]

    criterion = AlphaBalancedFocalGDLLoss(
        num_classes=num_classes,
        alpha=[0.5, 1.0, 2.0, 3.0, 5.0, 10.0],
        global_gdl_weights=[0.01, 0.2, 1.0, 5.0, 25.0, 100.0],
        gamma=2.0,
        dice_weight=1.0,
        boundary_weight=0.5,
    ).to(device)

    for tag, net in configs:
        net = net.to(device)
        net.train()
        logits, feats = net(x, return_features=True)
        assert logits.shape == (2, num_classes, 128, 128), f"{tag}: unexpected logit shape {logits.shape}"
        assert len(feats) == 4, f"{tag}: expected 4 feature stages, got {len(feats)}"

        loss = criterion(logits, seg_mask)
        assert torch.isfinite(loss), f"{tag}: non-finite loss {loss.item()}"
        loss.backward()

        # Verify all-ignore batch does not produce NaN
        net.zero_grad(set_to_none=True)
        all_ignore = torch.full((2, 128, 128), 255, dtype=torch.long, device=device)
        loss_ign = criterion(net(x), all_ignore)
        assert torch.isfinite(loss_ign), f"{tag}: non-finite loss on all-ignore batch!"
        loss_ign.backward()

    # Verify LLRD assigns undecayed base_lr to decode_head, patch_embed1.proj, and lora_
    v3_lora = configs[-1][1]
    base_lr = 1e-4
    opt = build_llrd_optimizer(v3_lora, base_lr=base_lr, weight_decay=1e-4, decay_rate=0.85, phase="microtune")
    lr_by_param = {id(g['params'][0]): g['lr'] for g in opt.param_groups}
    for name, p in v3_lora.named_parameters():
        if not p.requires_grad:
            continue
        if 'decode_head' in name or 'patch_embed1.proj' in name or 'lora_' in name:
            assert abs(lr_by_param[id(p)] - base_lr) < 1e-12, f"LLRD decayed {name}: got {lr_by_param[id(p)]}"
        elif 'block1' in name:
            assert lr_by_param[id(p)] < base_lr, f"LLRD failed to decay {name}"


# ====================================================================================
# TEST 5: CROSS-ABLATION TEACHER KD CHECKPOINT RECONSTRUCTION
# ====================================================================================

def test_teacher_kd_checkpoint_loading(device: torch.device):
    num_classes = 6
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = os.path.join(tmpdir, "teacher_mit_b2_best_model.pt")

        # 1. Create and serialize a Control Optimal mit_b2 teacher (isolated_stem=True, scSE=True)
        teacher_orig = models.TMLPN_Downstream_v3(
            num_classes=num_classes, backbone_name='mit_b2',
            isolated_stem=True, enable_dirac=True, enable_scse=True, use_lora=False
        ).to(device)
        torch.save(teacher_orig.state_dict(), ckpt_path)

        # 2. Create a structurally DIFFERENT student ablation (Ablation_NaiveFusion: isolated_stem=False, scSE=False)
        student = models.TMLPN_Downstream_v3(
            num_classes=num_classes, backbone_name='mit_b1',
            isolated_stem=False, enable_dirac=False, enable_scse=False, use_lora=False
        ).to(device)

        # 3. Reload teacher via checkpoint-aware utility with strict=True
        loaded_sd = torch.load(ckpt_path, map_location=device)
        teacher_reloaded = models.instantiate_model_from_checkpoint(
            ModelClass=models.TMLPN_Downstream_v3,
            num_classes=num_classes,
            backbone='mit_b2',
            state_dict=loaded_sd,
            device=device,
            strict=True
        )
        teacher_reloaded.eval()
        for p in teacher_reloaded.parameters():
            p.requires_grad = False

        # Assert bit-exact parameter restoration (including scSE and ModalityIsolatedPatchEmbed)
        for (k1, v1), (k2, v2) in zip(teacher_orig.state_dict().items(), teacher_reloaded.state_dict().items()):
            assert k1 == k2 and torch.equal(v1, v2), f"Teacher parameter mismatch on key: {k1}"

        # 4. Run Feature-Level KD forward/backward between mismatched stems
        x = make_synthetic_rgbdt(2, 128, 128, device)
        _, s_feats = student(x, return_features=True)
        with torch.no_grad():
            _, t_feats = teacher_reloaded(x, return_features=True)

        kd_loss, valid_stages = feature_distillation_loss(s_feats, t_feats)
        assert valid_stages == 4, f"Expected 4 aligned MiT stages, got {valid_stages}"
        assert torch.isfinite(kd_loss) and kd_loss.item() > 0.0, f"Invalid KD loss: {kd_loss.item()}"

        kd_loss.backward()
        student_grad = sum(p.grad.abs().sum().item() for p in student.parameters() if p.grad is not None)
        teacher_grad = any(p.grad is not None for p in teacher_reloaded.parameters())
        assert student_grad > 0.0, "Student received zero gradients from Feature-Level KD!"
        assert not teacher_grad, "Frozen teacher unexpectedly received gradients!"


# ====================================================================================
# TEST 6: TRUE SEGMENTATION GRAD-CAM HOOK VERIFICATION
# ====================================================================================

def test_gradcam_hook(device: torch.device):
    model = models.TMLPN_Downstream_v3(num_classes=6, backbone_name='mit_b1').to(device)
    cam = SegmentationGradCAM(model, model.decode_head.linear_fuse[2])
    x = make_synthetic_rgbdt(1, 128, 128, device)

    heatmap, logits = cam.generate_cam(x, target_class=1)
    assert heatmap.shape == (32, 32), f"Unexpected Grad-CAM shape: {heatmap.shape}"
    assert np.isfinite(heatmap).all(), "Non-finite values in Grad-CAM heatmap!"
    assert heatmap.min() >= 0.0 and heatmap.max() <= 1.0 + 1e-5, f"Grad-CAM out of [0, 1]: [{heatmap.min()}, {heatmap.max()}]"
    assert logits.shape == (1, 6, 128, 128)
    cam.remove_hooks()


# ====================================================================================
# TEST 7: ONNX EXPORT & RUNTIME NUMERICAL PARITY
# ====================================================================================

def test_onnx_export_and_parity(device: torch.device, opset: int = 16):
    model = models.TMLPN_Downstream_v3(
        num_classes=6, backbone_name='mit_b1', isolated_stem=True, enable_scse=True, use_lora=False
    ).to(device)
    model.eval()

    with tempfile.TemporaryDirectory() as tmpdir:
        weights_path = os.path.join(tmpdir, "best_model.pt")
        torch.save(model.state_dict(), weights_path)

        onnx_path = export_to_onnx(
            model=model,
            weights_path=weights_path,
            run_dir=tmpdir,
            device=device,
            img_size=(128, 128),
            opset_version=opset
        )
        assert os.path.exists(onnx_path) and os.path.getsize(onnx_path) > 0, "ONNX file missing or empty!"

        try:
            import onnx
            onnx_model = onnx.load(onnx_path)
            onnx.checker.check_model(onnx_model)
            actual_opset = onnx_model.opset_import[0].version
            assert actual_opset == opset, f"Requested ONNX opset {opset}, but exported graph is opset {actual_opset}!"
            print(f"    -> onnx.checker.check_model PASSED (verified opset={actual_opset})")
        except ImportError:
            print("    -> [!] `onnx` package not installed; skipping graph checker.")

        try:
            import onnxruntime as ort
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if device.type == 'cuda' else ['CPUExecutionProvider']
            sess = ort.InferenceSession(onnx_path, providers=providers)

            x_test = make_synthetic_rgbdt(2, 128, 128, device)  # Tests dynamic batch axis (B=2 != trace B=1)
            with torch.no_grad():
                pt_out = model(x_test).cpu().numpy()

            ort_out = sess.run(None, {'input_5channel': x_test.cpu().numpy()})[0]
            max_diff = float(np.max(np.abs(pt_out - ort_out)))
            print(f"    -> ONNXRuntime parity (B=2 dynamic batch): max_abs_diff = {max_diff:.6e}")
            assert max_diff < 1e-3, f"PyTorch vs ONNXRuntime numerical mismatch: {max_diff}"
        except ImportError:
            print("    -> [!] `onnxruntime` package not installed; skipping numerical parity check.")


# ====================================================================================
# RUNNER
# ====================================================================================

def main():
    parser = argparse.ArgumentParser(description="TMLPN V3 Pre-Flight Architecture & Pipeline Validator")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--opset", type=int, default=16, help="ONNX opset version to validate")
    args = parser.parse_args()

    device = torch.device(args.device)
    print("\n" + "=" * 72)
    print(f"🧪 TMLPN V3 PRE-FLIGHT VALIDATION SUITE | Device: {device} | ONNX Opset: {args.opset}")
    print("=" * 72)

    tests = [
        ("1. Stem Physical Calibration & Dropout Stability", lambda: test_stem_calibration_and_dropout(device)),
        ("2. 2D PE Asymmetry, Stride-32 Mask & Non-Leakage", lambda: test_jepa_geometry_and_non_leakage(device)),
        ("3. Phase 1 MM-JEPA, Encoder VICReg Grads & EMA", lambda: test_phase1_jepa_and_vicreg_gradients(device)),
        ("4. Downstream V1/V2/V3, LLRD & Loss Masking", lambda: test_downstream_models_llrd_and_losses(device)),
        ("5. Cross-Ablation Teacher KD Checkpoint Loading", lambda: test_teacher_kd_checkpoint_loading(device)),
        ("6. True SegmentationGradCAM Feature Hook", lambda: test_gradcam_hook(device)),
        ("7. ONNX Export & Runtime Numerical Parity", lambda: test_onnx_export_and_parity(device, opset=args.opset)),
    ]

    failures = []
    for name, fn in tests:
        print(f"\n[*] Running {name}...")
        try:
            fn()
            print(f"[PASS] {name}")
        except Exception as exc:
            print(f"[FAIL] {name}: {exc}")
            failures.append((name, exc))

    print("\n" + "=" * 72)
    if failures:
        print(f"[-] PRE-FLIGHT FAILED: {len(failures)}/{len(tests)} checks failed.")
        for name, exc in failures:
            print(f"    - {name}: {exc}")
        sys.exit(1)
    else:
        print(f"[+] ALL {len(tests)}/{len(tests)} PRE-FLIGHT CHECKS PASSED. Safe to launch N=5 GPU matrix.")
    print("=" * 72 + "\n")


if __name__ == "__main__":
    main()