import os
import math
import random
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.amp import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from models import MultimodalJEPA
from dataset_jepa import JEPAPretrainDataset, seed_worker
from config_utils import parse_with_config

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def enforce_reproducibility(seed: int = 42) -> None:
    """Locks the computational graph for rigorous ablation statistics."""
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def atomic_torch_save(obj, path: str) -> None:
    """Prevents corrupted checkpoints if the process is terminated mid-write."""
    tmp_path = f"{path}.tmp"
    torch.save(obj, tmp_path)
    os.replace(tmp_path, path)


def compute_vicreg_regularization(z_online: torch.Tensor, variance_type: str = 'spatial',
                                  enable_covariance: bool = True, eps: float = 1e-4) -> torch.Tensor:
    """
    Computes Variance and Covariance regularization in float32 on unnormalized online
    encoder representations (z_context) to directly prevent encoder dimensional collapse.
    Reference: Bardes et al., VICReg, ICLR 2022.
    """
    z_online = z_online.float()
    B, C, _, _ = z_online.shape

    if variance_type == 'spatial':
        flat = z_online.permute(1, 0, 2, 3).reshape(C, -1)
        N = flat.shape[1]
        if N <= 1:
            return torch.tensor(0.0, device=z_online.device)

        std = torch.sqrt(flat.var(dim=1) + eps)
        loss_var = torch.mean(F.relu(1.0 - std))

        if enable_covariance:
            centered = flat - flat.mean(dim=1, keepdim=True)
            cov_matrix = (centered @ centered.T) / (N - 1)
            off_diag_mask = ~torch.eye(C, dtype=torch.bool, device=z_online.device)
            loss_cov = (cov_matrix[off_diag_mask] ** 2).sum() / C
        else:
            loss_cov = torch.tensor(0.0, device=z_online.device)

        return loss_var + (0.05 * loss_cov)

    elif variance_type == 'batch':
        pooled = z_online.mean(dim=(2, 3))
        if B <= 1:
            return torch.tensor(0.0, device=z_online.device)

        std = torch.sqrt(pooled.var(dim=0) + eps)
        loss_var = torch.mean(F.relu(1.0 - std))

        if enable_covariance:
            centered = pooled - pooled.mean(dim=0, keepdim=True)
            cov_matrix = (centered.T @ centered) / (B - 1)
            off_diag_mask = ~torch.eye(C, dtype=torch.bool, device=z_online.device)
            loss_cov = (cov_matrix[off_diag_mask] ** 2).sum() / C
        else:
            loss_cov = torch.tensor(0.0, device=z_online.device)

        return loss_var + (0.05 * loss_cov)

    return torch.tensor(0.0, device=z_online.device)


def main():
    args, config = parse_with_config("Phase 1: MM-JEPA Self-Supervised Pre-Training")

    cfg = config['phase1_pretraining']
    img_size = (config['dataset']['image_height'], config['dataset']['image_width'])
    dataset_name = config['dataset']['name']
    model_name = getattr(args, 'model', None) or config.get('metadata', {}).get('model', 'TMLPN_Downstream_v3')

    trial_name = cfg.get('trial_name', 'baseline')
    active_seed = cfg.get('seed', 42)
    enforce_reproducibility(active_seed)
    print(f"[*] Locked PyTorch computational graph to Seed: {active_seed}")

    ablation_cfg = cfg.get('ablations', {})
    enable_isolation = ablation_cfg.get('enable_modality_isolation', True)
    variance_type = ablation_cfg.get('variance_type', 'spatial')
    mask_strategy = ablation_cfg.get('mask_strategy', 'random')
    enable_context = ablation_cfg.get('enable_context_consistency', True)
    enable_covariance = ablation_cfg.get('enable_covariance_penalty', True)
    enable_dirac = ablation_cfg.get('enable_dirac', True)
    enable_scse = ablation_cfg.get('enable_scse', True)
    enable_modality_dropout = ablation_cfg.get('enable_modality_dropout', True)
    modality_dropout_prob = cfg.get('modality_dropout_prob', 0.15) if enable_modality_dropout else 0.0

    weight_dir = os.path.join("weights", model_name, dataset_name, trial_name)
    os.makedirs(weight_dir, exist_ok=True)
    ckpt_path = os.path.join(weight_dir, f"jepa_checkpoint_{cfg['backbone']}.pt")
    encoder_export_path = os.path.join(weight_dir, f"jepa_context_encoder_{cfg['backbone']}.pt")

    if os.path.exists(encoder_export_path) and os.path.exists(ckpt_path):
        existing_ckpt = torch.load(ckpt_path, map_location="cpu")
        if existing_ckpt.get('completed', False) or existing_ckpt.get('epoch', 0) >= cfg['epochs']:
            print(f"[>>] Phase 1 already completed ({existing_ckpt.get('epoch')}/{cfg['epochs']} epochs). Exiting.")
            return

    print("\n" + "=" * 60)
    print("🔬 ABLATION STATE [PHASE 1: PRE-TRAINING (V3 RECTIFIED)]")
    print("=" * 60)
    print(f"[*] Target Architecture Namespace      : {model_name}")
    print(f"[*] Modality Isolated Stem (1x1 Dirac) : {enable_isolation} (Dirac: {enable_dirac})")
    print(f"[*] Dynamic Cross-Modal Gating (scSE)  : {enable_scse}")
    print(f"[*] Variance Regularization Topology   : {variance_type.upper()} (Applied to z_context)")
    print(f"[*] Masking Strategy                   : {mask_strategy.upper()}")
    print(f"[*] Context Consistency Evaluated      : {enable_context}")
    print(f"[*] Covariance Penalty Evaluated       : {enable_covariance}")
    print(f"[*] Modality Dropout Regularizer       : {enable_modality_dropout} (Rate: {modality_dropout_prob})")
    print("=" * 60 + "\n")

    splits_root = os.path.join("data", "splits")
    dataset = JEPAPretrainDataset(
        dataset_name=dataset_name,
        split="train",
        splits_root=splits_root,
        image_size=img_size,
        mask_strategy=mask_strategy,
        modality_dropout_prob=modality_dropout_prob
    )

    dl_generator = torch.Generator()
    dl_generator.manual_seed(active_seed)

    dataloader = DataLoader(
        dataset,
        batch_size=cfg['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=(DEVICE.type == 'cuda'),
        drop_last=True,
        worker_init_fn=seed_worker,
        generator=dl_generator
    )

    if len(dataloader) == 0:
        raise RuntimeError(
            f"[-] CRITICAL: DataLoader has 0 batches (dataset={len(dataset)}, batch_size={cfg['batch_size']}, drop_last=True)."
        )

    model = MultimodalJEPA(
        backbone_name=cfg['backbone'],
        isolated_stem=enable_isolation,
        enable_dirac=enable_dirac,
        enable_scse=enable_scse
    ).to(DEVICE)

    base_lr = float(cfg['optimizer']['lr'])
    weight_decay = float(cfg['optimizer']['weight_decay'])
    min_lr = float(cfg['optimizer'].get('min_lr', 1e-6))
    min_lr_ratio = min_lr / max(base_lr, 1e-12)

    optimizer = optim.AdamW(model.parameters(), lr=base_lr, weight_decay=weight_decay)

    total_epochs = int(cfg['epochs'])
    total_steps = max(1, len(dataloader) * total_epochs)
    warmup_epochs = int(cfg['optimizer'].get('warmup_epochs', max(1, int(0.05 * total_epochs))))
    warmup_steps = min(total_steps - 1, max(1, warmup_epochs * len(dataloader)))

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    start_epoch = 0
    if os.path.exists(ckpt_path):
        print(f"[*] Resuming Phase 1 Pre-Training from checkpoint: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=DEVICE)
        torch.nn.Module.load_state_dict(model.context_encoder, checkpoint['context_encoder_state_dict'], strict=True)
        torch.nn.Module.load_state_dict(model.target_encoder, checkpoint['target_encoder_state_dict'], strict=True)
        model.predictor.load_state_dict(checkpoint['predictor_state_dict'])
        if 'encoder_mask_token' in checkpoint:
            model.encoder_mask_token.data.copy_(checkpoint['encoder_mask_token'].to(DEVICE))
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = int(checkpoint['epoch'])
        print(f"[*] Resumed at Epoch {start_epoch}/{total_epochs}")

    print(f"\n🚀 INITIATING PHASE 1: SELF-SUPERVISED MM-JEPA PRE-TRAINING ({cfg['backbone']} - {trial_name})")

    base_tau = float(cfg['jepa']['tau_base'])
    global_step = start_epoch * len(dataloader)
    use_bf16 = (DEVICE.type == 'cuda')

    for epoch in range(start_epoch, total_epochs):
        model.train()
        epoch_loss = 0.0

        loop = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{total_epochs} [Pre-Train]")
        for batch in loop:
            x_full = batch['x_full'].to(DEVICE, non_blocking=True)
            x_visible = batch['x_visible'].to(DEVICE, non_blocking=True)
            high_res_mask = batch['mask'].to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with autocast('cuda', dtype=torch.bfloat16, enabled=use_bf16):
                z_pred, z_target, latent_mask, z_context = model(x_visible, x_full, high_res_mask)

            # Cast latent objectives to float32 for numerical precision
            z_pred_f = z_pred.float()
            z_target_f = z_target.float()
            z_context_f = z_context.float()
            mask_exp = latent_mask.expand(-1, z_pred_f.shape[1], -1, -1)

            # 1. VICReg directly on unnormalized online encoder representations (z_context)
            loss_reg = compute_vicreg_regularization(
                z_context_f,
                variance_type=variance_type,
                enable_covariance=enable_covariance
            )

            # 2. L2-Normalized Direction-Aligned Latent Prediction Loss
            z_pred_norm = F.normalize(z_pred_f, dim=1, eps=1e-8)
            z_target_norm = F.normalize(z_target_f, dim=1, eps=1e-8)

            if (mask_exp == 1).any():
                loss_target = F.mse_loss(z_pred_norm[mask_exp == 1], z_target_norm[mask_exp == 1])
            else:
                loss_target = F.mse_loss(z_pred_norm, z_target_norm)

            if enable_context and (mask_exp == 0).any():
                loss_context = F.mse_loss(z_pred_norm[mask_exp == 0], z_target_norm[mask_exp == 0])
            else:
                loss_context = torch.tensor(0.0, device=DEVICE)

            # 3. Physical Calibration Anchoring Prior (depth_scale -> 1.0, therm_scale -> 0.0 so 2*sigmoid(0)=1.0)
            stem = model.context_encoder.patch_embed1.proj
            if hasattr(stem, 'depth_scale'):
                loss_depth_prior = ((stem.depth_scale - 1.0) ** 2).mean() + (stem.depth_bias ** 2).mean()
                loss_therm_prior = (stem.therm_scale ** 2).mean() + (stem.therm_bias ** 2).mean()
                loss_physics = 0.01 * (loss_depth_prior + loss_therm_prior)
            else:
                loss_physics = torch.tensor(0.0, device=DEVICE)

            loss = loss_target + (0.5 * loss_context) + (0.1 * loss_reg) + loss_physics

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            current_tau = 1.0 - (1.0 - base_tau) * (math.cos(math.pi * global_step / total_steps) + 1.0) / 2.0
            model.update_target_network(tau=current_tau)

            global_step += 1
            epoch_loss += loss.item()
            loop.set_postfix(
                loss=f"{loss.item():.4f}",
                lr=f"{scheduler.get_last_lr()[0]:.2e}",
                tau=f"{current_tau:.5f}"
            )

        print(f"Epoch {epoch + 1}/{total_epochs} | Latent Objective Loss: {epoch_loss / len(dataloader):.4f}")

        is_completed = (epoch + 1) >= total_epochs
        checkpoint = {
            'epoch': epoch + 1,
            'completed': is_completed,
            'context_encoder_state_dict': model.context_encoder.state_dict(),
            'target_encoder_state_dict': model.target_encoder.state_dict(),
            'predictor_state_dict': model.predictor.state_dict(),
            'encoder_mask_token': model.encoder_mask_token.detach().cpu(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'backbone': cfg['backbone']
        }
        atomic_torch_save(checkpoint, ckpt_path)

    # Export foundation encoder strictly after full pre-training convergence
    atomic_torch_save(model.context_encoder.state_dict(), encoder_export_path)
    print(f"[+] Phase 1 completed. Exported converged Context Encoder to: {encoder_export_path}")


if __name__ == '__main__':
    main()