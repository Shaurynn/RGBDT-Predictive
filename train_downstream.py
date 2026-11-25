import os
import glob
import json
import random
import datetime
import cv2
import optuna
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset_jepa import DownstreamSegmentationDataset, TTA_DownstreamSegmentationDataset, seed_worker
import models
from config_utils import parse_with_config

PHASE_SEQUENCE = ("baseline", "hpo", "hero", "microtune", "export")

def enforce_reproducibility(seed: int = 42) -> None:
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


def load_phase1_encoder_weights(context_encoder: nn.Module, pt_weights_path: str, device: torch.device) -> None:
    """
    Loads Phase 1 MM-JEPA context_encoder weights into Phase 2 safely:
    1. Calls nn.Module.load_state_dict directly (SMP MixVisionTransformerEncoder rejects `strict=`).
    2. Remaps unwrapped `.q.*`/`.kv.*` keys to `.q.original_layer.*`/`.kv.original_layer.*`
       when LoRA wrappers are active so attention weights are never silently left random.
    """
    raw_sd = torch.load(pt_weights_path, map_location=device, weights_only=True)
    if isinstance(raw_sd, dict) and 'context_encoder_state_dict' in raw_sd:
        raw_sd = raw_sd['context_encoder_state_dict']

    target_keys = set(context_encoder.state_dict().keys())
    remapped_sd = {}
    for k, v in raw_sd.items():
        if k in target_keys:
            remapped_sd[k] = v
            continue

        remapped_k = k
        for suffix in ('.q.weight', '.q.bias', '.kv.weight', '.kv.bias'):
            if k.endswith(suffix):
                proj, param = suffix.rsplit('.', 1)
                candidate = k[:-len(suffix)] + f"{proj}.original_layer.{param}"
                if candidate in target_keys:
                    remapped_k = candidate
                break
        remapped_sd[remapped_k] = v

    incompatible = nn.Module.load_state_dict(context_encoder, remapped_sd, strict=False)
    non_lora_missing = [k for k in incompatible.missing_keys if 'lora_' not in k]
    if non_lora_missing or incompatible.unexpected_keys:
        raise RuntimeError(
            f"[-] CRITICAL: Phase 1 -> Phase 2 encoder key mismatch! "
            f"Missing non-LoRA: {non_lora_missing[:5]} | Unexpected: {incompatible.unexpected_keys[:5]}"
        )
        
        
def build_model_kwargs(ModelClass, num_classes: int, backbone_name: str, isolated_stem: bool,
                       enable_dirac: bool, enable_scse: bool,
                       use_lora: bool, lora_r: int, lora_alpha: int) -> dict:
    """Constructs architecture-valid kwargs across V1, V2, and V3 downstream classes."""
    kwargs = {
        "num_classes": num_classes,
        "backbone_name": backbone_name,
        "isolated_stem": isolated_stem,
    }
    if ModelClass.__name__ == "TMLPN_Downstream_v3":
        kwargs.update({
            "enable_dirac": enable_dirac,
            "enable_scse": enable_scse,
            "use_lora": use_lora,
            "lora_r": lora_r,
            "lora_alpha": lora_alpha,
        })
    elif ModelClass.__name__ == "TMLPN_Downstream_v2":
        kwargs.update({
            "use_lora": use_lora,
            "lora_r": lora_r,
            "lora_alpha": lora_alpha,
        })
    return kwargs


# ====================================================================================
# --- LOSSES & METRICS ---
# ====================================================================================

class AlphaBalancedFocalGDLLoss(nn.Module):
    def __init__(self, num_classes: int, alpha=None, global_gdl_weights=None, gamma: float = 2.0,
                 dice_weight: float = 1.0, boundary_weight: float = 0.5, ignore_index: int = 255,
                 eps: float = 1e-6, use_batch_dynamic: bool = False):
        super().__init__()
        self.num_classes = num_classes
        self.gamma = gamma
        self.base_dice_weight = dice_weight
        self.boundary_weight = boundary_weight
        self.ignore_index = ignore_index
        self.eps = eps
        self.use_batch_dynamic = use_batch_dynamic

        if alpha is None:
            self.register_buffer('alpha', torch.ones(num_classes, dtype=torch.float32))
        else:
            alpha_t = torch.tensor(alpha, dtype=torch.float32)
            if alpha_t.size(0) != num_classes:
                alpha_t = (
                    alpha_t[:num_classes]
                    if alpha_t.size(0) > num_classes
                    else torch.cat([alpha_t, torch.ones(num_classes - alpha_t.size(0))])
                )
            self.register_buffer('alpha', alpha_t)

        if global_gdl_weights is None:
            self.register_buffer('gdl_weights', torch.ones(num_classes, dtype=torch.float32))
        else:
            gdl_t = torch.tensor(global_gdl_weights, dtype=torch.float32)
            if gdl_t.size(0) != num_classes:
                gdl_t = (
                    gdl_t[:num_classes]
                    if gdl_t.size(0) > num_classes
                    else torch.cat([gdl_t, torch.ones(num_classes - gdl_t.size(0))])
                )
            self.register_buffer('gdl_weights', gdl_t)

        sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]], dtype=torch.float32)
        self.register_buffer('sobel_weight_x', sobel_x.view(1, 1, 3, 3).repeat(num_classes, 1, 1, 1))
        self.register_buffer('sobel_weight_y', sobel_y.view(1, 1, 3, 3).repeat(num_classes, 1, 1, 1))

    def _compute_edge_map(self, x: torch.Tensor) -> torch.Tensor:
        grad_x = F.conv2d(x, self.sobel_weight_x, padding=1, groups=self.num_classes)
        grad_y = F.conv2d(x, self.sobel_weight_y, padding=1, groups=self.num_classes)
        return torch.sqrt(grad_x ** 2 + grad_y ** 2 + self.eps)

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        inputs = torch.clamp(inputs.float(), min=-20.0, max=20.0)

        illegal_mask = (targets < 0) | (targets >= self.num_classes)
        valid_mask = (targets != self.ignore_index) & (~illegal_mask)

        # 1. Alpha-Balanced Focal Loss averaged strictly over valid pixels
        ce_targets = targets.clone()
        ce_targets[~valid_mask] = self.ignore_index
        ce_loss = F.cross_entropy(inputs, ce_targets, reduction='none', ignore_index=self.ignore_index)

        ce_loss_safe = torch.clamp(ce_loss, min=0.0, max=50.0)
        pt = torch.exp(-ce_loss_safe)

        safe_targets = torch.clamp(targets, min=0, max=self.num_classes - 1)
        alpha_t = self.alpha[safe_targets]
        focal_map = alpha_t * ((1.0 - pt) ** self.gamma) * ce_loss

        valid_pixel_count = valid_mask.sum()
        if valid_pixel_count > 0:
            focal_loss = focal_map[valid_mask].sum() / valid_pixel_count.float()
        else:
            focal_loss = inputs.sum() * 0.0

        # 2. Generalized Dice Loss (Global Volume Anchored or Batch Dynamic)
        inputs_soft = F.softmax(inputs, dim=1)
        if valid_pixel_count == 0:
            dice_loss = inputs.sum() * 0.0
        else:
            valid_inputs_soft = inputs_soft.permute(0, 2, 3, 1)[valid_mask]
            valid_targets = safe_targets[valid_mask]
            targets_one_hot = F.one_hot(valid_targets, num_classes=self.num_classes).float()

            intersection = (valid_inputs_soft * targets_one_hot).sum(dim=0)
            ground_truth_volume = targets_one_hot.sum(dim=0)
            pred_volume = valid_inputs_soft.sum(dim=0)

            if self.use_batch_dynamic:
                present_in_batch = ground_truth_volume > 0
                w = torch.zeros_like(ground_truth_volume)
                w[present_in_batch] = 1.0 / (ground_truth_volume[present_in_batch] ** 2 + self.eps)
            else:
                w = self.gdl_weights[:self.num_classes].to(inputs.device)

            present_classes = w > 0
            if present_classes.any():
                numerator = 2.0 * (w[present_classes] * intersection[present_classes]).sum()
                denominator = (
                    w[present_classes] * (ground_truth_volume[present_classes] + pred_volume[present_classes])
                ).sum()
                dice_loss = 1.0 - (numerator / (denominator + self.eps))
            else:
                dice_loss = inputs.sum() * 0.0

        # 3. Explicit Sobel Boundary Loss strictly on eroded interior valid pixels
        if self.boundary_weight > 0 and valid_pixel_count > 0:
            targets_one_hot_spatial = (
                F.one_hot(safe_targets, num_classes=self.num_classes).permute(0, 3, 1, 2).float()
            )
            invalid_spatial = (~valid_mask).unsqueeze(1).float()
            # Dilate invalid region by 1 pixel so 3x3 Sobel never sees artificial ignore-border steps
            eroded_valid_mask = (1.0 - F.max_pool2d(invalid_spatial, kernel_size=3, stride=1, padding=1)) > 0.5

            pred_edges = self._compute_edge_map(inputs_soft)
            target_edges = self._compute_edge_map(targets_one_hot_spatial)

            # Sum edge discrepancies across classes per pixel, then average over valid interior pixels
            edge_l1_per_pixel = torch.abs(pred_edges - target_edges).sum(dim=1, keepdim=True)
            interior_count = eroded_valid_mask.sum()
            if interior_count > 0:
                boundary_loss = edge_l1_per_pixel[eroded_valid_mask].mean()
            else:
                boundary_loss = inputs.sum() * 0.0
        else:
            boundary_loss = inputs.sum() * 0.0

        return focal_loss + (self.base_dice_weight * dice_loss) + (self.boundary_weight * boundary_loss)


def feature_distillation_loss(student_features, teacher_features):
    """
    Computes L2-normalized multi-stage feature MSE loss.
    Returns (loss_tensor, valid_stages) so callers never halve task loss on mismatched architectures.
    """
    device = student_features[0].device
    loss = torch.tensor(0.0, device=device, dtype=torch.float32)
    valid_stages = 0
    for s_feat, t_feat in zip(student_features, teacher_features):
        if s_feat.shape == t_feat.shape:
            s_norm = F.normalize(s_feat.float(), dim=1, eps=1e-8)
            t_norm = F.normalize(t_feat.float(), dim=1, eps=1e-8)
            loss = loss + F.mse_loss(s_norm, t_norm)
            valid_stages += 1
    if valid_stages == 0:
        return loss, 0
    return loss / valid_stages, valid_stages


class IoUMetric:
    def __init__(self, num_classes: int, device: torch.device):
        self.num_classes = num_classes
        self.device = device
        self.intersections = torch.zeros(num_classes, device=device, dtype=torch.float64)
        self.unions = torch.zeros(num_classes, device=device, dtype=torch.float64)

    def update(self, logits: torch.Tensor, targets: torch.Tensor, ignore_index: int = 255) -> None:
        preds = torch.argmax(logits, dim=1)
        valid_mask = (
            (targets != ignore_index)
            & (targets >= 0) & (targets < self.num_classes)
            & (preds >= 0) & (preds < self.num_classes)
        )
        preds = preds[valid_mask]
        targets = targets[valid_mask]
        if targets.numel() == 0:
            return

        bins = targets * self.num_classes + preds
        bincount = torch.bincount(bins, minlength=self.num_classes ** 2)
        if bincount.numel() > self.num_classes ** 2:
            bincount = bincount[:self.num_classes ** 2]

        conf_matrix = bincount.reshape(self.num_classes, self.num_classes).double()
        intersection = torch.diag(conf_matrix)
        union = conf_matrix.sum(dim=1) + conf_matrix.sum(dim=0) - intersection
        self.intersections += intersection
        self.unions += union

    def get_per_class_iou(self) -> torch.Tensor:
        ious = self.intersections / torch.clamp(self.unions, min=1.0)
        ious[self.unions == 0] = 1.0
        return ious

    def get_miou(self) -> float:
        valid_classes = self.unions > 0
        if valid_classes.sum() == 0:
            return 0.0
        return self.get_per_class_iou()[valid_classes].mean().item()


def export_to_onnx(model, weights_path: str, run_dir: str, device: torch.device,
                   img_size=(480, 640), opset_version: int = 16) -> str:
    print(f"\n--- Serializing Architecture to ONNX (opset={opset_version}, size={img_size}) ---")
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    dummy_rgb = torch.randn(1, 3, img_size[0], img_size[1], device=device)
    dummy_dt = torch.rand(1, 2, img_size[0], img_size[1], device=device) * 0.9 + 0.1
    dummy_input = torch.cat([dummy_rgb, dummy_dt], dim=1)

    export_dir = os.path.join(run_dir, "deployment")
    os.makedirs(export_dir, exist_ok=True)
    onnx_path = os.path.join(export_dir, f"{model.__class__.__name__}.onnx")

    # Use legacy TorchScript ONNX exporter for opset < 18 so TensorRT/RKNN get true opset 16/17 graphs
    use_dynamo = opset_version >= 18
    export_kwargs = dict(
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['input_5channel'],
        output_names=['output_mask'],
        dynamo=use_dynamo,
    )
    if use_dynamo:
        export_kwargs['dynamic_shapes'] = {'x_full': {0: torch.export.Dim('batch_size')}}
    else:
        export_kwargs['dynamic_axes'] = {
            'input_5channel': {0: 'batch_size'},
            'output_mask': {0: 'batch_size'},
        }

    with torch.no_grad():
        torch.onnx.export(model, (dummy_input,), onnx_path, **export_kwargs)

    print(f"[+] ONNX model exported to: {onnx_path}")
    return onnx_path


# ====================================================================================
# --- PIPELINE MANAGEMENT, OPTIMIZER & GRAD-CAM ---
# ====================================================================================

def build_llrd_optimizer(model, base_lr: float, weight_decay: float, decay_rate: float, phase: str):
    if phase != "microtune":
        trainable = [p for p in model.parameters() if p.requires_grad]
        return optim.AdamW(trainable, lr=base_lr, weight_decay=weight_decay)

    param_groups = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if 'decode_head' in name or 'lora_' in name or 'patch_embed1.proj' in name:
            lr = base_lr
        elif 'block4' in name or 'patch_embed4' in name:
            lr = base_lr * (decay_rate ** 1)
        elif 'block3' in name or 'patch_embed3' in name:
            lr = base_lr * (decay_rate ** 2)
        elif 'block2' in name or 'patch_embed2' in name:
            lr = base_lr * (decay_rate ** 3)
        elif 'block1' in name or 'patch_embed1' in name:
            lr = base_lr * (decay_rate ** 4)
        else:
            lr = base_lr * (decay_rate ** 5)

        param_groups.append({'params': [param], 'lr': lr, 'weight_decay': weight_decay})

    return optim.AdamW(param_groups)


def resolve_teacher_checkpoint(args, cfg, model_name: str, dataset_name: str, active_seed: int):
    explicit_path = getattr(args, 'teacher_weights', None) or cfg.get('teacher_weights', None)
    if explicit_path and os.path.exists(explicit_path):
        return explicit_path

    teacher_bb = cfg.get('teacher_backbone', 'mit_b4')
    if cfg.get('backbone') == teacher_bb:
        return None

    search_patterns = [
        os.path.join("results", model_name, dataset_name, f"{teacher_bb}_baseline_seed{active_seed}", "*_Microtune", "best_model.pt"),
        os.path.join("results", model_name, dataset_name, f"{teacher_bb}_baseline_seed{active_seed}", "*_Hero", "best_model.pt"),
        os.path.join("results", model_name, dataset_name, f"{teacher_bb}_baseline_seed*", "*_Microtune", "best_model.pt"),
        os.path.join("results", model_name, dataset_name, teacher_bb, "*_Microtune", "best_model.pt"),
    ]
    for pattern in search_patterns:
        matches = sorted(glob.glob(pattern))
        if matches:
            return matches[-1]
    return None


def find_hpo_best_params(model_dir: str):
    """Locates the latest HPO phase's best_params.json for both Hero and Microtune phases."""
    matches = sorted(glob.glob(os.path.join(model_dir, "*_Hpo", "best_params.json")))
    return matches[-1] if matches else None


class ExperimentManager:
    def __init__(self, model_instance, dataset: str = "MM5", backbone: str = "mit_b1", base_dir: str = "results"):
        self.model_name = model_instance.__class__.__name__
        self.dataset = dataset
        self.backbone = backbone
        self.model_dir = os.path.join(base_dir, self.model_name, self.dataset, self.backbone)
        os.makedirs(self.model_dir, exist_ok=True)
        self.phase_sequence = list(PHASE_SEQUENCE)

    def detect_state(self):
        existing_runs = sorted(glob.glob(os.path.join(self.model_dir, "*_*")))
        if not existing_runs:
            return self._create_new_run("baseline", resume_from=None)
        latest_run = existing_runs[-1]
        run_name = os.path.basename(latest_run)
        if os.path.exists(os.path.join(latest_run, "results.json")):
            current_phase = run_name.split("_")[-1].lower()
            if current_phase == self.phase_sequence[-1]:
                return None
            next_phase = self.phase_sequence[self.phase_sequence.index(current_phase) + 1]
            inherit_weights = os.path.join(latest_run, "best_model.pt")
            return self._create_new_run(next_phase, resume_from=inherit_weights)
        return self._resume_run(latest_run)

    def _create_new_run(self, phase: str, resume_from):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(self.model_dir, f"{timestamp}_{phase.capitalize()}")
        os.makedirs(run_dir)
        os.makedirs(os.path.join(run_dir, "weights"))
        os.makedirs(os.path.join(run_dir, "logs"))
        state = {
            "run_dir": run_dir,
            "phase": phase,
            "is_resume": False,
            "inherit_weights": resume_from,
            "start_epoch": 0,
            "best_miou": -1.0,
            "best_train_loss": None,
            "best_val_loss": None,
            "patience_counter": 0
        }
        with open(os.path.join(run_dir, "state.json"), 'w') as f:
            json.dump(state, f, indent=4)
        return state

    def _resume_run(self, run_dir: str):
        with open(os.path.join(run_dir, "state.json"), 'r') as f:
            state = json.load(f)
        state["is_resume"] = True
        return state


class SegmentationGradCAM:
    """
    True Grad-CAM hooked onto an intermediate decoder feature map (e.g., decode_head.linear_fuse[2])
    using a tensor-level backward hook (immune to PyTorch BackwardHookFunction view+inplace errors).
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self._fwd_handle = self.target_layer.register_forward_hook(self._save_activation)

    def _save_activation(self, module, inp, output):
        self.activations = output
        if output.requires_grad:
            output.register_hook(self._save_gradient)

    def _save_gradient(self, grad):
        self.gradients = grad

    def remove_hooks(self):
        self._fwd_handle.remove()

    def generate_cam(self, input_tensor: torch.Tensor, target_class: int):
        self.model.eval()
        self.model.zero_grad(set_to_none=True)
        self.activations = None
        self.gradients = None

        logits = self.model(input_tensor)
        loss = logits[:, target_class, :, :].sum()
        loss.backward()

        if self.activations is None or self.gradients is None:
            raise RuntimeError("[-] Grad-CAM failed to capture activations or gradients.")

        pooled_gradients = torch.mean(self.gradients.detach(), dim=(0, 2, 3))
        activations = self.activations.detach()
        weights = pooled_gradients.view(1, -1, 1, 1)
        heatmap = F.relu((activations * weights).sum(dim=1).squeeze(0))

        max_val = heatmap.max()
        if max_val > 0:
            heatmap = heatmap / max_val

        self.model.zero_grad(set_to_none=True)
        self.activations = None
        self.gradients = None
        return heatmap.cpu().numpy(), logits.detach()


def get_dynamic_class_count(data_dir: str) -> int:
    class_file = os.path.join(data_dir, "classes.txt")
    with open(class_file, "r") as f:
        return len([line for line in f if line.strip()])


def compute_dataset_statistics(dataset_name: str, splits_root: str, img_size, num_classes: int,
                               ignore_index: int = 255):
    """
    Computes exact Median Frequency Balancing alpha and full-precision Global Volume Anchored
    GDL weights over the full deterministic (unaugmented, unshuffled) training split.
    Caches weights per dataset + resolution to avoid repeated full-split scans.
    """
    cache_path = os.path.join(splits_root, dataset_name, f"class_loss_weights_{img_size[0]}x{img_size[1]}.json")
    if os.path.exists(cache_path):
        with open(cache_path, "r") as f:
            cached = json.load(f)
        if len(cached.get("alpha", [])) == num_classes and len(cached.get("gdl", [])) == num_classes:
            return cached["alpha"], cached["gdl"]

    print(f"[*] Computing full-split deterministic class statistics for {dataset_name} ({img_size})...")
    stats_dataset = DownstreamSegmentationDataset(
        dataset_name=dataset_name,
        split="train",
        splits_root=splits_root,
        image_size=img_size,
        enable_augmentation=False,
        modality_dropout_prob=0.0
    )
    stats_loader = DataLoader(stats_dataset, batch_size=8, shuffle=False, num_workers=4)

    pixel_counts = torch.zeros((num_classes,), dtype=torch.float64)
    for batch in stats_loader:
        targets = batch['seg_mask']
        valid_targets = targets[(targets != ignore_index) & (targets >= 0) & (targets < num_classes)]
        if valid_targets.numel() > 0:
            counts = torch.bincount(valid_targets.flatten(), minlength=num_classes).double()
            pixel_counts += counts

    present = pixel_counts > 0
    total_valid = pixel_counts.sum().clamp(min=1.0)
    frequencies = pixel_counts / total_valid

    alpha = torch.ones((num_classes,), dtype=torch.float64)
    gdl = torch.zeros((num_classes,), dtype=torch.float64)

    if present.any():
        present_freqs = frequencies[present]
        median_freq = torch.median(present_freqs)
        alpha[present] = median_freq / present_freqs

        # Normalize GDL weights by their median across present classes (preserves exact 1/f_c^2
        # ratios in float32 without rounding non-rare classes to 0.0000)
        gdl_raw = 1.0 / (present_freqs ** 2)
        gdl_norm = gdl_raw / torch.median(gdl_raw)
        gdl[present] = torch.clamp(gdl_norm, min=1e-6, max=1e6)

    alpha_list = alpha.tolist()
    gdl_list = gdl.tolist()

    with open(cache_path, "w") as f:
        json.dump({"alpha": alpha_list, "gdl": gdl_list, "pixel_counts": pixel_counts.tolist()}, f, indent=4)

    return alpha_list, gdl_list


def run_hpo_phase(run_dir: str, inherit_weights, ModelClass, model_kwargs: dict,
                  train_loader, val_loader, num_classes: int, empirical_alpha, global_gdl,
                  device: torch.device, cfg_hpo: dict, accum_steps: int = 4, active_seed: int = 42,
                  enable_boundary_loss: bool = True, gdl_type: str = 'global_anchored',
                  teacher_model=None, enable_kd: bool = False):
    db_path = os.path.join(run_dir, "hpo_sweep.db")
    sampler = optuna.samplers.TPESampler(seed=active_seed)
    study = optuna.create_study(
        study_name="TMLPN_HPO_Sweep",
        storage=f"sqlite:///{db_path}",
        direction="maximize",
        sampler=sampler,
        pruner=optuna.pruners.HyperbandPruner(),
        load_if_exists=True
    )

    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None]
    best_study_miou = [max((t.value for t in completed_trials), default=-1.0)]
    use_bf16 = (device.type == 'cuda')
    hpo_epochs = int(cfg_hpo.get('epochs_per_trial', 30))

    def objective(trial):
        model = ModelClass(**model_kwargs).to(device)
        if inherit_weights and os.path.exists(inherit_weights):
            model.load_state_dict(torch.load(inherit_weights, map_location=device, weights_only=True), strict=True)

        for name, param in model.context_encoder.named_parameters():
            param.requires_grad = ('patch_embed1.proj' in name)

        lr = trial.suggest_float("lr", cfg_hpo['lr_min'], cfg_hpo['lr_max'], log=True)
        wd = trial.suggest_float("weight_decay", cfg_hpo['wd_min'], cfg_hpo['wd_max'], log=True)
        gamma = trial.suggest_float("gamma", cfg_hpo.get('gamma_min', 1.0), cfg_hpo.get('gamma_max', 2.5))
        dice = trial.suggest_float("dice_weight", 0.5, 2.0)
        boundary_weight = trial.suggest_float("boundary_weight", 0.1, 1.0) if enable_boundary_loss else 0.0

        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=wd)

        criterion = AlphaBalancedFocalGDLLoss(
            num_classes=num_classes,
            alpha=empirical_alpha,
            global_gdl_weights=global_gdl if gdl_type == 'global_anchored' else None,
            gamma=gamma,
            dice_weight=dice,
            boundary_weight=boundary_weight,
            use_batch_dynamic=(gdl_type == 'batch_dynamic')
        ).to(device)

        best_trial_miou = -1.0

        for epoch in range(hpo_epochs):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            for batch_idx, batch in enumerate(train_loader):
                x_full = batch['x_full'].to(device, non_blocking=True)
                seg_mask = batch['seg_mask'].to(device, non_blocking=True)

                with autocast('cuda', dtype=torch.bfloat16, enabled=use_bf16):
                    pred_seg, s_feats = model(x_full, return_features=True)
                    loss = criterion(pred_seg, seg_mask)

                    if enable_kd and teacher_model is not None:
                        with torch.no_grad():
                            _, t_feats = teacher_model(x_full, return_features=True)
                        loss_kd, valid_stages = feature_distillation_loss(s_feats, t_feats)
                        if valid_stages > 0:
                            loss = (0.5 * loss) + (0.5 * loss_kd)

                    loss = loss / accum_steps

                loss.backward()

                if (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == len(train_loader):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    is_valid = all(
                        torch.isfinite(p.grad).all()
                        for p in model.parameters()
                        if p.grad is not None
                    )
                    if is_valid:
                        optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

            model.eval()
            val_iou = IoUMetric(num_classes, device)
            with torch.no_grad():
                for batch in val_loader:
                    x_full = batch['x_full'].to(device, non_blocking=True)
                    seg_mask = batch['seg_mask'].to(device, non_blocking=True)
                    with autocast('cuda', dtype=torch.bfloat16, enabled=use_bf16):
                        pred_seg = model(x_full)
                    val_iou.update(pred_seg, seg_mask)

            score = val_iou.get_miou()
            if score > best_trial_miou:
                best_trial_miou = score
            if score > best_study_miou[0]:
                best_study_miou[0] = score
                atomic_torch_save(model.state_dict(), os.path.join(run_dir, "best_model.pt"))

            trial.report(score, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        return best_trial_miou

    print("\n" + "=" * 80)
    print("📊 [MONITOR HPO] To track the Optuna sweep in real-time, run in a new terminal:")
    print(f"    optuna-dashboard sqlite:///{db_path}")
    print("=" * 80 + "\n")

    study.optimize(objective, n_trials=cfg_hpo['n_trials'])
    with open(os.path.join(run_dir, "best_params.json"), "w") as f:
        json.dump(study.best_params, f, indent=4)
    return study.best_value


# ====================================================================================
# --- MAIN DOWNSTREAM ENGINE ---
# ====================================================================================

def main():
    args, config = parse_with_config("Phase 2: Downstream Supervised Semantic Segmentation")
    cfg = config['phase2_downstream']
    img_size = (config['dataset']['image_height'], config['dataset']['image_width'])
    dataset_name = config['dataset']['name']
    trial_name = cfg.get('trial_name', 'baseline')
    active_seed = cfg.get('seed', 42)
    enforce_reproducibility(active_seed)

    model_name = getattr(args, 'model', None) or config.get('metadata', {}).get('model', 'TMLPN_Downstream_v3')
    ModelClass = getattr(models, model_name, None)
    if ModelClass is None:
        raise ValueError(f"[-] CRITICAL: Architecture class '{model_name}' not found in models.py")

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_bf16 = (DEVICE.type == 'cuda')
    splits_root = os.path.join("data", "splits")
    NUM_CLASSES = get_dynamic_class_count(os.path.join(splits_root, dataset_name))

    ablation_cfg = cfg.get('ablations', {})
    enable_modality_dropout = ablation_cfg.get('enable_modality_dropout', True)
    enable_boundary_loss = ablation_cfg.get('enable_boundary_loss', True)
    modality_dropout_prob = cfg.get('modality_dropout_prob', 0.15) if enable_modality_dropout else 0.0

    gdl_type = ablation_cfg.get('gdl_type', 'global_anchored')
    enable_kd = ablation_cfg.get('enable_kd', True)
    use_lora = ablation_cfg.get('use_lora', cfg.get('use_lora', False))
    accum_steps = cfg.get('gradient_accumulation_steps', 4)
    lora_r = cfg.get('lora', {}).get('r', 8)
    lora_alpha = cfg.get('lora', {}).get('alpha', 16)
    llrd_decay = ablation_cfg.get('llrd_decay', cfg.get('llrd_decay', 0.85))

    high_capacity_backbones = {'mit_b3', 'mit_b4', 'mit_b5'}
    if use_lora and ablation_cfg.get('dynamic_lora_rank', False) and cfg['backbone'] in high_capacity_backbones:
        print(f"[*] Dynamic LoRA Active: Scaling down adapter capacity for {cfg['backbone']}")
        lora_r = max(4, lora_r // 2)
        lora_alpha = max(8, lora_alpha // 2)

    if ablation_cfg.get('adaptive_llrd_decay', False) and cfg['backbone'] in high_capacity_backbones:
        print(f"[*] Adaptive LLRD Active: Enforcing steeper gradient decay for {cfg['backbone']}")
        llrd_decay = 0.75

    model_kwargs = build_model_kwargs(
        ModelClass=ModelClass,
        num_classes=NUM_CLASSES,
        backbone_name=cfg['backbone'],
        isolated_stem=ablation_cfg.get('enable_modality_isolation', True),
        enable_dirac=ablation_cfg.get('enable_dirac', True),
        enable_scse=ablation_cfg.get('enable_scse', True),
        use_lora=use_lora,
        lora_r=lora_r,
        lora_alpha=lora_alpha
    )

    model = ModelClass(**model_kwargs).to(DEVICE)

    isolated_backbone = f"{cfg['backbone']}_{trial_name}" if trial_name != "baseline" else cfg['backbone']
    manager = ExperimentManager(model_instance=model, dataset=dataset_name, backbone=isolated_backbone)
    state = manager.detect_state()
    if state is None:
        return
    run_dir, phase = state["run_dir"], state["phase"]

    # Fast-path export phase without spinning up training DataLoaders or dataset statistics
    if phase == "export":
        onnx_opset = int(cfg.get('onnx_opset', 16))
        export_to_onnx(model, state["inherit_weights"], run_dir, DEVICE, img_size=img_size, opset_version=onnx_opset)
        with open(os.path.join(run_dir, "results.json"), 'w') as f:
            json.dump({"phase": phase, "onnx_opset": onnx_opset}, f, indent=4)
        return

    # Deterministic full-split class statistics (never perturbs train_loader RNG)
    empirical_alpha, global_gdl = compute_dataset_statistics(dataset_name, splits_root, img_size, NUM_CLASSES)

    train_dataset = DownstreamSegmentationDataset(
        dataset_name=dataset_name,
        split="train",
        splits_root=splits_root,
        image_size=img_size,
        modality_dropout_prob=modality_dropout_prob
    )
    val_dataset = DownstreamSegmentationDataset(
        dataset_name=dataset_name,
        split="val",
        splits_root=splits_root,
        image_size=img_size,
        enable_augmentation=False
    )

    dl_generator = torch.Generator()
    dl_generator.manual_seed(active_seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=(DEVICE.type == 'cuda'),
        worker_init_fn=seed_worker,
        generator=dl_generator
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=(DEVICE.type == 'cuda')
    )

    # Reconstruct Teacher strictly from its own checkpoint topology (never student ablation kwargs)
    teacher_model = None
    teacher_ckpt_path = resolve_teacher_checkpoint(args, cfg, model_name, dataset_name, active_seed) if enable_kd else None
    if enable_kd and teacher_ckpt_path and os.path.exists(teacher_ckpt_path):
        teacher_bb = cfg.get('teacher_backbone', 'mit_b4')
        print(f"[*] Initializing Feature-Level KD Teacher ({teacher_bb}) from: {teacher_ckpt_path}")
        t_state_dict = torch.load(teacher_ckpt_path, map_location=DEVICE, weights_only=True)
        teacher_model = models.instantiate_model_from_checkpoint(
            ModelClass=ModelClass,
            num_classes=NUM_CLASSES,
            backbone=teacher_bb,
            state_dict=t_state_dict,
            device=DEVICE,
            strict=True
        )
        teacher_model.eval()
        for p in teacher_model.parameters():
            p.requires_grad = False

    patience_cfg = cfg.get('early_stopping_patience', {'default': 20})
    if isinstance(patience_cfg, dict):
        early_stop_patience = int(patience_cfg.get(phase, patience_cfg.get('default', 20)))
    else:
        early_stop_patience = int(patience_cfg)

    if phase == "baseline" and not state["is_resume"]:
        pt_weights = os.path.join("weights", model.__class__.__name__, dataset_name, trial_name, f"jepa_context_encoder_{cfg['backbone']}.pt")
        if not os.path.exists(pt_weights):
            fallback_pt = os.path.join("weights", model.__class__.__name__, dataset_name, f"baseline_seed{active_seed}", f"jepa_context_encoder_{cfg['backbone']}.pt")
            if os.path.exists(fallback_pt):
                pt_weights = fallback_pt
        if os.path.exists(pt_weights):
            print(f"[*] Injecting Phase 1 MM-JEPA Foundation Weights: {pt_weights}")
            load_phase1_encoder_weights(model.context_encoder, pt_weights, DEVICE)

    if phase in ["baseline", "hpo", "hero"]:
        print(f"[*] Enforcing frozen foundation (with trainable Modality-Isolated Stem) for phase: {phase}")
        for name, param in model.context_encoder.named_parameters():
            param.requires_grad = ('patch_embed1.proj' in name)

    if phase == "hpo":
        score = run_hpo_phase(
            run_dir=run_dir,
            inherit_weights=state["inherit_weights"],
            ModelClass=ModelClass,
            model_kwargs=model_kwargs,
            train_loader=train_loader,
            val_loader=val_loader,
            num_classes=NUM_CLASSES,
            empirical_alpha=empirical_alpha,
            global_gdl=global_gdl,
            device=DEVICE,
            cfg_hpo=cfg['hpo'],
            accum_steps=accum_steps,
            active_seed=active_seed,
            enable_boundary_loss=enable_boundary_loss,
            gdl_type=gdl_type,
            teacher_model=teacher_model,
            enable_kd=enable_kd
        )
        with open(os.path.join(run_dir, "results.json"), 'w') as f:
            json.dump({"phase": phase, "best_mIoU": score, "val_mIoU_checkpoint": score}, f, indent=4)
        return

    MAX_EPOCHS = int(cfg['epochs'].get(phase, cfg['epochs']['default']))
    lr = float(cfg['learning_rates'].get(phase, cfg['learning_rates']['default']))
    weight_decay = float(cfg.get('optimizer', {}).get('weight_decay', 1e-4))
    gamma, dice_weight = 2.0, 1.0
    boundary_weight = 0.5 if enable_boundary_loss else 0.0

    if phase in ["hero", "microtune"]:
        hpo_params_path = find_hpo_best_params(manager.model_dir)
        if hpo_params_path and os.path.exists(hpo_params_path):
            print(f"[*] Loading HPO optimal hyperparameters from: {hpo_params_path}")
            with open(hpo_params_path, 'r') as f:
                best_params = json.load(f)
            lr = float(best_params.get("lr", lr))
            weight_decay = float(best_params.get("weight_decay", weight_decay))
            gamma = float(best_params.get("gamma", gamma))
            dice_weight = float(best_params.get("dice_weight", dice_weight))
            boundary_weight = float(best_params.get("boundary_weight", boundary_weight)) if enable_boundary_loss else 0.0
            if phase == "microtune":
                lr = float(cfg['learning_rates'].get('microtune', 1e-5))

        if state.get("inherit_weights") and os.path.exists(state["inherit_weights"]):
            print(f"[*] Inheriting converged weights from previous phase: {state['inherit_weights']}")
            model.load_state_dict(torch.load(state["inherit_weights"], map_location=DEVICE, weights_only=True), strict=True)

    if phase == "microtune":
        print("\n[*] Initializing Microtune Phase")
        if use_lora:
            print(f"[*] LoRA Active (r={lora_r}, alpha={lora_alpha}): Tuning Decoder, Modality-Isolated Stem, and Low-Rank matrices.")
            for name, param in model.named_parameters():
                param.requires_grad = ('lora_' in name or 'decode_head' in name or 'patch_embed1.proj' in name)
        else:
            print("[*] LoRA Inactive: Full backbone unfreezing with Layer-Wise Learning Rate Decay (LLRD).")
            for param in model.parameters():
                param.requires_grad = True

    optimizer = build_llrd_optimizer(model, lr, weight_decay, llrd_decay, phase)

    criterion = AlphaBalancedFocalGDLLoss(
        num_classes=NUM_CLASSES,
        alpha=empirical_alpha,
        global_gdl_weights=global_gdl if gdl_type == 'global_anchored' else None,
        gamma=gamma,
        dice_weight=dice_weight,
        boundary_weight=boundary_weight,
        use_batch_dynamic=(gdl_type == 'batch_dynamic')
    ).to(DEVICE)

    active_hparams = {
        **ablation_cfg,
        "lr": lr,
        "weight_decay": weight_decay,
        "gamma": gamma,
        "dice_weight": dice_weight,
        "boundary_weight": boundary_weight,
        "batch_size": cfg['batch_size'],
        "use_lora": use_lora,
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "llrd_decay": llrd_decay,
        "modality_dropout_prob": modality_dropout_prob,
        "teacher_checkpoint": teacher_ckpt_path,
    }
    state["hyperparameters"] = active_hparams

    writer = SummaryWriter(log_dir=os.path.join(run_dir, "logs"))
    writer.add_text("Configuration/Hyperparameters", json.dumps(active_hparams, indent=4), 0)

    print("\n" + "=" * 80)
    print(f"📈 [MONITOR CURRENT PHASE] tensorboard --logdir {os.path.join(run_dir, 'logs')}")
    print(f"🌐 [MONITOR FULL PIPELINE] tensorboard --logdir {os.path.dirname(os.path.dirname(run_dir))}")
    print("=" * 80 + "\n")

    # Hook post-ReLU fused feature layer for genuine Grad-CAM (not raw linear_pred logits)
    cam_extractor = SegmentationGradCAM(model, model.decode_head.linear_fuse[2])

    start_epoch = 0
    patience_counter = int(state.get("patience_counter", 0))

    if state["is_resume"]:
        checkpoint_path = os.path.join(run_dir, "checkpoint.pt")
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=True)
            model.load_state_dict(checkpoint['model_state_dict'], strict=True)
            try:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            except ValueError:
                pass
            start_epoch = int(checkpoint['epoch'])
            state["best_miou"] = float(checkpoint['best_miou'])
            state["best_train_loss"] = checkpoint.get('best_train_loss', state.get("best_train_loss"))
            state["best_val_loss"] = checkpoint.get('best_val_loss', state.get("best_val_loss"))
            patience_counter = int(checkpoint.get('patience_counter', 0))

    avg_train_loss = 0.0
    avg_val_loss = 0.0

    for epoch in range(start_epoch, MAX_EPOCHS):
        model.train()
        train_loss_sum = 0.0

        optimizer.zero_grad(set_to_none=True)
        loop = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{MAX_EPOCHS} [{phase}]")

        for batch_idx, batch in enumerate(loop):
            x_full = batch['x_full'].to(DEVICE, non_blocking=True)
            seg_mask = batch['seg_mask'].to(DEVICE, non_blocking=True)

            with autocast('cuda', dtype=torch.bfloat16, enabled=use_bf16):
                pred_seg, s_feats = model(x_full, return_features=True)
                loss = criterion(pred_seg, seg_mask)

                if enable_kd and teacher_model is not None:
                    with torch.no_grad():
                        _, t_feats = teacher_model(x_full, return_features=True)
                    loss_kd, valid_stages = feature_distillation_loss(s_feats, t_feats)
                    if valid_stages > 0:
                        loss = (0.5 * loss) + (0.5 * loss_kd)

                scaled_loss = loss / accum_steps

            scaled_loss.backward()

            if (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                is_valid_gradients = all(
                    torch.isfinite(p.grad).all()
                    for p in model.parameters()
                    if p.grad is not None
                )
                if is_valid_gradients:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            step_loss_val = loss.item()
            train_loss_sum += step_loss_val
            loop.set_postfix(loss=f"{step_loss_val:.4f}")

        avg_train_loss = train_loss_sum / max(1, len(train_loader))
        writer.add_scalar("Loss/Train", avg_train_loss, epoch)

        model.eval()
        val_loss_sum = 0.0
        val_iou = IoUMetric(NUM_CLASSES, DEVICE)
        logged_image = False

        with torch.no_grad():
            for batch in val_loader:
                x_full = batch['x_full'].to(DEVICE, non_blocking=True)
                seg_mask = batch['seg_mask'].to(DEVICE, non_blocking=True)

                with autocast('cuda', dtype=torch.bfloat16, enabled=use_bf16):
                    pred_seg = model(x_full)
                    loss = criterion(pred_seg, seg_mask)
                    val_loss_sum += loss.item()
                val_iou.update(pred_seg, seg_mask)

                if not logged_image and epoch % 5 == 0:
                    sample_mask = seg_mask[0]
                    fg_classes = sample_mask[(sample_mask > 0) & (sample_mask < NUM_CLASSES)]
                    target_cls = int(fg_classes[0].item()) if fg_classes.numel() > 0 else min(1, NUM_CLASSES - 1)

                    with torch.enable_grad():
                        heatmap, _ = cam_extractor.generate_cam(x_full[0:1], target_class=target_cls)

                    rgb_vis = x_full[0, :3].cpu().numpy().transpose(1, 2, 0)
                    rgb_vis = (rgb_vis - rgb_vis.min()) / (rgb_vis.max() - rgb_vis.min() + 1e-8)
                    heatmap_resized = cv2.resize(heatmap, (rgb_vis.shape[1], rgb_vis.shape[0]))
                    heatmap_resized = np.nan_to_num(heatmap_resized, nan=0.0)
                    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
                    heatmap_color = (np.float32(heatmap_color) / 255.0)[:, :, ::-1]
                    overlay = 0.5 * rgb_vis + 0.5 * heatmap_color
                    writer.add_image("Explainability/GradCAM_Defect_Focus", overlay.transpose(2, 0, 1), epoch)
                    logged_image = True

        avg_val_loss = val_loss_sum / max(1, len(val_loader))
        avg_val_miou = val_iou.get_miou()
        writer.add_scalar("Loss/Validation", avg_val_loss, epoch)
        writer.add_scalar("Metrics/mIoU_Validation", avg_val_miou, epoch)

        if avg_val_miou > state.get("best_miou", -1.0):
            state["best_miou"] = avg_val_miou
            state["best_train_loss"] = avg_train_loss
            state["best_val_loss"] = avg_val_loss
            patience_counter = 0
            atomic_torch_save(model.state_dict(), os.path.join(run_dir, "best_model.pt"))
        else:
            patience_counter += 1

        state["patience_counter"] = patience_counter

        atomic_torch_save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_miou': state["best_miou"],
            'best_train_loss': state["best_train_loss"],
            'best_val_loss': state["best_val_loss"],
            'patience_counter': patience_counter
        }, os.path.join(run_dir, "checkpoint.pt"))

        with open(os.path.join(run_dir, "state.json"), 'w') as f:
            json.dump(state, f, indent=4)

        if patience_counter >= early_stop_patience:
            print(f"\n[*] Active Early Stopping Triggered at Epoch {epoch + 1} (Best Val mIoU: {state['best_miou']:.4f}).")
            break

    cam_extractor.remove_hooks()
    writer.close()

    results_payload = {
        "phase": phase,
        "hyperparameters": active_hparams,
        "val_mIoU_checkpoint": state.get("best_miou", 0.0),
        "best_mIoU": state.get("best_miou", 0.0),
        "final_train_loss": state.get("best_train_loss", avg_train_loss),
        "final_val_loss": state.get("best_val_loss", avg_val_loss),
        "last_epoch_train_loss": avg_train_loss,
        "last_epoch_val_loss": avg_val_loss,
    }

    # Preserve blind test-set isolation: evaluate test.csv strictly at the final converged Microtune phase
    if phase == "microtune":
        print("\n[*] Microtune complete. Loading best validation checkpoint for Blind Test Set TTA Evaluation...")
        best_model_path = os.path.join(run_dir, "best_model.pt")
        if os.path.exists(best_model_path):
            model.load_state_dict(torch.load(best_model_path, map_location=DEVICE, weights_only=True), strict=True)

        test_dataset_tta = TTA_DownstreamSegmentationDataset(
            dataset_name=dataset_name,
            split="test",
            splits_root=splits_root,
            image_size=img_size
        )
        test_loader_tta = DataLoader(test_dataset_tta, batch_size=1, shuffle=False, num_workers=4)

        model.eval()
        base_test_iou = IoUMetric(NUM_CLASSES, DEVICE)
        tta_test_iou = IoUMetric(NUM_CLASSES, DEVICE)

        with torch.no_grad():
            for batch in tqdm(test_loader_tta, desc="Evaluating Blind Test TTA Multi-Views"):
                x_tta_batch = batch['x_tta_batch'].to(DEVICE, non_blocking=True)
                seg_mask = batch['seg_mask'].to(DEVICE, non_blocking=True)

                x_base = x_tta_batch[:, 0]
                x_hflip = x_tta_batch[:, 1]

                with autocast('cuda', dtype=torch.bfloat16, enabled=use_bf16):
                    pred_base = model(x_base)
                    pred_hflip = model(x_hflip)

                pred_hflip_inv = torch.flip(pred_hflip, dims=[3])
                pred_tta = (pred_base + pred_hflip_inv) / 2.0

                base_test_iou.update(pred_base, seg_mask)
                tta_test_iou.update(pred_tta, seg_mask)

        final_base_miou = base_test_iou.get_miou()
        final_tta_miou = tta_test_iou.get_miou()
        print(f"[*] Blind Test Base mIoU: {final_base_miou:.4f} | Blind Test TTA mIoU: {final_tta_miou:.4f}")

        results_payload.update({
            "best_mIoU": final_tta_miou,
            "test_tta_mIoU": final_tta_miou,
            "base_test_mIoU": final_base_miou,
        })

    with open(os.path.join(run_dir, "results.json"), 'w') as f:
        json.dump(results_payload, f, indent=4)


if __name__ == '__main__':
    main()