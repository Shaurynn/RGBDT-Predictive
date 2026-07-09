import os
import glob
import json
import inspect
import torch
import cv2
import optuna
import argparse
import datetime
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from torchinfo import summary
from tqdm import tqdm

from dataset import TriModalPredictiveDataset
import models  

# ====================================================================================
# --- 1. CUSTOM LOSSES & ARCHITECTURE METRICS ---
# ====================================================================================

class FocalDiceLoss(nn.Module):
    """
    Evaluates spatial boundaries and heavily penalizes minority class failures.
    Integrates a Dynamic Class-Weighting (DCW) schedule to scale the dice penalty.
    """
    def __init__(self, num_classes, gamma=2.0, dice_weight=1.0, ignore_index=255):
        super().__init__()
        self.num_classes = num_classes
        self.gamma = gamma
        self.base_dice_weight = dice_weight
        self.ignore_index = ignore_index
        
        # Base CE weights (Suppresses background dominance)
        self.register_buffer('ce_weights', torch.ones(num_classes))
        self.ce_weights[0] = 0.1 
        
        # Dynamic Dice weights initialized to 1.0
        self.register_buffer('dynamic_dice_weights', torch.ones(num_classes))

    def update_dynamic_weights(self, per_class_iou, momentum=0.9, tau=2.0):
        """
        Dynamic Class-Weighting Schedule (DCW).
        Exponentially scales the dice penalty for minority/hard classes based on validation IoU.
        Uses Exponential Moving Average (EMA) to prevent gradient shock.
        """
        per_class_iou = per_class_iou.to(self.dynamic_dice_weights.device)
        
        target_weights = torch.exp(tau * (1.0 - per_class_iou))
        target_weights[0] = 1.0 # Prevent background weight inflation
        
        self.dynamic_dice_weights = (momentum * self.dynamic_dice_weights) + ((1.0 - momentum) * target_weights)
        self.dynamic_dice_weights = torch.clamp(self.dynamic_dice_weights, min=1.0, max=10.0)

    def forward(self, inputs, targets):
        # UPCAST & CLAMP: Prevent extreme logit spreads from deep architectures
        inputs = torch.clamp(inputs.float(), min=-100.0, max=100.0)
        
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', ignore_index=self.ignore_index, weight=self.ce_weights)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()

        valid_mask = (targets != self.ignore_index)
        valid_inputs = inputs.permute(0, 2, 3, 1)[valid_mask] 
        valid_targets = targets[valid_mask]                   
        
        if valid_targets.numel() == 0:
            dice_loss = torch.tensor(0.0, device=inputs.device, requires_grad=True)
        else:
            inputs_soft = F.softmax(valid_inputs.float(), dim=1)
            targets_one_hot = F.one_hot(valid_targets, num_classes=self.num_classes).float()
            
            intersection = (inputs_soft * targets_one_hot).sum(dim=0)
            denominator = inputs_soft.sum(dim=0) + targets_one_hot.sum(dim=0)
            dice_raw = (2.0 * intersection + 1e-5) / (denominator + 1e-5) 
            
            present_classes = targets_one_hot.sum(dim=0) > 0
            if present_classes.sum() > 0:
                weighted_dice = (1.0 - dice_raw[present_classes]) * self.dynamic_dice_weights[present_classes]
                dice_loss = weighted_dice.mean() / (self.dynamic_dice_weights[present_classes].mean() + 1e-8)
            else:
                dice_loss = torch.tensor(0.0, device=inputs.device, requires_grad=True)

        return focal_loss + (self.base_dice_weight * dice_loss)

def masked_mse_loss(preds, targets, mask):
    """Legacy Generative Loss for Pixel-Space Reconstruction."""
    # UPCAST & CLAMP FIX
    preds = torch.clamp(preds.float(), min=-1000.0, max=1000.0)
    targets = torch.clamp(targets.float(), min=-1000.0, max=1000.0)
    
    diff = (preds - targets) ** 2
    return (diff * mask.float()).sum() / (mask.float().sum() + 1e-8)

class LatentRegularizationLoss(nn.Module):
    """
    The mathematically stabilized VICReg Triad.
    Physically prevents Representation Collapse without using an EMA teacher.
    """
    def __init__(self, sim_weight=25.0, var_weight=25.0, cov_weight=0.01, eps=1e-4):
        super().__init__()
        self.sim_weight = sim_weight
        self.var_weight = var_weight
        self.cov_weight = cov_weight # Reduced to standard academic ranges (0.01)
        self.eps = eps

    def off_diagonal(self, x):
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    def forward(self, z_pred, z_target):
        B, C, H, W = z_pred.shape
        N = B * H * W
        
        # Flatten spatial dimensions: [N, C]
        z_pred = z_pred.permute(0, 2, 3, 1).reshape(N, C)
        z_target = z_target.permute(0, 2, 3, 1).reshape(N, C)

        # 1. Invariance (Similarity) Loss
        sim_loss = F.mse_loss(z_pred, z_target)

        # 2. Mathematical Bounding via LayerNorm
        # Explicitly prevents exponential variance, allowing us to remove torch.clamp()
        z_pred_norm = F.layer_norm(z_pred, (C,))
        z_target_norm = F.layer_norm(z_target, (C,))

        # Mean-centering
        z_pred_centered = z_pred_norm - z_pred_norm.mean(dim=0)
        z_target_centered = z_target_norm - z_target_norm.mean(dim=0)

        # 3. Variance Loss
        std_pred = torch.sqrt(z_pred_centered.var(dim=0) + self.eps)
        std_target = torch.sqrt(z_target_centered.var(dim=0) + self.eps)
        var_loss = torch.mean(F.relu(1.0 - std_pred)) + torch.mean(F.relu(1.0 - std_target))

        # 4. Covariance Loss (Decorrelator)
        cov_pred = (z_pred_centered.T @ z_pred_centered) / (N - 1)
        cov_target = (z_target_centered.T @ z_target_centered) / (N - 1)
        
        cov_loss = (self.off_diagonal(cov_pred).pow(2).sum() / C) + \
                   (self.off_diagonal(cov_target).pow(2).sum() / C)

        return (self.sim_weight * sim_loss) + (self.var_weight * var_loss) + (self.cov_weight * cov_loss)

def knowledge_distillation_loss(student_logits, teacher_logits, temperature=4.0):
    """
    Computes the Kullback-Leibler (KL) Divergence between the Student and Teacher soft probabilities.
    Extracts 'Dark Knowledge' (inter-class relationships and noise suppression) from the teacher.
    """
    soft_targets = F.softmax(teacher_logits / temperature, dim=1)
    student_log_probs = F.log_softmax(student_logits / temperature, dim=1)
    kd_loss = F.kl_div(student_log_probs, soft_targets, reduction='batchmean') * (temperature ** 2)
    return kd_loss

# ====================================================================================
# --- 2. ADVANCED ROBUSTNESS EVALUATION SUITE ---
# ====================================================================================

class IoUMetric:
    def __init__(self, num_classes, device):
        self.num_classes = num_classes
        self.device = device
        self.intersections = torch.zeros(num_classes, device=device)
        self.unions = torch.zeros(num_classes, device=device)
        
    def update(self, logits, targets, ignore_index=255):
        preds = torch.argmax(logits, dim=1)
        valid_mask = targets != ignore_index
        preds = preds[valid_mask]
        targets = targets[valid_mask]
        
        if targets.numel() == 0:
            return
            
        bins = targets * self.num_classes + preds
        bincount = torch.bincount(bins, minlength=self.num_classes**2)
        conf_matrix = bincount.reshape(self.num_classes, self.num_classes)
        
        intersection = torch.diag(conf_matrix)
        union = conf_matrix.sum(dim=1) + conf_matrix.sum(dim=0) - intersection
        
        self.intersections += intersection
        self.unions += union
        
    def get_per_class_iou(self):
        ious = self.intersections / torch.clamp(self.unions, min=1.0)
        ious[self.unions == 0] = 1.0 
        return ious
        
    def get_miou(self):
        valid_classes = self.unions > 0
        if valid_classes.sum() == 0: return 0.0
        return self.get_per_class_iou()[valid_classes].mean().item()

def compute_ece(logits, targets, num_bins=10, ignore_index=255):
    """Expected Calibration Error (ECE) metric."""
    preds = torch.argmax(logits, dim=1)
    confs = torch.max(F.softmax(logits, dim=1), dim=1)[0]
    valid_mask = targets != ignore_index
    preds, confs, targets = preds[valid_mask], confs[valid_mask], targets[valid_mask]
    
    ece = 0.0
    bin_boundaries = torch.linspace(0, 1, num_bins + 1, device=logits.device)
    
    for i in range(num_bins):
        in_bin = (confs > bin_boundaries[i]) & (confs <= bin_boundaries[i+1])
        if in_bin.sum() > 0:
            acc_in_bin = (preds[in_bin] == targets[in_bin]).float().mean()
            avg_conf_in_bin = confs[in_bin].mean()
            ece += torch.abs(avg_conf_in_bin - acc_in_bin) * (in_bin.sum().float() / preds.numel())
    return ece.item()

def compute_boundary_iou(logits, targets, num_classes, ignore_index=255, dilation_kernel=5):
    """Evaluates strict perimeter adherence rather than overall volumetric mass."""
    preds = torch.argmax(logits, dim=1).unsqueeze(1).float()
    targets_f = targets.unsqueeze(1).float()
    kernel = torch.ones((1, 1, dilation_kernel, dilation_kernel), device=logits.device)
    
    pred_dilated = F.conv2d(preds, kernel, padding=dilation_kernel//2).clamp(0, 1)
    pred_eroded = -F.conv2d(-preds, kernel, padding=dilation_kernel//2).clamp(0, 1)
    pred_boundary = (pred_dilated - pred_eroded) > 0
    
    target_dilated = F.conv2d(targets_f, kernel, padding=dilation_kernel//2).clamp(0, 1)
    target_eroded = -F.conv2d(-targets_f, kernel, padding=dilation_kernel//2).clamp(0, 1)
    target_boundary = (target_dilated - target_eroded) > 0
    
    intersection = (pred_boundary & target_boundary).sum().item()
    union = (pred_boundary | target_boundary).sum().item()
    return intersection / max(union, 1e-8)

def apply_ood_noise(tensor, noise_std=0.3):
    """Applies Out-of-Distribution (OOD) structural noise for resilience testing."""
    noise = torch.randn_like(tensor) * noise_std
    return torch.clamp(tensor + noise, 0, 1)

class TTAWrapper(nn.Module):
    """
    Test-Time Augmentation (TTA) Wrapper.
    Evaluates spatial hesitation and epistemic uncertainty by measuring variance
    across multiple geometric orientations.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, rgbd, therm):
        out_orig = self.model(rgbd, therm)
        out_orig = out_orig[0] if isinstance(out_orig, tuple) else out_orig

        out_hf = self.model(torch.flip(rgbd, dims=[3]), torch.flip(therm, dims=[3]))
        out_hf = out_hf[0] if isinstance(out_hf, tuple) else out_hf
        out_hf = torch.flip(out_hf, dims=[3]) 

        out_vf = self.model(torch.flip(rgbd, dims=[2]), torch.flip(therm, dims=[2]))
        out_vf = out_vf[0] if isinstance(out_vf, tuple) else out_vf
        out_vf = torch.flip(out_vf, dims=[2]) 

        preds_stack = torch.stack([out_orig, out_hf, out_vf], dim=0)
        return preds_stack.mean(dim=0), preds_stack.var(dim=0).mean(dim=1)

# ====================================================================================
# --- 3. EXPLAINABILITY ENGINE ---
# ====================================================================================

class SemanticGradCAM:
    """Generates high-resolution class-activation heatmaps anchored to the fusion layers."""
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients, self.activations = None, None
        self._register_hooks()

    def _register_hooks(self):
        if self.target_layer is None: return
        def forward_hook(m, i, o): self.activations = o
        def backward_hook(m, gi, go): self.gradients = go[0]
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate_heatmap(self, rgbd_tensor, therm_tensor, target_class):
        self.model.zero_grad()
        outputs = self.model(rgbd_tensor, therm_tensor)
        logits = outputs if not isinstance(outputs, tuple) else outputs[0]
        
        class_mask = logits[:, target_class, :, :]
        class_mask.sum().backward(retain_graph=True)

        if self.gradients is None or self.activations is None:
            return torch.zeros((rgbd_tensor.shape[2], rgbd_tensor.shape[3]))

        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1).squeeze().detach().cpu().numpy()
        cam = np.maximum(cam, 0)
        if cam.max() > 0: cam = cam / cam.max()
        return cv2.resize(cam, (rgbd_tensor.shape[3], rgbd_tensor.shape[2]))

# ====================================================================================
# --- 4. PIPELINE MANAGEMENT & HPO ---
# ====================================================================================

class ExperimentManager:
    """Controls the sequential state machine for training lifecycle (baseline -> export)."""
    def __init__(self, model_instance, backbone="mit_b1", data_domain="MM5", base_dir="results"):
        self.model_name = model_instance.__class__.__name__
        self.data_domain = data_domain
        self.backbone = backbone
        self.model_dir = os.path.join(base_dir, self.model_name, self.backbone, self.data_domain)
        os.makedirs(self.model_dir, exist_ok=True)
        self.phase_sequence = ["baseline", "hpo", "hero", "microtune", "export"]
        
    def detect_state(self):
        existing_runs = sorted(glob.glob(os.path.join(self.model_dir, "*_*")))
        if not existing_runs: return self._create_new_run("baseline", resume_from=None)
            
        latest_run = existing_runs[-1]
        run_name = os.path.basename(latest_run)
        
        if os.path.exists(os.path.join(latest_run, "results.json")):
            current_phase = run_name.split("_")[-1].lower()
            if current_phase == self.phase_sequence[-1]: return None
                
            next_phase_idx = self.phase_sequence.index(current_phase) + 1
            next_phase = self.phase_sequence[next_phase_idx]
            
            if next_phase == "hpo": inherit_weights = os.path.join(latest_run, "best_model.pt") 
            elif next_phase == "hero":
                baseline_runs = [r for r in existing_runs if r.lower().endswith("_baseline")]
                inherit_weights = os.path.join(baseline_runs[-1], "best_model.pt") if baseline_runs else None
            else: 
                hero_runs = [r for r in existing_runs if r.lower().endswith("_hero")]
                inherit_weights = os.path.join(hero_runs[-1], "best_model.pt") if hero_runs else None
                
            return self._create_new_run(next_phase, resume_from=inherit_weights)
        else: return self._resume_run(latest_run)

    def _create_new_run(self, phase, resume_from):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(self.model_dir, f"{timestamp}_{phase.capitalize()}")
        os.makedirs(run_dir); os.makedirs(os.path.join(run_dir, "weights")); os.makedirs(os.path.join(run_dir, "logs"))
        if phase != "hpo": os.makedirs(os.path.join(run_dir, "explainability")) 
        
        state = {"run_dir": run_dir, "phase": phase, "is_resume": False, "inherit_weights": resume_from, "start_epoch": 0, "best_miou": 0.0, "patience_counter": 0}
        self._save_state(run_dir, state)
        return state

    def _resume_run(self, run_dir):
        with open(os.path.join(run_dir, "state.json"), 'r') as f: state = json.load(f)
        state["is_resume"] = True
        return state
        
    def _save_state(self, run_dir, state_dict):
        with open(os.path.join(run_dir, "state.json"), 'w') as f: json.dump(state_dict, f, indent=4)

def build_phase_config(phase, model, max_epochs, model_dir, num_classes, is_latent_model,
                       custom_lr=None, custom_wd=None, custom_sim=25.0, custom_var=25.0, custom_cov=15.0):
    """Constructs the exact optimizer, scheduler, and loss environment for the active phase."""
    
    # FIX: Lowered AdamW learning rate from 0.0753 to 0.00006 for ViT stability
    lr, gamma, dice, opt_type, momentum, weight_decay = 0.00006, 1.6627, 0.6250, "AdamW", 0.9685, 0.0003
    
    # Apply direct JSON overrides (crucial for Knowledge Distillation specific tuning)
    if custom_lr is not None: lr = custom_lr
    if custom_wd is not None: weight_decay = custom_wd
    
    alpha = 0.1 if is_latent_model else 0.5 
    beta = 0.4 
    
    # Ingest HPO findings if progressing through standard state machine
    if phase in ["hero", "microtune"]:
        existing_runs = sorted(glob.glob(os.path.join(model_dir, "*_*")))
        hpo_runs = [r for r in existing_runs if r.lower().endswith("_hpo")]
        if hpo_runs:
            params_path = os.path.join(hpo_runs[-1], "best_params.json")
            if os.path.exists(params_path):
                with open(params_path, "r") as f: p = json.load(f)
                lr = p.get("lr", lr); gamma = p.get("gamma", gamma); dice = p.get("dice_weight", dice)
                opt_type = p.get("optimizer", opt_type); momentum = p.get("sgd_momentum", momentum)
                weight_decay = p.get("weight_decay", weight_decay); alpha = p.get("alpha", alpha)
                if not is_latent_model: beta = p.get("beta", beta)

    criterion = FocalDiceLoss(num_classes=num_classes, ignore_index=255, gamma=gamma, dice_weight=dice)
    latent_criterion = LatentRegularizationLoss(sim_weight=custom_sim, var_weight=custom_var, cov_weight=custom_cov)

    if phase == "baseline":
        opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_epochs, eta_min=1e-6)
    elif phase == "hero":
        if opt_type == "AdamW": opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        else: opt = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum, nesterov=True)
        sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_epochs, eta_min=1e-6)
    elif phase == "microtune":
        micro_lr = min(lr * 0.1, 1e-5) 
        if opt_type == "AdamW": opt = optim.AdamW(model.parameters(), lr=micro_lr, weight_decay=weight_decay)
        else: opt = optim.SGD(model.parameters(), lr=micro_lr, weight_decay=weight_decay, momentum=0.9, nesterov=True)
        sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_epochs, eta_min=1e-7)
        
    return opt, sch, criterion, latent_criterion, alpha, beta

def run_hpo_phase(run_dir, inherit_weights, ModelClass, model_kwargs, train_loader, eval_loader, num_classes, device):
    """Executes a 30-Trial Bayesian Hyperparameter Optimization Sweep via Optuna."""
    print("\n" + "="*75)
    print("🚀 INITIATING OPTUNA HYPERPARAMETER SWEEP (30 Trials)")
    study_db_path = os.path.join(run_dir, "optuna_study.db")
    print(f"📊 [MONITORING] run: optuna-dashboard sqlite:///{study_db_path}")
    print("="*75 + "\n")
    
    is_latent_model = (ModelClass.__name__ == "TriModalLatentPredictiveNetwork")
    
    def objective(trial):
        model = ModelClass(**model_kwargs).to(device)
        if inherit_weights and os.path.exists(inherit_weights): model.load_state_dict(torch.load(inherit_weights))
            
        lr = trial.suggest_float("lr", 1e-5, 5e-3, log=True)
        wd = trial.suggest_float("weight_decay", 1e-4, 1e-1, log=True)
        gamma = trial.suggest_float("gamma", 1.0, 4.0)
        dice = trial.suggest_float("dice_weight", 0.5, 2.0)
        mask_ratio = trial.suggest_float("mask_ratio", 0.15, 0.65)
        
        if is_latent_model:
            alpha = trial.suggest_float("alpha", 0.05, 0.5)
            beta = 0.0
        else:
            alpha = trial.suggest_float("alpha", 0.1, 1.5)
            beta = trial.suggest_float("beta", 0.1, 1.0)
            
        train_loader.dataset.mask_ratio = mask_ratio 
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
        
        # HPO FIX: Explicitly push dynamically created loss modules to the device
        criterion = FocalDiceLoss(num_classes, gamma=gamma, dice_weight=dice).to(device)
        latent_criterion = LatentRegularizationLoss().to(device)

        scaler = GradScaler(device.type, enabled=True)
        
        best_miou = 0.0
        for epoch in range(30): 
            model.train()
            for batch in train_loader:
                rgbd, therm_masked = batch['rgbd'].to(device), batch['therm_masked'].to(device)
                therm_target, seg_mask, block_mask = batch['therm_target'].to(device), batch['seg_mask'].to(device), batch['block_mask'].to(device)

                optimizer.zero_grad()
                # Mixed-Precision Engine
                with autocast(device_type=device.type, dtype=torch.bfloat16):
                    if is_latent_model:
                        pred_seg, z_pred, z_target = model(rgbd, therm_masked, therm_target, block_mask)
                        loss_seg = criterion(pred_seg, seg_mask)
                        # FIX: The updated LatentRegularizationLoss now returns a single stabilized scalar
                        loss_phys = latent_criterion(z_pred, z_target)
                        loss = loss_seg + (alpha * loss_phys)
                    else:
                        # Legacy Generative TMPN logic (preserved)
                        pred_seg, pred_therm, aux_therm_seg = model(rgbd, therm_masked)
                        loss_seg = criterion(pred_seg, seg_mask)
                        loss_phys = masked_mse_loss(pred_therm, therm_target, block_mask)
                        loss = loss_seg + (alpha * loss_phys) + (beta * criterion(aux_therm_seg, seg_mask))
                        
                # Standard scaled backpropagation
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            
            model.eval()
            val_iou_tracker = IoUMetric(num_classes, device)
            batches = 0
            with torch.no_grad():
                for batch in eval_loader:
                    rgbd, therm_masked, seg_mask = batch['rgbd'].to(device), batch['therm_masked'].to(device), batch['seg_mask'].to(device)
                    # FIX: Set dtype to torch.bfloat16
                    with autocast(device_type=device.type, dtype=torch.bfloat16):
                        # By passing only rgbd and therm_masked, therm_target defaults to None, 
                        # cleanly bypassing the Target Encoder during inference.
                        outputs = model(rgbd, therm_masked)
                        logits = outputs if not isinstance(outputs, tuple) else outputs[0]
                    val_iou_tracker.update(logits, seg_mask)
                    batches += 1
            
            score = val_iou_tracker.get_miou()
            if score > best_miou: best_miou = score
            trial.report(score, epoch)
            if trial.should_prune(): raise optuna.exceptions.TrialPruned()
        return best_miou

    study = optuna.create_study(direction="maximize", storage=f"sqlite:///{study_db_path}", pruner=optuna.pruners.HyperbandPruner())
    study.optimize(objective, n_trials=30)
    with open(os.path.join(run_dir, "best_params.json"), "w") as f: json.dump(study.best_params, f, indent=4)
    return study.best_value

def export_to_onnx(model, weights_path, run_dir, device):
    """Compiles the fully optimized network down to ONNX Opset 18 for edge TensorRT deployment."""
    print(f"\n--- Serializing Architecture to ONNX ---")
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()

    dummy_rgbd = torch.randn(1, 4, 480, 640, device=device)
    dummy_therm = torch.randn(1, 1, 480, 640, device=device)
    export_dir = os.path.join(run_dir, "deployment")
    os.makedirs(export_dir, exist_ok=True)
    onnx_path = os.path.join(export_dir, f"{model.__class__.__name__}.onnx")

    with torch.no_grad():
        torch.onnx.export(
            model, (dummy_rgbd, dummy_therm), onnx_path,
            export_params=True, opset_version=18, do_constant_folding=True,
            input_names=['input_rgbd', 'input_therm'],
            output_names=['output_mask'],
            dynamic_axes={'input_rgbd': {0: 'batch_size'}, 'input_therm': {0: 'batch_size'}, 'output_mask': {0: 'batch_size'}}
        )
    print(f"[SUCCESS] ONNX graph serialized to: {onnx_path}")
    return onnx_path

# ====================================================================================
# --- 5. MAIN EXECUTION ENGINE ---
# ====================================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="TriModalPredictiveNetwork")
    parser.add_argument("--backbone", type=str, default="mit_b1", help="Vision Transformer backbone")
    parser.add_argument("--params", type=str, default="{}")
    parser.add_argument("--data_domain", type=str, default="MM5")
    parser.add_argument("--data_dir", type=str, default="dataset/MM5")
    parser.add_argument("--disable_tta", action="store_true", help="Skip TTA computation in diagnostics")
    
    # Knowledge Distillation Defaults (Overridable via JSON config)
    parser.add_argument("--teacher_backbone", type=str, default=None)
    parser.add_argument("--teacher_weights", type=str, default=None)
    parser.add_argument("--kd_temp", type=float, default=4.0)
    parser.add_argument("--kd_alpha", type=float, default=0.5)
    
    args = parser.parse_args()
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = 6 

    ModelClass = getattr(models, args.model)
    model_kwargs = {"num_classes": 10, "backbone_name": args.backbone} # NUM_CLASSES default init
    
    # -------------------------------------------------------------------------
    # JSON OVERRIDE & PARAMETER ROUTING
    # -------------------------------------------------------------------------
    if os.path.exists(args.params):
        with open(args.params, 'r') as f:
            config_params = json.load(f)
            
        # 1. Override execution args with JSON variables
        if "teacher_backbone" in config_params: args.teacher_backbone = config_params["teacher_backbone"]
        if "teacher_weights" in config_params: args.teacher_weights = config_params["teacher_weights"]
        if "kd_temp" in config_params: args.kd_temp = float(config_params["kd_temp"])
        if "kd_alpha" in config_params: args.kd_alpha = float(config_params["kd_alpha"])
        
        if "max_epoch" in config_params: args.max_epoch = int(config_params["max_epoch"])
        if "patience" in config_params: args.patience = int(config_params["patience"])
        if "learning_rate" in config_params: args.learning_rate = float(config_params["learning_rate"])
        if "weight_decay" in config_params: args.weight_decay = float(config_params["weight_decay"])
        if "sim_weight" in config_params: args.sim_weight = float(config_params["sim_weight"])
        if "var_weight" in config_params: args.var_weight = float(config_params["var_weight"])
        if "cov_weight" in config_params: args.cov_weight = float(config_params["cov_weight"])

        # 2. Safely filter Model kwargs using inspect to prevent constructor crashes
        valid_keys = inspect.signature(ModelClass.__init__).parameters.keys()
        for k, v in config_params.items():
            if k in valid_keys:
                model_kwargs[k] = v
                
        # 3. Safely isolate routing configuration (e.g. lane_id)
        current_lane = config_params.get("lane_id", "default_lane")
    else:
        current_lane = "default_lane"

    # Dataset Initialization
    train_dataset = TriModalPredictiveDataset(data_dir=args.data_dir, split="train", mask_ratio=config_params.get("mask_ratio", 0.30) if os.path.exists(args.params) else 0.30)
    eval_dataset = TriModalPredictiveDataset(data_dir=args.data_dir, split="eval", mask_ratio=0.0)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    eval_loader = DataLoader(eval_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    NUM_CLASSES = train_dataset.num_classes
    model_kwargs["num_classes"] = NUM_CLASSES

    # Model Instantiation
    model = ModelClass(**model_kwargs).to(DEVICE)
    is_latent_model = (args.model == "TriModalLatentPredictiveNetwork")
    
    # --------------------------------------------------------------------------------
    # KNOWLEDGE DISTILLATION TEACHER INSTANTIATION
    # --------------------------------------------------------------------------------
    teacher_model = None
    if args.teacher_weights and os.path.exists(args.teacher_weights):
        print(f"\n[INFO] Loading Teacher Model ({args.teacher_backbone}) for Knowledge Distillation...")
        teacher_kwargs = {"num_classes": NUM_CLASSES, "backbone_name": args.teacher_backbone}
        teacher_model = ModelClass(**teacher_kwargs).to(DEVICE)
        teacher_model.load_state_dict(torch.load(args.teacher_weights, map_location=DEVICE))
        teacher_model.eval()
        for param in teacher_model.parameters():
            param.requires_grad = False
    
    # --------------------------------------------------------------------------------
    # KERAS-STYLE TOPOLOGY REPORT
    # --------------------------------------------------------------------------------
    print("\n" + "="*75)
    print("🧠 ARCHITECTURE TOPOLOGY")
    print("="*75)
    summary(model, input_size=[(BATCH_SIZE, 4, 480, 640), (BATCH_SIZE, 1, 480, 640)], col_names=["input_size", "output_size", "num_params", "mult_adds"], depth=4)
    print("="*75)
    
    # --------------------------------------------------------------------------------
    # EXPERIMENT ROUTING & STATE MACHINE
    # --------------------------------------------------------------------------------
    # Appends the lane_id to isolate specific timelines (e.g. distillation) from the standard path
    experiment_id = f"{args.backbone}_{current_lane}" if current_lane != "default_lane" else args.backbone
    
    manager = ExperimentManager(model_instance=model, backbone=experiment_id, data_domain=args.data_domain)
    state = manager.detect_state()
    
    if state is None: 
        print(f"\n[SYSTEM] State machine exited. Pipeline for {experiment_id} is already complete.")
        return 
        
    run_dir, phase = state["run_dir"], state["phase"]

    # Trigger distinct phase behaviors
    if phase == "hpo":
        best_score = run_hpo_phase(run_dir, state["inherit_weights"], ModelClass, model_kwargs, train_loader, eval_loader, NUM_CLASSES, DEVICE)
        with open(os.path.join(run_dir, "results.json"), 'w') as f: json.dump({"phase": phase, "best_hpo_mIoU": best_score}, f, indent=4)
        return

    if phase == "export":
        onnx_file = export_to_onnx(model, state["inherit_weights"], run_dir, DEVICE)
        with open(os.path.join(run_dir, "results.json"), 'w') as f: json.dump({"phase": phase, "artifact": onnx_file}, f, indent=4)
        return

    # Apply Hyperparameter Overrides
    MAX_EPOCHS = getattr(args, 'max_epoch', 150 if phase == "baseline" else (300 if phase == "hero" else 200))
    PATIENCE = getattr(args, 'patience', 25 if phase == "baseline" else 40)

    print(f"\n🚀 PHASE: {phase.upper()} | EPOCHS: {MAX_EPOCHS} | PATIENCE: {PATIENCE}")
    print(f"📈 [MONITORING] run: tensorboard --logdir={os.path.join(run_dir, 'logs')}\n")

    optimizer, scheduler, criterion, latent_criterion, alpha, beta = build_phase_config(
        phase, model, MAX_EPOCHS, manager.model_dir, NUM_CLASSES, is_latent_model,
        custom_lr=getattr(args, 'learning_rate', None),
        custom_wd=getattr(args, 'weight_decay', None),
        custom_sim=getattr(args, 'sim_weight', 25.0),
        custom_var=getattr(args, 'var_weight', 25.0),
        custom_cov=getattr(args, 'cov_weight', 15.0)
    )
    
    # FIX: Disable GradScaler when using bfloat16 to prevent artificial gradient explosion
    scaler = GradScaler(DEVICE.type, enabled=False)

    criterion = criterion.to(DEVICE)
    latent_criterion = latent_criterion.to(DEVICE)
    
    if state["is_resume"]:
        checkpoint = torch.load(os.path.join(run_dir, "latest_checkpoint.pt"))
        model.load_state_dict(checkpoint['model_state']); optimizer.load_state_dict(checkpoint['optimizer_state'])
        scheduler.load_state_dict(checkpoint['scheduler_state']); scaler.load_state_dict(checkpoint['scaler_state'])
    elif state["inherit_weights"]:
        model.load_state_dict(torch.load(state["inherit_weights"]))

    writer = SummaryWriter(log_dir=os.path.join(run_dir, "logs"))

    # --------------------------------------------------------------------------------
    # CORE TRAINING LOOP
    # --------------------------------------------------------------------------------
    for epoch in range(state["start_epoch"], MAX_EPOCHS):
        model.train()
        train_loss, train_seg_loss, train_physics_loss, train_kd_loss = 0.0, 0.0, 0.0, 0.0
        train_sim, train_var, train_cov = 0.0, 0.0, 0.0 
        
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{MAX_EPOCHS} [Train]")
        for batch in loop:
            rgbd, therm_masked = batch['rgbd'].to(DEVICE), batch['therm_masked'].to(DEVICE)
            therm_target, seg_mask, block_mask = batch['therm_target'].to(DEVICE), batch['seg_mask'].to(DEVICE), batch['block_mask'].to(DEVICE)
            
            optimizer.zero_grad()
            # FIX: Set dtype to torch.bfloat16
            with autocast(device_type=DEVICE.type, dtype=torch.bfloat16):
                if is_latent_model:
                    pred_seg, z_pred, z_target = model(rgbd, therm_masked, therm_target)
                    loss_seg = criterion(pred_seg, seg_mask)
                    
                    loss_phys, sim, var, cov = latent_criterion(z_pred, z_target)
                    loss = loss_seg + (alpha * loss_phys)
                else:
                    pred_seg, pred_therm, aux_therm_seg = model(rgbd, therm_masked)
                    loss_seg = criterion(pred_seg, seg_mask)
                    loss_phys = masked_mse_loss(pred_therm, therm_target, block_mask)
                    loss = loss_seg + (alpha * loss_phys) + (beta * criterion(aux_therm_seg, seg_mask))
                
                # Apply Knowledge Distillation Soft Targets
                if teacher_model is not None:
                    with torch.no_grad():
                        teacher_outputs = teacher_model(rgbd, therm_masked)
                        # Standardize extraction to prevent KL Divergence crash
                        teacher_seg = teacher_outputs[0] if isinstance(teacher_outputs, tuple) else teacher_outputs
                    
                    loss_kd = knowledge_distillation_loss(pred_seg, teacher_seg, temperature=args.kd_temp)
                    loss = (1 - args.kd_alpha) * loss + (args.kd_alpha * loss_kd)
                    train_kd_loss += loss_kd.item()
                    
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            train_seg_loss += loss_seg.item()
            train_physics_loss += loss_phys.item()
            
            if is_latent_model:
                train_sim += sim.item()
                train_var += var.item()
                train_cov += cov.item()
                
            loop.set_postfix(loss=loss.item())
            
        scheduler.step()
        
        # --------------------------------------------------------------------------------
        # VALIDATION EVALUATION
        # --------------------------------------------------------------------------------
        model.eval()
        val_loss, batches = 0.0, 0
        val_iou_tracker = IoUMetric(NUM_CLASSES, DEVICE)
        with torch.no_grad():
            for batch in eval_loader:
                rgbd, therm_masked = batch['rgbd'].to(DEVICE), batch['therm_masked'].to(DEVICE)
                therm_target, seg_mask, block_mask = batch['therm_target'].to(DEVICE), batch['seg_mask'].to(DEVICE), batch['block_mask'].to(DEVICE)
                
                # FIX: Set dtype to torch.bfloat16
                with autocast(device_type=DEVICE.type, dtype=torch.bfloat16):
                    if is_latent_model:
                        pred_seg = model(rgbd, therm_masked) 
                        loss = criterion(pred_seg, seg_mask)
                    else:
                        pred_seg, pred_therm = model(rgbd, therm_masked)
                        loss = criterion(pred_seg, seg_mask) + (alpha * masked_mse_loss(pred_therm, therm_target, block_mask))
                    
                val_loss += loss.item()
                val_iou_tracker.update(pred_seg, seg_mask)
                batches += 1
                
        avg_val_loss = val_loss / batches
        avg_val_miou = val_iou_tracker.get_miou()
        
        # Apply DCW scaling during later phases
        if phase == "hero":
            criterion.update_dynamic_weights(val_iou_tracker.get_per_class_iou())
        
        writer.add_scalars("Loss/Total", {'Train': train_loss / len(train_loader), 'Validation': avg_val_loss}, epoch + 1)
        writer.add_scalar("Metrics/Validation_mIoU", avg_val_miou, epoch + 1)
        
        if teacher_model is not None:
            writer.add_scalar("Loss/Distillation", train_kd_loss / len(train_loader), epoch + 1)
        
        if is_latent_model:
            writer.add_scalars("Loss/Latent_Components", {
                'Similarity': train_sim / len(train_loader),
                'Variance': train_var / len(train_loader),
                'Covariance': train_cov / len(train_loader)
            }, epoch + 1)
            
        writer.flush() 
        
        if teacher_model is not None:
            print(f"Epoch {epoch+1} | Seg: {train_seg_loss/len(train_loader):.4f} | Phys: {train_physics_loss/len(train_loader):.4f} | KD: {train_kd_loss/len(train_loader):.4f} | Val mIoU: {avg_val_miou:.4f}")
        else:
            print(f"Epoch {epoch+1} | Seg: {train_seg_loss/len(train_loader):.4f} | Physics: {train_physics_loss/len(train_loader):.4f} | Val mIoU: {avg_val_miou:.4f}")

        state["start_epoch"] = epoch + 1
        torch.save({'model_state': model.state_dict(), 'optimizer_state': optimizer.state_dict(), 'scheduler_state': scheduler.state_dict(), 'scaler_state': scaler.state_dict()}, os.path.join(run_dir, "latest_checkpoint.pt"))

        if avg_val_miou > state["best_miou"]:
            state["best_miou"] = avg_val_miou
            state["patience_counter"] = 0
            torch.save(model.state_dict(), os.path.join(run_dir, "best_model.pt"))
        else:
            state["patience_counter"] += 1

        with open(os.path.join(run_dir, "state.json"), 'w') as f: json.dump(state, f, indent=4)
        if state["patience_counter"] >= PATIENCE: break

    writer.close()

    # ====================================================================================
    # --- 6. FINAL DIAGNOSTIC & ROBUSTNESS SUITE ---
    # ====================================================================================
    print("\n" + "="*75)
    print("🔬 INITIATING COMPREHENSIVE ROBUSTNESS DIAGNOSTICS")
    print("="*75)
    
    model.load_state_dict(torch.load(os.path.join(run_dir, "best_model.pt")))
    model.eval()
    tta_model = TTAWrapper(model)
    tta_model.eval()
    grad_cam = SemanticGradCAM(model, target_layer=model.fusion_head)
    
    metrics = {
        'base': {'miou': IoUMetric(NUM_CLASSES, DEVICE), 'ece': 0, 'bound': 0, 'ood': IoUMetric(NUM_CLASSES, DEVICE)},
        'tta': {'miou': IoUMetric(NUM_CLASSES, DEVICE), 'ece': 0, 'bound': 0, 'ood': IoUMetric(NUM_CLASSES, DEVICE)}
    }
    batches = 0
    
    for i, batch in enumerate(tqdm(eval_loader, desc="Diagnostic Pass")):
        rgbd, therm_masked, seg_mask = batch['rgbd'].to(DEVICE), batch['therm_masked'].to(DEVICE), batch['seg_mask'].to(DEVICE)
        
        # Base Evaluation
        # FIX: Set dtype to torch.bfloat16
        with torch.no_grad(), autocast(device_type=DEVICE.type, dtype=torch.bfloat16):
            outputs_base = model(rgbd, therm_masked)
            logits_base = outputs_base[0] if isinstance(outputs_base, tuple) else outputs_base
            
            outputs_ood = model(apply_ood_noise(rgbd), apply_ood_noise(therm_masked))
            logits_ood_base = outputs_ood[0] if isinstance(outputs_ood, tuple) else outputs_ood
            
            metrics['base']['miou'].update(logits_base, seg_mask)
            metrics['base']['ece'] += compute_ece(logits_base, seg_mask)
            metrics['base']['bound'] += compute_boundary_iou(logits_base, seg_mask, NUM_CLASSES)
            metrics['base']['ood'].update(logits_ood_base, seg_mask)

        var_map = None
        # TTA Evaluation (Only runs if the flag is NOT triggered)
        if not args.disable_tta:
            # FIX: Set dtype to torch.bfloat16
            with torch.no_grad(), autocast(device_type=DEVICE.type, dtype=torch.bfloat16):
                logits_tta, var_map = tta_model(rgbd, therm_masked)
                logits_ood_tta, _ = tta_model(apply_ood_noise(rgbd), apply_ood_noise(therm_masked))
                
                metrics['tta']['miou'].update(logits_tta, seg_mask)
                metrics['tta']['ece'] += compute_ece(logits_tta, seg_mask)
                metrics['tta']['bound'] += compute_boundary_iou(logits_tta, seg_mask, NUM_CLASSES)
                metrics['tta']['ood'].update(logits_ood_tta, seg_mask)
            
        batches += 1
        
        # Visual Artifact Generation
        if i < 5:
            predictions = torch.argmax(logits_base, dim=1)
            for b in range(rgbd.size(0)):
                rgbd_input = rgbd[b].unsqueeze(0)
                therm_input = therm_masked[b].unsqueeze(0)
                
                # Lifted RGB extraction to safely overlay Epistemic Uncertainty
                rgb_tensor = rgbd_input[0, :3, :, :].clone().cpu()
                mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                rgb_unnorm = torch.clamp((rgb_tensor * std) + mean, 0, 1)
                rgb_img = cv2.cvtColor((rgb_unnorm.permute(1, 2, 0).numpy() * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

                for cls in torch.unique(predictions[b]):
                    if cls == 0 or cls == 255: continue
                    heatmap = grad_cam.generate_heatmap(rgbd_input, therm_input, cls.item())
                    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
                    overlay = cv2.addWeighted(rgb_img, 0.6, heatmap_colored, 0.4, 0)
                    cv2.imwrite(os.path.join(run_dir, "explainability", f"batch{i}_img{b}_class{cls.item()}_gradcam.png"), overlay)
                    
                if var_map is not None:
                    uncert_map = var_map[b].cpu().numpy()
                    uncert_norm = np.uint8(255 * (uncert_map - uncert_map.min()) / (uncert_map.max() - uncert_map.min() + 1e-8))
                    uncert_colored = cv2.applyColorMap(uncert_norm, cv2.COLORMAP_INFERNO)
                    uncert_overlay = cv2.addWeighted(rgb_img, 0.6, uncert_colored, 0.4, 0)
                    cv2.imwrite(os.path.join(run_dir, "explainability", f"batch{i}_img{b}_epistemic_uncertainty.png"), uncert_overlay)

    for k in metrics.keys():
        metrics[k]['miou'] = metrics[k]['miou'].get_miou()
        metrics[k]['ood'] = metrics[k]['ood'].get_miou()
        metrics[k]['ece'] /= batches
        metrics[k]['bound'] /= batches

    print("\n" + "="*75)
    print(f"📊 ARCHITECTURAL ROBUSTNESS REPORT: {phase.upper()} PHASE")
    print("="*75)
    print(f"{'Metric':<25} | {'Baseline (No TTA)':<20} | {'TTA Wrapper':<20}")
    print("-" * 75)
    print(f"{'Standard mIoU (Higher ↑)':<25} | {metrics['base']['miou']:<20.4f} | {metrics['tta']['miou']:<20.4f}")
    print(f"{'Boundary IoU (Higher ↑)':<25} | {metrics['base']['bound']:<20.4f} | {metrics['tta']['bound']:<20.4f}")
    print(f"{'ECE (Lower ↓)':<25} | {metrics['base']['ece']:<20.4f} | {metrics['tta']['ece']:<20.4f}")
    print(f"{'OOD Stress mIoU (Higher ↑)':<25} | {metrics['base']['ood']:<20.4f} | {metrics['tta']['ood']:<20.4f}")
    print("="*75 + "\n")

    # State-Machine Handoff Logic
    results_payload = {
        "model_architecture": model.__class__.__name__,
        "phase": phase,
        "completed_at": datetime.datetime.now().isoformat(),
        "final_base_mIoU": metrics['base']['miou'],
        "final_tta_mIoU": metrics['tta']['miou']
    }
    
    with open(os.path.join(run_dir, "results.json"), 'w') as f:
        json.dump(results_payload, f, indent=4)
        
    print(f"\n[SUCCESS] Phase {phase.upper()} complete and recorded in results.json.")
    print("Run your training command again to automatically begin the next phase.\n")

if __name__ == '__main__':
    main()