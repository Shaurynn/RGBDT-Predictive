import os
import glob
import json
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
    def __init__(self, num_classes, gamma=2.0, dice_weight=1.0, ignore_index=255):
        super().__init__()
        self.num_classes = num_classes
        self.gamma = gamma
        self.dice_weight = dice_weight
        self.ignore_index = ignore_index

    def forward(self, inputs, targets):
        class_weights = torch.ones(self.num_classes, device=inputs.device)
        class_weights[0] = 0.1 

        ce_loss = F.cross_entropy(inputs, targets, reduction='none', ignore_index=self.ignore_index, weight=class_weights)
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
            dice = (2.0 * intersection + 1e-5) / (denominator + 1e-5) 
            
            present_classes = targets_one_hot.sum(dim=0) > 0
            if present_classes.sum() > 0:
                dice_loss = 1.0 - dice[present_classes].mean()
            else:
                dice_loss = torch.tensor(0.0, device=inputs.device, requires_grad=True)

        return focal_loss + (self.dice_weight * dice_loss)

def masked_mse_loss(preds, targets, mask):
    diff = (preds - targets) ** 2
    return (diff * mask).sum() / (mask.sum() + 1e-8)

class LatentRegularizationLoss(nn.Module):
    def __init__(self, sim_weight=25.0, var_weight=25.0, cov_weight=15.0):
        super().__init__()
        self.sim_weight = sim_weight
        self.var_weight = var_weight
        self.cov_weight = cov_weight

    def forward(self, z_pred, z_target):
        B, C, H, W = z_pred.shape
        x = z_pred.permute(0, 2, 3, 1).reshape(-1, C)
        y = z_target.permute(0, 2, 3, 1).reshape(-1, C)

        sim_loss = F.mse_loss(x, y)
        std_x = torch.sqrt(x.var(dim=0) + 1e-4)
        std_y = torch.sqrt(y.var(dim=0) + 1e-4)
        var_loss = torch.mean(F.relu(1 - std_x)) + torch.mean(F.relu(1 - std_y))

        x_centered = x - x.mean(dim=0)
        y_centered = y - y.mean(dim=0)
        cov_x = (x_centered.T @ x_centered) / (x.shape[0] - 1)
        cov_y = (y_centered.T @ y_centered) / (y.shape[0] - 1)
        
        cov_loss = (self.off_diagonal(cov_x).pow_(2).sum() / C) + (self.off_diagonal(cov_y).pow_(2).sum() / C)
        
        # Calculate the aggregate weighted loss
        total_latent_loss = (self.sim_weight * sim_loss) + (self.var_weight * var_loss) + (self.cov_weight * cov_loss)
        
        # Return total, and unpack the individual metrics for TensorBoard tracking
        return total_latent_loss, sim_loss, var_loss, cov_loss

    def off_diagonal(self, x):
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

# ====================================================================================
# --- 2. ADVANCED ROBUSTNESS EVALUATION SUITE ---
# ====================================================================================

def compute_batch_miou(logits, targets, num_classes, ignore_index=255):
    preds = torch.argmax(logits, dim=1)
    ious = []
    for c in range(num_classes):
        if c == ignore_index: continue
        pred_inds = preds == c
        target_inds = targets == c
        intersection = (pred_inds & target_inds).sum().item()
        union = (pred_inds | target_inds).sum().item()
        if union > 0: ious.append(intersection / float(union))
    return sum(ious) / max(len(ious), 1) if ious else 0.0

def compute_ece(logits, targets, num_bins=10, ignore_index=255):
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
    noise = torch.randn_like(tensor) * noise_std
    return torch.clamp(tensor + noise, 0, 1)

class TTAWrapper(nn.Module):
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

def build_phase_config(phase, model, max_epochs, model_dir, num_classes, is_latent_model):
    lr, gamma, dice, opt_type, momentum, weight_decay = 0.0753, 1.6627, 0.6250, "AdamW", 0.9685, 0.0003
    alpha = 0.1 if is_latent_model else 0.5 
    beta = 0.4 
    
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
    latent_criterion = LatentRegularizationLoss()

    if phase == "baseline":
        opt = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
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
        criterion = FocalDiceLoss(num_classes, gamma=gamma, dice_weight=dice)
        latent_criterion = LatentRegularizationLoss()
        scaler = GradScaler(device.type)
        
        best_miou = 0.0
        for epoch in range(30): 
            model.train()
            for batch in train_loader:
                rgbd, therm_masked = batch['rgbd'].to(device), batch['therm_masked'].to(device)
                therm_target, seg_mask, block_mask = batch['therm_target'].to(device), batch['seg_mask'].to(device), batch['block_mask'].to(device)

                optimizer.zero_grad()
                with autocast(device_type=device.type):
                    if is_latent_model:
                        pred_seg, z_pred, z_target = model(rgbd, therm_masked, therm_target)
                        loss_seg = criterion(pred_seg, seg_mask)
                        loss_phys, _, _, _ = latent_criterion(z_pred, z_target)
                        loss = loss_seg + (alpha * loss_phys)
                    else:
                        pred_seg, pred_therm, aux_therm_seg = model(rgbd, therm_masked)
                        loss_seg = criterion(pred_seg, seg_mask)
                        loss_phys = masked_mse_loss(pred_therm, therm_target, block_mask)
                        loss = loss_seg + (alpha * loss_phys) + (beta * criterion(aux_therm_seg, seg_mask))
                        
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            
            model.eval()
            val_miou, batches = 0.0, 0
            with torch.no_grad():
                for batch in eval_loader:
                    rgbd, therm_masked, seg_mask = batch['rgbd'].to(device), batch['therm_masked'].to(device), batch['seg_mask'].to(device)
                    with autocast(device_type=device.type):
                        outputs = model(rgbd, therm_masked)
                        logits = outputs if not isinstance(outputs, tuple) else outputs[0]
                    val_miou += compute_batch_miou(logits, seg_mask, num_classes)
                    batches += 1
            
            score = val_miou / batches
            if score > best_miou: best_miou = score
            trial.report(score, epoch)
            if trial.should_prune(): raise optuna.exceptions.TrialPruned()
        return best_miou

    study = optuna.create_study(direction="maximize", storage=f"sqlite:///{study_db_path}", pruner=optuna.pruners.HyperbandPruner())
    study.optimize(objective, n_trials=30)
    with open(os.path.join(run_dir, "best_params.json"), "w") as f: json.dump(study.best_params, f, indent=4)
    return study.best_value

def export_to_onnx(model, weights_path, run_dir, device):
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
            export_params=True, opset_version=14, do_constant_folding=True,
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
    parser.add_argument("--backbone", type=str, default="mit_b1", help="Vision Transformer backbone (e.g., mit_b1, mit_b2, mit_b5)")
    parser.add_argument("--params", type=str, default="{}")
    parser.add_argument("--data_domain", type=str, default="MM5")
    parser.add_argument("--data_dir", type=str, default="dataset/MM5")
    parser.add_argument("--disable_tta", action="store_true", help="Skip TTA computation in diagnostics")
    args = parser.parse_args()

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = 6 

    train_dataset = TriModalPredictiveDataset(data_dir=args.data_dir, split="train", mask_ratio=0.30)
    eval_dataset = TriModalPredictiveDataset(data_dir=args.data_dir, split="eval", mask_ratio=0.0)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    eval_loader = DataLoader(eval_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    NUM_CLASSES = train_dataset.num_classes

    ModelClass = getattr(models, args.model)
    model_kwargs = {"num_classes": NUM_CLASSES, "backbone_name": args.backbone}
    model_kwargs.update(json.loads(args.params))
    model = ModelClass(**model_kwargs).to(DEVICE)
    
    is_latent_model = (args.model == "TriModalLatentPredictiveNetwork")
    
    # --------------------------------------------------------------------------------
    # KERAS-STYLE TOPOLOGY REPORT
    # --------------------------------------------------------------------------------
    print("\n" + "="*75)
    print("🧠 ARCHITECTURE TOPOLOGY")
    print("="*75)
    summary(model, input_size=[(BATCH_SIZE, 4, 480, 640), (BATCH_SIZE, 1, 480, 640)], col_names=["input_size", "output_size", "num_params", "mult_adds"], depth=4)
    print("="*75)
    
    manager = ExperimentManager(model_instance=model, backbone=args.backbone, data_domain=args.data_domain)
    state = manager.detect_state()
    if state is None: return 
        
    run_dir, phase = state["run_dir"], state["phase"]

    if phase == "hpo":
        best_score = run_hpo_phase(run_dir, state["inherit_weights"], ModelClass, model_kwargs, train_loader, eval_loader, NUM_CLASSES, DEVICE)
        with open(os.path.join(run_dir, "results.json"), 'w') as f: json.dump({"phase": phase, "best_hpo_mIoU": best_score}, f, indent=4)
        return

    if phase == "export":
        onnx_file = export_to_onnx(model, state["inherit_weights"], run_dir, DEVICE)
        with open(os.path.join(run_dir, "results.json"), 'w') as f: json.dump({"phase": phase, "artifact": onnx_file}, f, indent=4)
        return

    MAX_EPOCHS = 150 if phase == "baseline" else (300 if phase == "hero" else 200)
    PATIENCE = 25 if phase == "baseline" else 40

    print(f"\n🚀 PHASE: {phase.upper()} | EPOCHS: {MAX_EPOCHS} | PATIENCE: {PATIENCE}")
    print(f"📈 [MONITORING] run: tensorboard --logdir={os.path.join(run_dir, 'logs')}\n")

    optimizer, scheduler, criterion, latent_criterion, alpha, beta = build_phase_config(phase, model, MAX_EPOCHS, manager.model_dir, NUM_CLASSES, is_latent_model)
    scaler = GradScaler(DEVICE.type)

    if state["is_resume"]:
        checkpoint = torch.load(os.path.join(run_dir, "latest_checkpoint.pt"))
        model.load_state_dict(checkpoint['model_state']); optimizer.load_state_dict(checkpoint['optimizer_state'])
        scheduler.load_state_dict(checkpoint['scheduler_state']); scaler.load_state_dict(checkpoint['scaler_state'])
    elif state["inherit_weights"]:
        model.load_state_dict(torch.load(state["inherit_weights"]))

    writer = SummaryWriter(log_dir=os.path.join(run_dir, "logs"))

    for epoch in range(state["start_epoch"], MAX_EPOCHS):
        model.train()
        train_loss, train_seg_loss, train_physics_loss = 0.0, 0.0, 0.0
        
        # New variables to track the Latent Triad
        train_sim, train_var, train_cov = 0.0, 0.0, 0.0 
        
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{MAX_EPOCHS} [Train]")
        for batch in loop:
            rgbd, therm_masked = batch['rgbd'].to(DEVICE), batch['therm_masked'].to(DEVICE)
            therm_target, seg_mask, block_mask = batch['therm_target'].to(DEVICE), batch['seg_mask'].to(DEVICE), batch['block_mask'].to(DEVICE)
            
            optimizer.zero_grad()
            with autocast(device_type=DEVICE.type):
                # --- DYNAMIC FORWARD PASS (UNIVERSAL RUNNER) ---
                if is_latent_model:
                    pred_seg, z_pred, z_target = model(rgbd, therm_masked, therm_target)
                    loss_seg = criterion(pred_seg, seg_mask)
                    
                    # Unpack the specific VICReg loss metrics
                    loss_phys, sim, var, cov = latent_criterion(z_pred, z_target)
                    loss = loss_seg + (alpha * loss_phys)
                else:
                    pred_seg, pred_therm, aux_therm_seg = model(rgbd, therm_masked)
                    loss_seg = criterion(pred_seg, seg_mask)
                    loss_phys = masked_mse_loss(pred_therm, therm_target, block_mask)
                    loss = loss_seg + (alpha * loss_phys) + (beta * criterion(aux_therm_seg, seg_mask))
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            train_seg_loss += loss_seg.item()
            train_physics_loss += loss_phys.item()
            
            # Accumulate Latent Triad
            if is_latent_model:
                train_sim += sim.item()
                train_var += var.item()
                train_cov += cov.item()
                
            loop.set_postfix(loss=loss.item())
            
        scheduler.step()
        
        model.eval()
        val_loss, val_miou_accum, batches = 0.0, 0.0, 0
        with torch.no_grad():
            for batch in eval_loader:
                rgbd, therm_masked = batch['rgbd'].to(DEVICE), batch['therm_masked'].to(DEVICE)
                therm_target, seg_mask, block_mask = batch['therm_target'].to(DEVICE), batch['seg_mask'].to(DEVICE), batch['block_mask'].to(DEVICE)
                
                with autocast(device_type=DEVICE.type):
                    if is_latent_model:
                        pred_seg = model(rgbd, therm_masked) 
                        loss = criterion(pred_seg, seg_mask)
                    else:
                        pred_seg, pred_therm = model(rgbd, therm_masked)
                        loss = criterion(pred_seg, seg_mask) + (alpha * masked_mse_loss(pred_therm, therm_target, block_mask))
                    
                val_loss += loss.item()
                val_miou_accum += compute_batch_miou(pred_seg, seg_mask, NUM_CLASSES, ignore_index=255)
                batches += 1
                
        avg_val_loss = val_loss / batches
        avg_val_miou = val_miou_accum / batches
        
        writer.add_scalars("Loss/Total", {'Train': train_loss / len(train_loader), 'Validation': avg_val_loss}, epoch + 1)
        writer.add_scalar("Metrics/Validation_mIoU", avg_val_miou, epoch + 1)
        
        # Output the dissected latent metrics to TensorBoard
        if is_latent_model:
            writer.add_scalars("Loss/Latent_Components", {
                'Similarity': train_sim / len(train_loader),
                'Variance': train_var / len(train_loader),
                'Covariance': train_cov / len(train_loader)
            }, epoch + 1)
            
        writer.flush() 
        
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
        'base': {'miou': 0, 'ece': 0, 'bound': 0, 'ood': 0},
        'tta': {'miou': 0, 'ece': 0, 'bound': 0, 'ood': 0}
    }
    batches = 0
    
    for i, batch in enumerate(tqdm(eval_loader, desc="Diagnostic Pass")):
        rgbd, therm_masked, seg_mask = batch['rgbd'].to(DEVICE), batch['therm_masked'].to(DEVICE), batch['seg_mask'].to(DEVICE)
        
        # Base Evaluation
        with torch.no_grad(), autocast(device_type=DEVICE.type):
            outputs_base = model(rgbd, therm_masked)
            logits_base = outputs_base[0] if isinstance(outputs_base, tuple) else outputs_base
            
            outputs_ood = model(apply_ood_noise(rgbd), apply_ood_noise(therm_masked))
            logits_ood_base = outputs_ood[0] if isinstance(outputs_ood, tuple) else outputs_ood
            
            metrics['base']['miou'] += compute_batch_miou(logits_base, seg_mask, NUM_CLASSES)
            metrics['base']['ece'] += compute_ece(logits_base, seg_mask)
            metrics['base']['bound'] += compute_boundary_iou(logits_base, seg_mask, NUM_CLASSES)
            metrics['base']['ood'] += compute_batch_miou(logits_ood_base, seg_mask, NUM_CLASSES)

        # TTA Evaluation (Only runs if the flag is NOT triggered)
        if not args.disable_tta:
            with torch.no_grad(), autocast(device_type=DEVICE.type):
                logits_tta, var_map = tta_model(rgbd, therm_masked)
                logits_ood_tta, _ = tta_model(apply_ood_noise(rgbd), apply_ood_noise(therm_masked))
                
                metrics['tta']['miou'] += compute_batch_miou(logits_tta, seg_mask, NUM_CLASSES)
                metrics['tta']['ece'] += compute_ece(logits_tta, seg_mask)
                metrics['tta']['bound'] += compute_boundary_iou(logits_tta, seg_mask, NUM_CLASSES)
                metrics['tta']['ood'] += compute_batch_miou(logits_ood_tta, seg_mask, NUM_CLASSES)
            
        batches += 1
        
        # Visual Artifact Generation
        if i < 5:
            predictions = torch.argmax(logits_base, dim=1)
            for b in range(rgbd.size(0)):
                for cls in torch.unique(predictions[b]):
                    if cls == 0 or cls == 255: continue
                    rgbd_input, therm_input = rgbd[b].unsqueeze(0), therm_masked[b].unsqueeze(0)
                    
                    heatmap = grad_cam.generate_heatmap(rgbd_input, therm_input, cls.item())
                    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
                    
                    rgb_tensor = rgbd_input[0, :3, :, :].clone().cpu()
                    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                    rgb_unnorm = torch.clamp((rgb_tensor * std) + mean, 0, 1)
                    rgb_img = cv2.cvtColor((rgb_unnorm.permute(1, 2, 0).numpy() * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
                    
                    overlay = cv2.addWeighted(rgb_img, 0.6, heatmap_colored, 0.4, 0)
                    cv2.imwrite(os.path.join(run_dir, "explainability", f"batch{i}_img{b}_class{cls.item()}_gradcam.png"), overlay)
                    
                uncert_map = var_map[b].cpu().numpy()
                uncert_norm = np.uint8(255 * (uncert_map - uncert_map.min()) / (uncert_map.max() - uncert_map.min() + 1e-8))
                uncert_colored = cv2.applyColorMap(uncert_norm, cv2.COLORMAP_INFERNO)
                cv2.imwrite(os.path.join(run_dir, "explainability", f"batch{i}_img{b}_epistemic_uncertainty.png"), uncert_colored)

    for k in metrics.keys():
        for metric in metrics[k]: metrics[k][metric] /= batches

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

    # =========================================================================
    # --- MISSING STATE-MACHINE HANDOFF LOGIC ---
    # =========================================================================
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