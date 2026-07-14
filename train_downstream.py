import os
import glob
import json
import random
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
from tqdm import tqdm

from dataset_jepa import DownstreamSegmentationDataset
import models  
from config_utils import parse_with_config

def enforce_reproducibility(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ====================================================================================
# --- LOSSES & METRICS ---
# ====================================================================================

class AlphaBalancedFocalGDLLoss(nn.Module):
    def __init__(self, num_classes, alpha=None, global_gdl_weights=None, gamma=2.0, dice_weight=1.0, ignore_index=255, eps=1e-6, use_batch_dynamic=False):
        super().__init__()
        self.num_classes = num_classes
        self.gamma = gamma
        self.base_dice_weight = dice_weight
        self.ignore_index = ignore_index
        self.eps = eps
        self.use_batch_dynamic = use_batch_dynamic
        
        if alpha is None:
            self.register_buffer('alpha', torch.ones(num_classes))
        else:
            alpha_t = torch.tensor(alpha, dtype=torch.float32)
            if alpha_t.size(0) != num_classes:
                alpha_t = alpha_t[:num_classes] if alpha_t.size(0) > num_classes else torch.cat([alpha_t, torch.ones(num_classes - alpha_t.size(0))])
            self.register_buffer('alpha', alpha_t)
            
        if global_gdl_weights is None:
            self.register_buffer('gdl_weights', torch.ones(num_classes))
        else:
            gdl_t = torch.tensor(global_gdl_weights, dtype=torch.float32)
            if gdl_t.size(0) != num_classes:
                gdl_t = gdl_t[:num_classes] if gdl_t.size(0) > num_classes else torch.cat([gdl_t, torch.ones(num_classes - gdl_t.size(0))])
            self.register_buffer('gdl_weights', gdl_t)

    def forward(self, inputs, targets):
        inputs = torch.clamp(inputs.float(), min=-20.0, max=20.0)
        
        illegal_mask = (targets < 0) | (targets >= self.num_classes)
        ce_targets = targets.clone()
        ce_targets[illegal_mask & (targets != self.ignore_index)] = 0
        
        ce_loss = F.cross_entropy(inputs, ce_targets, reduction='none', ignore_index=self.ignore_index)
        ce_loss = ce_loss.clone()
        ce_loss[illegal_mask & (targets != self.ignore_index)] = 0.0
        
        ce_loss_safe = torch.clamp(ce_loss, min=0.0, max=50.0)
        pt = torch.exp(-ce_loss_safe)
        
        valid_mask = (targets != self.ignore_index) & (~illegal_mask)
        alpha_t = torch.ones_like(targets, dtype=torch.float32)
        safe_targets = torch.clamp(targets, min=0, max=self.num_classes - 1)
        alpha_t[valid_mask] = self.alpha[safe_targets[valid_mask]]
        
        focal_loss = (alpha_t * (1 - pt) ** self.gamma * ce_loss).mean()

        valid_inputs = inputs.permute(0, 2, 3, 1)[valid_mask] 
        valid_targets = targets[valid_mask]                   
        
        if valid_targets.numel() == 0:
            dice_loss = torch.tensor(0.0, device=inputs.device, requires_grad=True)
        else:
            inputs_soft = F.softmax(valid_inputs, dim=1)
            safe_valid_targets = torch.clamp(valid_targets, min=0, max=self.num_classes - 1)
            targets_one_hot = F.one_hot(safe_valid_targets, num_classes=self.num_classes).float()
            
            intersection = (inputs_soft * targets_one_hot).sum(dim=0)
            ground_truth_volume = targets_one_hot.sum(dim=0)
            pred_volume = inputs_soft.sum(dim=0)
            
            if self.use_batch_dynamic:
                w = 1.0 / (ground_truth_volume ** 2 + self.eps)
            else:
                w = self.gdl_weights[:self.num_classes].to(inputs.device)
                
            present_classes = w > 0
            
            if present_classes.sum() > 0:
                numerator = 2.0 * (w[present_classes] * intersection[present_classes]).sum()
                denominator = (w[present_classes] * (ground_truth_volume[present_classes] + pred_volume[present_classes])).sum()
                dice_loss = 1.0 - (numerator / (denominator + self.eps))
            else:
                dice_loss = torch.tensor(0.0, device=inputs.device, requires_grad=True)

        return focal_loss + (self.base_dice_weight * dice_loss)

def feature_distillation_loss(student_features, teacher_features):
    """
    V2 KD: Aligns the intermediate representational manifolds of the student 
    directly with the teacher, circumventing noisy logit predictions.
    """
    loss = 0.0
    valid_stages = 0
    for s_feat, t_feat in zip(student_features, teacher_features):
        if s_feat.shape == t_feat.shape:
            # L2 Normalization guarantees gradient magnitude stability regardless of stage depth
            s_norm = F.normalize(s_feat, dim=1)
            t_norm = F.normalize(t_feat, dim=1)
            loss += F.mse_loss(s_norm, t_norm)
            valid_stages += 1
    return loss / max(1, valid_stages)

class IoUMetric:
    def __init__(self, num_classes, device):
        self.num_classes = num_classes
        self.device = device
        self.intersections = torch.zeros(num_classes, device=device)
        self.unions = torch.zeros(num_classes, device=device)
        
    def update(self, logits, targets, ignore_index=255):
        preds = torch.argmax(logits, dim=1)
        valid_mask = (
            (targets != ignore_index) & 
            (targets >= 0) & (targets < self.num_classes) & 
            (preds >= 0) & (preds < self.num_classes)
        )
        preds = preds[valid_mask]
        targets = targets[valid_mask]
        if targets.numel() == 0: return
        
        bins = targets * self.num_classes + preds
        bincount = torch.bincount(bins, minlength=self.num_classes**2)
        if bincount.numel() > self.num_classes**2:
            bincount = bincount[:self.num_classes**2]
            
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

def export_to_onnx(model, weights_path, run_dir, device):
    print(f"\n--- Serializing Architecture to ONNX ---")
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()
    dummy_input = torch.randn(1, 5, 480, 640, device=device)
    export_dir = os.path.join(run_dir, "deployment")
    os.makedirs(export_dir, exist_ok=True)
    onnx_path = os.path.join(export_dir, f"{model.__class__.__name__}.onnx")
    with torch.no_grad():
        torch.onnx.export(
            model, dummy_input, onnx_path,
            export_params=True, opset_version=18, do_constant_folding=True,
            input_names=['input_5channel'], output_names=['output_mask'],
            dynamic_axes={'input_5channel': {0: 'batch_size'}, 'output_mask': {0: 'batch_size'}}
        )
    return onnx_path

# ====================================================================================
# --- PIPELINE MANAGEMENT & TRAINING ---
# ====================================================================================

def build_llrd_optimizer(model, base_lr, weight_decay, decay_rate, phase):
    """
    Implements Layer-Wise Learning Rate Decay (LLRD).
    Applies exponential decay down the backbone to prevent catastrophic forgetting.
    """
    if phase != "microtune":
        trainable = [p for p in model.parameters() if p.requires_grad]
        return optim.AdamW(trainable, lr=base_lr, weight_decay=weight_decay)
        
    param_groups = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        # Group by architectural depth
        if 'decode_head' in name:
            lr = base_lr
        elif 'block4' in name or 'patch_embed4' in name:
            lr = base_lr * (decay_rate ** 1)
        elif 'block3' in name or 'patch_embed3' in name:
            lr = base_lr * (decay_rate ** 2)
        elif 'block2' in name or 'patch_embed2' in name:
            lr = base_lr * (decay_rate ** 3)
        elif 'block1' in name or 'patch_embed1' in name:
            lr = base_lr * (decay_rate ** 4)
        elif 'patch_embed1.proj' in name or 'dt_alignment' in name:
            lr = base_lr * (decay_rate ** 5)
        else:
            lr = base_lr * (decay_rate ** 5)
            
        param_groups.append({'params': [param], 'lr': lr, 'weight_decay': weight_decay})
        
    return optim.AdamW(param_groups)

class ExperimentManager:
    def __init__(self, model_instance, dataset="MM5", backbone="mit_b1", base_dir="results"):
        self.model_name = model_instance.__class__.__name__
        self.dataset = dataset
        self.backbone = backbone
        self.model_dir = os.path.join(base_dir, self.model_name, self.dataset, self.backbone)
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
            next_phase = self.phase_sequence[self.phase_sequence.index(current_phase) + 1]
            inherit_weights = os.path.join(latest_run, "best_model.pt") 
            return self._create_new_run(next_phase, resume_from=inherit_weights)
        return self._resume_run(latest_run)

    def _create_new_run(self, phase, resume_from):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(self.model_dir, f"{timestamp}_{phase.capitalize()}")
        os.makedirs(run_dir); os.makedirs(os.path.join(run_dir, "weights")); os.makedirs(os.path.join(run_dir, "logs"))
        state = {"run_dir": run_dir, "phase": phase, "is_resume": False, "inherit_weights": resume_from, "start_epoch": 0, "best_miou": 0.0, "patience_counter": 0}
        with open(os.path.join(run_dir, "state.json"), 'w') as f: json.dump(state, f, indent=4)
        return state

    def _resume_run(self, run_dir):
        with open(os.path.join(run_dir, "state.json"), 'r') as f: state = json.load(f)
        state["is_resume"] = True
        return state

def run_hpo_phase(run_dir, inherit_weights, ModelClass, model_kwargs, train_loader, eval_loader, num_classes, empirical_alpha, global_gdl, device, cfg_hpo):
    db_path = os.path.join(run_dir, "hpo_sweep.db")
    
    def objective(trial):
        model = ModelClass(**model_kwargs).to(device)
        if inherit_weights and os.path.exists(inherit_weights): model.load_state_dict(torch.load(inherit_weights))
        
        lr = trial.suggest_float("lr", cfg_hpo['lr_min'], cfg_hpo['lr_max'], log=True)
        wd = trial.suggest_float("weight_decay", cfg_hpo['wd_min'], cfg_hpo['wd_max'], log=True)
        gamma = trial.suggest_float("gamma", 1.0, 4.0)
        dice = trial.suggest_float("dice_weight", 0.5, 2.0)
        
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=wd)
        
        criterion = AlphaBalancedFocalGDLLoss(num_classes=num_classes, alpha=empirical_alpha, global_gdl_weights=global_gdl, gamma=gamma, dice_weight=dice).to(device)
        scaler = GradScaler('cuda', enabled=True)
        best_miou = 0.0
        
        for epoch in range(30): 
            model.train()
            for batch in train_loader:
                x_full, seg_mask = batch['x_full'].to(device), batch['seg_mask'].to(device)
                optimizer.zero_grad(set_to_none=True)
                with autocast('cuda', dtype=torch.bfloat16):
                    pred_seg = model(x_full)
                    loss = criterion(pred_seg, seg_mask)
                
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                is_valid_gradients = True
                for param in model.parameters():
                    if param.grad is not None and not torch.isfinite(param.grad).all():
                        is_valid_gradients = False
                        break
                if is_valid_gradients:
                    scaler.step(optimizer)
                scaler.update()
            
            model.eval()
            val_iou = IoUMetric(num_classes, device)
            with torch.no_grad():
                for batch in eval_loader:
                    x_full, seg_mask = batch['x_full'].to(device), batch['seg_mask'].to(device)
                    with autocast('cuda', dtype=torch.bfloat16):
                        pred_seg = model(x_full)
                    val_iou.update(pred_seg, seg_mask)
            score = val_iou.get_miou()
            if score > best_miou: best_miou = score
            trial.report(score, epoch)
            if trial.should_prune(): raise optuna.exceptions.TrialPruned()
        return best_miou

    study = optuna.create_study(study_name="TMLPN_HPO_Sweep", storage=f"sqlite:///{db_path}", direction="maximize", pruner=optuna.pruners.HyperbandPruner(), load_if_exists=True)
    study.optimize(objective, n_trials=cfg_hpo['n_trials'])
    with open(os.path.join(run_dir, "best_params.json"), "w") as f: json.dump(study.best_params, f, indent=4)
    return study.best_value

def get_dynamic_class_count(data_dir):
    class_file = os.path.join(data_dir, "classes.txt")
    with open(class_file, "r") as f:
        num_classes = len([line for line in f if line.strip()])
    return num_classes

def compute_dataset_statistics(dataloader, num_classes, ignore_index=255, max_batches=100):
    pseudo_count = 1.0
    pixel_counts = torch.full((num_classes,), pseudo_count, dtype=torch.float64)
    for i, batch in enumerate(dataloader):
        if i >= max_batches: break
        targets = batch['seg_mask']
        valid_targets = targets[targets != ignore_index]
        if valid_targets.numel() > 0:
            max_val = valid_targets.max().item()
            if max_val >= pixel_counts.size(0):
                new_size = max_val + 1
                expanded = torch.full((new_size,), pseudo_count, dtype=torch.float64)
                expanded[:pixel_counts.size(0)] = pixel_counts
                pixel_counts = expanded
                num_classes = new_size
            counts = torch.bincount(valid_targets.flatten(), minlength=num_classes)
            pixel_counts[:counts.size(0)] += counts.cpu().double()
            
    frequencies = pixel_counts / pixel_counts.sum()
    alpha = torch.median(frequencies) / frequencies
    gdl_raw = 1.0 / (frequencies ** 2)
    gdl = gdl_raw / gdl_raw.max()
    return [round(a, 4) for a in alpha.tolist()], [round(g, 4) for g in gdl.tolist()]

class SegmentationGradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate_cam(self, input_tensor, target_class):
        self.model.eval()
        self.model.zero_grad()
        logits = self.model(input_tensor)
        target_logits = logits[:, target_class, :, :]
        loss = target_logits.sum()
        loss.backward(retain_graph=True)
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        activations = self.activations.detach()
        for i in range(activations.shape[1]):
            activations[:, i, :, :] *= pooled_gradients[i]
        heatmap = F.relu(torch.mean(activations, dim=1).squeeze())
        if torch.max(heatmap) > 0: heatmap /= torch.max(heatmap)
        return heatmap.cpu().numpy(), logits.detach()

def main():
    args, config = parse_with_config("Phase 2: Downstream Supervised Semantic Segmentation")
    cfg = config['phase2_downstream']
    img_size = (config['dataset']['image_height'], config['dataset']['image_width'])
    dataset_name = config['dataset']['name']
    trial_name = cfg.get('trial_name', 'baseline')
    active_seed = cfg.get('seed', 42)
    enforce_reproducibility(active_seed)

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    splits_root = os.path.join("data", "splits")
    NUM_CLASSES = get_dynamic_class_count(os.path.join(splits_root, dataset_name))
    
    train_dataset = DownstreamSegmentationDataset(dataset_name=dataset_name, split="train", splits_root=splits_root, image_size=img_size)
    eval_dataset = DownstreamSegmentationDataset(dataset_name=dataset_name, split="eval", splits_root=splits_root, image_size=img_size)
    
    train_loader = DataLoader(train_dataset, batch_size=cfg['batch_size'], shuffle=True, num_workers=4)
    eval_loader = DataLoader(eval_dataset, batch_size=cfg['batch_size'], shuffle=False, num_workers=4)
    
    empirical_alpha, global_gdl = compute_dataset_statistics(train_loader, NUM_CLASSES)
    
    ablation_cfg = cfg.get('ablations', {})
    gdl_type = ablation_cfg.get('gdl_type', 'global_anchored')
    enable_kd = ablation_cfg.get('enable_kd', True)
    use_lora = cfg.get('use_lora', True)
    
    model_kwargs = {
        "num_classes": NUM_CLASSES,
        "backbone_name": cfg['backbone'],
        "isolated_stem": ablation_cfg.get('enable_modality_isolation', True),
        "use_lora": use_lora,
        "lora_r": cfg.get('lora', {}).get('r', 8),
        "lora_alpha": cfg.get('lora', {}).get('alpha', 16)
    }

    # Model upgraded to V2 Architecture
    model = models.TMLPN_Downstream_v2(**model_kwargs).to(DEVICE)
    
    teacher_model = None
    if enable_kd and getattr(args, 'teacher_weights', None) and os.path.exists(args.teacher_weights):
        t_kwargs = model_kwargs.copy()
        t_kwargs['backbone_name'] = cfg['teacher_backbone']
        t_kwargs['use_lora'] = False # Teachers deploy fully merged weights
        teacher_model = models.TMLPN_Downstream_v2(**t_kwargs).to(DEVICE)
        teacher_model.load_state_dict(torch.load(args.teacher_weights))
        teacher_model.eval()
        for p in teacher_model.parameters(): p.requires_grad = False
    
    isolated_backbone = f"{cfg['backbone']}_{trial_name}" if trial_name != "baseline" else cfg['backbone']
    manager = ExperimentManager(model_instance=model, dataset=dataset_name, backbone=isolated_backbone)
    state = manager.detect_state()
    if state is None: return 
    run_dir, phase = state["run_dir"], state["phase"]

    if phase == "baseline" and not state["is_resume"]:
        pt_weights = os.path.join("weights", dataset_name, trial_name, f"jepa_context_encoder_{cfg['backbone']}.pt")
        if os.path.exists(pt_weights):
            print(f"[*] Injecting Phase 1 MM-JEPA Foundation Weights: {pt_weights}")
            torch.nn.Module.load_state_dict(model.context_encoder, torch.load(pt_weights), strict=False)
            for param in model.context_encoder.parameters(): param.requires_grad = False

    if phase == "hpo":
        score = run_hpo_phase(run_dir, state["inherit_weights"], models.TMLPN_Downstream_v2, model_kwargs, train_loader, eval_loader, NUM_CLASSES, empirical_alpha, global_gdl, DEVICE, cfg['hpo'])
        with open(os.path.join(run_dir, "results.json"), 'w') as f: json.dump({"phase": phase, "best_mIoU": score}, f)
        return
    elif phase == "export":
        export_to_onnx(model, state["inherit_weights"], run_dir, DEVICE)
        with open(os.path.join(run_dir, "results.json"), 'w') as f: json.dump({"phase": phase}, f)
        return

    MAX_EPOCHS = cfg['epochs'].get(phase, cfg['epochs']['default'])
    lr = cfg['learning_rates'].get(phase, cfg['learning_rates']['default'])
    weight_decay = cfg.get('optimizer', {}).get('weight_decay', 1e-4)
    gamma, dice_weight = 2.0, 1.0
    
    if phase in ["hero", "microtune"] and state.get("inherit_weights"):
        hpo_params_path = os.path.join(os.path.dirname(state["inherit_weights"]), "best_params.json")
        if os.path.exists(hpo_params_path):
            with open(hpo_params_path, 'r') as f: best_params = json.load(f)
            lr = best_params.get("lr", lr)
            weight_decay = best_params.get("weight_decay", weight_decay)
            gamma = best_params.get("gamma", gamma)
            dice_weight = best_params.get("dice_weight", dice_weight)
            if phase == "microtune":
                lr = cfg['learning_rates'].get('microtune', 1e-5)
                
        # --- NEW INJECTION: Load the inherited weights ---
        if os.path.exists(state["inherit_weights"]):
            print(f"\n[*] Inheriting converged weights from previous phase: {state['inherit_weights']}")
            model.load_state_dict(torch.load(state["inherit_weights"], map_location=DEVICE), strict=False)
        # -------------------------------------------------

    if phase == "microtune":
        print("\n[*] Initializing Microtune Phase")
        if use_lora:
            print("[*] LoRA Active: Tuning ONLY Decoder and Low-Rank matrices. Foundation remains frozen.")
            for name, param in model.named_parameters():
                if 'lora_' in name or 'decode_head' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
        else:
            print("[*] LoRA Inactive: Full backbone unfreezing with Layer-Wise Learning Rate Decay (LLRD).")
            for param in model.parameters():
                param.requires_grad = True

    optimizer = build_llrd_optimizer(model, lr, weight_decay, cfg.get('llrd_decay', 0.85), phase)
    scaler = GradScaler('cuda', enabled=True)
    
    criterion = AlphaBalancedFocalGDLLoss(
        num_classes=NUM_CLASSES, alpha=empirical_alpha, global_gdl_weights=global_gdl if gdl_type == 'global_anchored' else None,
        gamma=gamma, dice_weight=dice_weight, use_batch_dynamic=(gdl_type == 'batch_dynamic')
    ).to(DEVICE)
    
    active_hparams = {"lr": lr, "weight_decay": weight_decay, "gamma": gamma, "dice_weight": dice_weight, "batch_size": cfg['batch_size'], "use_lora": use_lora, **ablation_cfg}
    state["hyperparameters"] = active_hparams
    
    writer = SummaryWriter(log_dir=os.path.join(run_dir, "logs"))
    writer.add_text("Configuration/Hyperparameters", json.dumps(active_hparams, indent=4), 0)
    cam_extractor = SegmentationGradCAM(model, model.decode_head.linear_pred)

    start_epoch = 0
    if state["is_resume"]:
        checkpoint_path = os.path.join(run_dir, "checkpoint.pt")
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            try: optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            except ValueError: pass
            try: scaler.load_state_dict(checkpoint['scaler_state_dict'])
            except Exception: pass
            start_epoch = checkpoint['epoch']
            state["best_miou"] = checkpoint['best_miou']

    for epoch in range(start_epoch, MAX_EPOCHS):
        model.train()
        train_loss = 0.0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{MAX_EPOCHS} [{phase}]")
        
        for batch in loop:
            x_full, seg_mask = batch['x_full'].to(DEVICE), batch['seg_mask'].to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            
            with autocast('cuda', dtype=torch.bfloat16):
                pred_seg, s_feats = model(x_full, return_features=True)
                loss = criterion(pred_seg, seg_mask)
                
                if enable_kd and teacher_model is not None:
                    with torch.no_grad():
                        _, t_feats = teacher_model(x_full, return_features=True)
                    loss_kd = feature_distillation_loss(s_feats, t_feats)
                    loss = (0.5 * loss) + (0.5 * loss_kd)
                    
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            is_valid_gradients = True
            for param in model.parameters():
                if param.grad is not None and not torch.isfinite(param.grad).all():
                    is_valid_gradients = False
                    break
            
            if is_valid_gradients: scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            loop.set_postfix(loss=loss.item())
            
        writer.add_scalar("Loss/Train", train_loss / len(train_loader), epoch)
            
        model.eval()
        val_loss = 0.0
        val_iou = IoUMetric(NUM_CLASSES, DEVICE)
        logged_image = False 
        
        with torch.no_grad(): 
            for batch in eval_loader:
                x_full, seg_mask = batch['x_full'].to(DEVICE), batch['seg_mask'].to(DEVICE)
                with autocast('cuda', dtype=torch.bfloat16):
                    pred_seg = model(x_full)
                    loss = criterion(pred_seg, seg_mask)
                    val_loss += loss.item()
                val_iou.update(pred_seg, seg_mask)
                
                if not logged_image and epoch % 5 == 0:
                    with torch.enable_grad(): 
                        heatmap, _ = cam_extractor.generate_cam(x_full[0:1], target_class=1)
                    rgb_vis = x_full[0, :3].cpu().numpy().transpose(1, 2, 0)
                    rgb_vis = (rgb_vis - rgb_vis.min()) / (rgb_vis.max() - rgb_vis.min() + 1e-8)
                    heatmap_resized = cv2.resize(heatmap, (rgb_vis.shape[1], rgb_vis.shape[0]))
                    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
                    heatmap_color = (np.float32(heatmap_color) / 255.0)[:, :, ::-1]
                    overlay = 0.5 * rgb_vis + 0.5 * heatmap_color
                    writer.add_image("Explainability/GradCAM_Defect_Focus", overlay.transpose(2, 0, 1), epoch)
                    logged_image = True
        
        avg_val_miou = val_iou.get_miou()
        writer.add_scalar("Loss/Validation", val_loss / len(eval_loader), epoch)
        writer.add_scalar("Metrics/mIoU_Validation", avg_val_miou, epoch)
        
        if avg_val_miou > state.get("best_miou", 0.0):
            state["best_miou"] = avg_val_miou
            torch.save(model.state_dict(), os.path.join(run_dir, "best_model.pt"))
            
        torch.save({
            'epoch': epoch + 1, 'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(), 'scaler_state_dict': scaler.state_dict(),
            'best_miou': state.get("best_miou", 0.0)
        }, os.path.join(run_dir, "checkpoint.pt"))
        with open(os.path.join(run_dir, "state.json"), 'w') as f: json.dump(state, f, indent=4)
            
    with open(os.path.join(run_dir, "results.json"), 'w') as f:
        json.dump({"phase": phase, "hyperparameters": active_hparams, "best_mIoU": state.get("best_miou", 0.0), "final_train_loss": train_loss / len(train_loader), "final_val_loss": val_loss / len(eval_loader)}, f, indent=4)

if __name__ == '__main__':
    main()