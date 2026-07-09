import os
import glob
import json
import inspect
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

from dataset_jepa import DownstreamSegmentationDataset # (Must implement this in your dataset.py)
import models  

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
# --- LOSSES & METRICS (Restored from Original Pipeline) ---
# ====================================================================================

class FocalDiceLoss(nn.Module):
    def __init__(self, num_classes, gamma=2.0, dice_weight=1.0, ignore_index=255):
        super().__init__()
        self.num_classes = num_classes
        self.gamma = gamma
        self.base_dice_weight = dice_weight
        self.ignore_index = ignore_index
        self.register_buffer('ce_weights', torch.ones(num_classes))
        self.ce_weights[0] = 0.1 
        self.register_buffer('dynamic_dice_weights', torch.ones(num_classes))

    def update_dynamic_weights(self, per_class_iou, momentum=0.9, tau=2.0):
        per_class_iou = per_class_iou.to(self.dynamic_dice_weights.device)
        target_weights = torch.exp(tau * (1.0 - per_class_iou))
        target_weights[0] = 1.0 
        self.dynamic_dice_weights = (momentum * self.dynamic_dice_weights) + ((1.0 - momentum) * target_weights)
        self.dynamic_dice_weights = torch.clamp(self.dynamic_dice_weights, min=1.0, max=10.0)

    def forward(self, inputs, targets):
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
            inputs_soft = F.softmax(valid_inputs, dim=1)
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

def knowledge_distillation_loss(student_logits, teacher_logits, temperature=4.0):
    soft_targets = F.softmax(teacher_logits / temperature, dim=1)
    student_log_probs = F.log_softmax(student_logits / temperature, dim=1)
    return F.kl_div(student_log_probs, soft_targets, reduction='batchmean') * (temperature ** 2)

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
        if targets.numel() == 0: return
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

class TTAWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x_full):
        out_orig = self.model(x_full)
        out_hf = self.model(torch.flip(x_full, dims=[3]))
        out_hf = torch.flip(out_hf, dims=[3]) 
        out_vf = self.model(torch.flip(x_full, dims=[2]))
        out_vf = torch.flip(out_vf, dims=[2]) 
        preds_stack = torch.stack([out_orig, out_hf, out_vf], dim=0)
        return preds_stack.mean(dim=0), preds_stack.var(dim=0).mean(dim=1)

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
class ExperimentManager:
    def __init__(self, model_instance, backbone="mit_b1", base_dir="results"):
        self.model_name = model_instance.__class__.__name__
        self.model_dir = os.path.join(base_dir, self.model_name, backbone)
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

def run_hpo_phase(run_dir, inherit_weights, ModelClass, model_kwargs, train_loader, eval_loader, num_classes, device):
    def objective(trial):
        model = ModelClass(**model_kwargs).to(device)
        if inherit_weights and os.path.exists(inherit_weights): model.load_state_dict(torch.load(inherit_weights))
        lr = trial.suggest_float("lr", 1e-5, 5e-3, log=True)
        wd = trial.suggest_float("weight_decay", 1e-4, 1e-1, log=True)
        gamma = trial.suggest_float("gamma", 1.0, 4.0)
        dice = trial.suggest_float("dice_weight", 0.5, 2.0)
        
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
        criterion = FocalDiceLoss(num_classes, gamma=gamma, dice_weight=dice).to(device)
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

    study = optuna.create_study(direction="maximize", pruner=optuna.pruners.HyperbandPruner())
    study.optimize(objective, n_trials=30)
    with open(os.path.join(run_dir, "best_params.json"), "w") as f: json.dump(study.best_params, f, indent=4)
    return study.best_value

def main():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = 6 
    NUM_CLASSES = 10
    
    train_dataset = DownstreamSegmentationDataset(data_dir="dataset/MM5", split="train")
    eval_dataset = DownstreamSegmentationDataset(data_dir="dataset/MM5", split="eval")
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    eval_loader = DataLoader(eval_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    model = models.TMLPN_Downstream(num_classes=NUM_CLASSES, backbone_name='mit_b1').to(DEVICE)
    
    # KNOWLEDGE DISTILLATION (Optional Teacher)
    teacher_model = None
    teacher_weights_path = "weights/best_teacher_mit_b4.pt" # Example path
    if os.path.exists(teacher_weights_path):
        teacher_model = models.TMLPN_Downstream(num_classes=NUM_CLASSES, backbone_name='mit_b4').to(DEVICE)
        teacher_model.load_state_dict(torch.load(teacher_weights_path))
        teacher_model.eval()
        for p in teacher_model.parameters(): p.requires_grad = False
    
    manager = ExperimentManager(model_instance=model, backbone='mit_b1')
    state = manager.detect_state()
    if state is None: return 
    run_dir, phase = state["run_dir"], state["phase"]

    # INJECT PHASE 1 WEIGHTS IF IN BASELINE
    if phase == "baseline" and not state["is_resume"]:
        pt_weights = "weights/jepa_context_encoder.pt"
        if os.path.exists(pt_weights):
            print("[*] Injecting Phase 1 MM-JEPA Foundation Weights")
            model.context_encoder.load_state_dict(torch.load(pt_weights))
            # Freeze the Context Backbone to evaluate strict linear probing
            for param in model.context_encoder.parameters():
                param.requires_grad = False

    if phase == "hpo":
        score = run_hpo_phase(run_dir, state["inherit_weights"], models.TMLPN_Downstream, {"num_classes": NUM_CLASSES}, train_loader, eval_loader, NUM_CLASSES, DEVICE)
        with open(os.path.join(run_dir, "results.json"), 'w') as f: json.dump({"phase": phase, "best_mIoU": score}, f)
        return
    elif phase == "export":
        export_to_onnx(model, state["inherit_weights"], run_dir, DEVICE)
        with open(os.path.join(run_dir, "results.json"), 'w') as f: json.dump({"phase": phase}, f)
        return

    MAX_EPOCHS = 150 if phase == "baseline" else (300 if phase == "hero" else 200)
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
    criterion = FocalDiceLoss(num_classes=NUM_CLASSES).to(DEVICE)
    scaler = GradScaler('cuda', enabled=True)
    writer = SummaryWriter(log_dir=os.path.join(run_dir, "logs"))

    for epoch in range(state["start_epoch"], MAX_EPOCHS):
        model.train()
        train_loss = 0.0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{MAX_EPOCHS} [{phase}]")
        
        for batch in loop:
            x_full, seg_mask = batch['x_full'].to(DEVICE), batch['seg_mask'].to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            
            with autocast('cuda', dtype=torch.bfloat16):
                pred_seg = model(x_full)
                loss = criterion(pred_seg, seg_mask)
                
                if teacher_model is not None:
                    with torch.no_grad():
                        t_seg = teacher_model(x_full)
                    loss_kd = knowledge_distillation_loss(pred_seg, t_seg, temperature=4.0)
                    loss = (0.5 * loss) + (0.5 * loss_kd)
                    
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()
            
        # Evaluation
        model.eval()
        val_iou = IoUMetric(NUM_CLASSES, DEVICE)
        with torch.no_grad():
            for batch in eval_loader:
                x_full, seg_mask = batch['x_full'].to(DEVICE), batch['seg_mask'].to(DEVICE)
                with autocast('cuda', dtype=torch.bfloat16):
                    pred_seg = model(x_full)
                val_iou.update(pred_seg, seg_mask)
                
        avg_val_miou = val_iou.get_miou()
        writer.add_scalar("Metrics/Validation_mIoU", avg_val_miou, epoch)
        
        if phase == "hero": criterion.update_dynamic_weights(val_iou.get_per_class_iou())
        
        if avg_val_miou > state["best_miou"]:
            state["best_miou"] = avg_val_miou
            torch.save(model.state_dict(), os.path.join(run_dir, "best_model.pt"))
            
        with open(os.path.join(run_dir, "state.json"), 'w') as f: json.dump(state, f)

    # FINAL DIAGNOSTICS (TTA)
    print("\n🔬 INITIATING COMPREHENSIVE TTA ROBUSTNESS DIAGNOSTICS")
    model.load_state_dict(torch.load(os.path.join(run_dir, "best_model.pt")))
    tta_model = TTAWrapper(model).eval()
    
    metrics = {'base': IoUMetric(NUM_CLASSES, DEVICE), 'tta': IoUMetric(NUM_CLASSES, DEVICE)}
    with torch.no_grad():
        for batch in tqdm(eval_loader, desc="Diagnostic Pass"):
            x_full, seg_mask = batch['x_full'].to(DEVICE), batch['seg_mask'].to(DEVICE)
            with autocast('cuda', dtype=torch.bfloat16):
                logits_base = model(x_full)
                logits_tta, _ = tta_model(x_full)
                metrics['base'].update(logits_base, seg_mask)
                metrics['tta'].update(logits_tta, seg_mask)

    print(f"Standard mIoU: {metrics['base'].get_miou():.4f} | TTA mIoU: {metrics['tta'].get_miou():.4f}")
    with open(os.path.join(run_dir, "results.json"), 'w') as f:
        json.dump({"phase": phase, "final_base_mIoU": metrics['base'].get_miou(), "final_tta_mIoU": metrics['tta'].get_miou()}, f)

if __name__ == '__main__':
    enforce_reproducibility()
    main()