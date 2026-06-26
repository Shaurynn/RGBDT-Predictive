import os
import glob
import json
import torch
import cv2
import optuna
import argparse
import datetime
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# V2 Specific Imports
from dataset import TriModalPredictiveDataset
import models  

# --- 1. Custom Loss & Metrics ---
class FocalDiceLoss(nn.Module):
    def __init__(self, num_classes, ignore_index=0, gamma=1.6627, dice_weight=0.6250):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.gamma = gamma
        self.dice_weight = dice_weight
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction='none')

    def forward(self, logits, targets):
        ce_loss = self.ce(logits, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()

        probs = F.softmax(logits, dim=1)
        targets_one_hot = F.one_hot(targets, num_classes=self.num_classes).permute(0, 3, 1, 2).float()

        dice_loss = 0.0
        valid_classes = 0

        for c in range(self.num_classes):
            if c == self.ignore_index: continue
            t = targets_one_hot[:, c]
            if t.sum() == 0: continue
            
            p = probs[:, c]
            intersection = (p * t).sum()
            union = p.sum() + t.sum()
            
            dice_c = 1.0 - (2.0 * intersection + 1e-6) / (union + 1e-6)
            dice_loss += dice_c
            valid_classes += 1

        dice_loss = (dice_loss / valid_classes) if valid_classes > 0 else 0.0
        return focal_loss + (self.dice_weight * dice_loss)

def masked_mse_loss(preds, targets, mask):
    """Computes MSE exclusively within the block-masked region for predictive physics learning."""
    diff = (preds - targets) ** 2
    masked_diff = diff * mask
    return masked_diff.sum() / (mask.sum() + 1e-8)

def compute_batch_miou(logits, targets, num_classes, ignore_index=0):
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

# --- 2. Explainability Engine (Segmentation Grad-CAM) ---
class SemanticGradCAM:
    def __init__(self, model):
        self.model = model
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        target_layer = None
        # Attaching hook to the final convolutions of the fusion head or classifier
        for module in self.model.modules():
            if isinstance(module, nn.Conv2d): target_layer = module
        if target_layer is None: return

        def forward_hook(module, input, output): self.activations = output
        def backward_hook(module, grad_input, grad_output): self.gradients = grad_output[0]

        target_layer.register_forward_hook(forward_hook)
        target_layer.register_full_backward_hook(backward_hook)

    def generate_heatmap(self, rgbd_tensor, therm_tensor, target_class):
        self.model.zero_grad()
        logits, _ = self.model(rgbd_tensor, therm_tensor)
        class_mask = logits[:, target_class, :, :]
        loss = class_mask.sum()
        loss.backward(retain_graph=True)

        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1).squeeze().detach().cpu().numpy()
        cam = np.maximum(cam, 0)
        if cam.max() > 0: cam = cam / cam.max()
        cam = cv2.resize(cam, (rgbd_tensor.shape[3], rgbd_tensor.shape[2]))
        return cam

# --- 3. The State Manager ---
class ExperimentManager:
    def __init__(self, model_instance, data_domain="MM5", base_dir="results"):
        self.model_name = model_instance.__class__.__name__
        self.data_domain = data_domain
        
        # NESTED ROUTING: results/Architecture/Data_Domain/
        self.model_dir = os.path.join(base_dir, self.model_name, self.data_domain)
        os.makedirs(self.model_dir, exist_ok=True)
        
        self.phase_sequence = ["baseline", "hpo", "hero", "microtune", "export"]
        
    def detect_state(self):
        existing_runs = sorted(glob.glob(os.path.join(self.model_dir, "*_*")))
        if not existing_runs:
            return self._create_new_run("baseline", resume_from=None)
            
        latest_run = existing_runs[-1]
        run_name = os.path.basename(latest_run)
        
        if os.path.exists(os.path.join(latest_run, "results.json")):
            current_phase = run_name.split("_")[-1].lower()
            if current_phase == self.phase_sequence[-1]:
                print(f"[*] Pipeline for {self.model_name} is fully complete.")
                return None
                
            next_phase_idx = self.phase_sequence.index(current_phase) + 1
            next_phase = self.phase_sequence[next_phase_idx]
            
            if next_phase == "hpo":
                inherit_weights = os.path.join(latest_run, "best_model.pt") 
            elif next_phase == "hero":
                baseline_runs = [r for r in existing_runs if r.lower().endswith("_baseline")]
                inherit_weights = os.path.join(baseline_runs[-1], "best_model.pt") if baseline_runs else None
            else: 
                hero_runs = [r for r in existing_runs if r.lower().endswith("_hero")]
                inherit_weights = os.path.join(hero_runs[-1], "best_model.pt") if hero_runs else None
                
            return self._create_new_run(next_phase, resume_from=inherit_weights)
        else:
            return self._resume_run(latest_run)

    def _create_new_run(self, phase, resume_from):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(self.model_dir, f"{timestamp}_{phase.capitalize()}")
        
        os.makedirs(run_dir)
        os.makedirs(os.path.join(run_dir, "weights"))
        os.makedirs(os.path.join(run_dir, "logs"))
        if phase != "hpo":
            os.makedirs(os.path.join(run_dir, "explainability")) 
        
        state = {
            "run_dir": run_dir,
            "phase": phase,
            "is_resume": False,
            "inherit_weights": resume_from,
            "start_epoch": 0,
            "best_miou": 0.0,
            "patience_counter": 0
        }
        self._save_state(run_dir, state)
        return state

    def _resume_run(self, run_dir):
        state_file = os.path.join(run_dir, "state.json")
        with open(state_file, 'r') as f: state = json.load(f)
        state["is_resume"] = True
        return state
        
    def _save_state(self, run_dir, state_dict):
        with open(os.path.join(run_dir, "state.json"), 'w') as f:
            json.dump(state_dict, f, indent=4)

# --- 4. Dynamic Configuration Injector ---
def build_phase_config(phase, model, max_epochs, model_dir, num_classes):
    lr, gamma, dice, opt_type = 0.0753, 1.6627, 0.6250, "AdamW"
    momentum, weight_decay = 0.9685, 0.0003
    alpha = 0.5 # Default physics/anatomy (JEPA) loss balance
    beta = 0.4  # Default Thermal Expert auxiliary loss balance
    
    if phase in ["hero", "microtune"]:
        existing_runs = sorted(glob.glob(os.path.join(model_dir, "*_*")))
        hpo_runs = [r for r in existing_runs if r.lower().endswith("_hpo")]
        if hpo_runs:
            params_path = os.path.join(hpo_runs[-1], "best_params.json")
            if os.path.exists(params_path):
                with open(params_path, "r") as f: p = json.load(f)
                lr = p.get("lr", lr)
                gamma = p.get("gamma", gamma)
                dice = p.get("dice_weight", dice)
                opt_type = p.get("optimizer", opt_type)
                momentum = p.get("sgd_momentum", momentum)
                weight_decay = p.get("weight_decay", weight_decay)
                alpha = p.get("alpha", alpha)
                beta = p.get("beta", beta) # Auto-load beta if HPO optimized it

    criterion = FocalDiceLoss(num_classes=num_classes, ignore_index=0, gamma=gamma, dice_weight=dice)

    if phase == "baseline":
        opt = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4) # Lower LR for Transformers
        sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_epochs, eta_min=1e-6)
    elif phase == "hero":
        if opt_type == "AdamW":
            opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            opt = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum, nesterov=True)
        sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_epochs, eta_min=1e-6)
    elif phase == "microtune":
        if opt_type == "AdamW":
            opt = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=weight_decay)
        else:
            opt = optim.SGD(model.parameters(), lr=1e-4, weight_decay=weight_decay, momentum=0.9, nesterov=True)
        sch = optim.lr_scheduler.StepLR(opt, step_size=50, gamma=0.5)
        
    return opt, sch, criterion, alpha, beta

# --- 5. Isolated HPO Engine (Predictive Adaptation) ---
def run_hpo_phase(run_dir, inherit_weights, ModelClass, model_kwargs, train_loader, eval_loader, num_classes, device):
    print("--- Initiating Optuna Hyperparameter Sweep (30 Trials) ---")
    study_db_path = os.path.join(run_dir, "optuna_study.db")
    
    def objective(trial):
        model = ModelClass(**model_kwargs).to(device)
        if inherit_weights and os.path.exists(inherit_weights):
            model.load_state_dict(torch.load(inherit_weights))
            
        # 1. ViT-Targeted Optimization Params
        lr = trial.suggest_float("lr", 1e-5, 5e-3, log=True)
        wd = trial.suggest_float("weight_decay", 1e-4, 1e-1, log=True)
        
        # 2. Segmentation Loss Params
        gamma = trial.suggest_float("gamma", 1.0, 4.0)
        dice = trial.suggest_float("dice_weight", 0.5, 2.0)
        
        # 3. JEPA Predictive & Expert Params
        alpha = trial.suggest_float("alpha", 0.1, 1.5) # Physics vs Anatomy
        beta = trial.suggest_float("beta", 0.1, 1.0)   # Thermal Expert penalty
        
        # Dynamically inject the mask ratio directly through the loader
        mask_ratio = trial.suggest_float("mask_ratio", 0.15, 0.65)
        train_loader.dataset.mask_ratio = mask_ratio 
        
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
        criterion = FocalDiceLoss(num_classes, gamma=gamma, dice_weight=dice)
        scaler = GradScaler(device.type)
        
        best_miou = 0.0
        
        for epoch in range(30): 
            model.train()
            for batch in train_loader:
                rgbd = batch['rgbd'].to(device)
                therm_masked = batch['therm_masked'].to(device)
                therm_target = batch['therm_target'].to(device)
                block_mask = batch['block_mask'].to(device)
                seg_mask = batch['seg_mask'].to(device)
                
                optimizer.zero_grad()
                with autocast(device_type=device.type):
                    # Unpack all 3 outputs from the training-mode forward pass
                    pred_seg, pred_therm, aux_therm_seg = model(rgbd, therm_masked)
                    
                    # Calculate independent objectives
                    loss_seg = criterion(pred_seg, seg_mask)
                    loss_therm = masked_mse_loss(pred_therm, therm_target, block_mask)
                    loss_aux = criterion(aux_therm_seg, seg_mask)
                    
                    # The Tri-Objective Gradient
                    loss = loss_seg + (alpha * loss_therm) + (beta * loss_aux)
                    
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            
            model.eval()
            val_miou, batches = 0.0, 0
            with torch.no_grad():
                for batch in eval_loader:
                    rgbd = batch['rgbd'].to(device)
                    therm_masked = batch['therm_masked'].to(device)
                    seg_mask = batch['seg_mask'].to(device)
                    
                    with autocast(device_type=device.type):
                        # Eval mode only returns 2 outputs
                        logits, _ = model(rgbd, therm_masked)
                    val_miou += compute_batch_miou(logits, seg_mask, num_classes)
                    batches += 1
            
            score = val_miou / batches
            if score > best_miou: best_miou = score
            
            trial.report(score, epoch)
            if trial.should_prune(): raise optuna.exceptions.TrialPruned()
                
        return best_miou

    study = optuna.create_study(direction="maximize", storage=f"sqlite:///{study_db_path}", pruner=optuna.pruners.HyperbandPruner())
    study.optimize(objective, n_trials=30)
    
    with open(os.path.join(run_dir, "best_params.json"), "w") as f:
        json.dump(study.best_params, f, indent=4)
        
    return study.best_value

# --- 6. Edge Deployment Engine (Dual-Input ONNX Export) ---
def export_to_onnx(model, weights_path, run_dir, device):
    print(f"\n--- Serializing Architecture to ONNX ---")
    
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()

    # Bifurcated Synthetic Tensors
    dummy_rgbd = torch.randn(1, 4, 480, 640, device=device)
    dummy_therm = torch.randn(1, 1, 480, 640, device=device)
    
    export_dir = os.path.join(run_dir, "deployment")
    os.makedirs(export_dir, exist_ok=True)
    onnx_path = os.path.join(export_dir, "trimodal_predictive_dynamic.onnx")

    print("[*] Tracing computational graph and folding constants...")
    
    with torch.no_grad():
        torch.onnx.export(
            model, 
            (dummy_rgbd, dummy_therm), 
            onnx_path,
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=['input_rgbd', 'input_therm'],
            output_names=['output_mask', 'output_therm_pred'],
            dynamic_axes={
                'input_rgbd': {0: 'batch_size'}, 
                'input_therm': {0: 'batch_size'},
                'output_mask': {0: 'batch_size'},
                'output_therm_pred': {0: 'batch_size'}
            }
        )
        
    print(f"[SUCCESS] ONNX graph serialized to: {onnx_path}")
    return onnx_path

# --- 7. Main Execution Engine ---
def main():
    parser = argparse.ArgumentParser(description="TriModal Predictive State-Machine Pipeline")
    parser.add_argument("--model", type=str, default="TriModalPredictiveNetwork")
    parser.add_argument("--params", type=str, default="{}")
    
    # NEW: Data routing arguments
    parser.add_argument("--data_domain", type=str, default="MM5", help="Name of the physical domain (e.g., MM5, Structural_Defects, UVSS)")
    parser.add_argument("--data_dir", type=str, default="dataset/MM5", help="Path to the dataset")
    
    args = parser.parse_args()

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = 6 

    # Route the custom data directory to the datasets
    train_dataset = TriModalPredictiveDataset(data_dir=args.data_dir, split="train", mask_ratio=0.30)
    eval_dataset = TriModalPredictiveDataset(data_dir=args.data_dir, split="eval", mask_ratio=0.0)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    eval_loader = DataLoader(eval_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    
    # Note: If future datasets have different class counts, you may want to expose NUM_CLASSES as a CLI arg as well.
    NUM_CLASSES = 14 

    try:
        ModelClass = getattr(models, args.model)
    except AttributeError:
        raise AttributeError(f"[!] Architecture '{args.model}' not found in models.py.")

    model_kwargs = {"num_classes": NUM_CLASSES}
    model_kwargs.update(json.loads(args.params))

    model = ModelClass(**model_kwargs).to(DEVICE)
    
    # Route the custom domain name to the State Machine
    manager = ExperimentManager(model_instance=model, data_domain=args.data_domain)
    state = manager.detect_state()
    
    if state is None or state == (None, None): 
        return 
        
    run_dir = state["run_dir"]
    phase = state["phase"]

    if phase == "hpo":
        print(f"\n🚀 INITIALIZING HPO SWEEP FOR {model.__class__.__name__}")
        best_score = run_hpo_phase(run_dir, state["inherit_weights"], ModelClass, model_kwargs, train_loader, eval_loader, NUM_CLASSES, DEVICE)
        
        results_payload = {
            "model_architecture": model.__class__.__name__,
            "phase": phase,
            "completed_at": datetime.datetime.now().isoformat(),
            "best_hpo_mIoU": best_score
        }
        with open(os.path.join(run_dir, "results.json"), 'w') as f:
            json.dump(results_payload, f, indent=4)
        print("\n[SUCCESS] HPO Complete. Run `python train.py` to auto-start the Hero Phase.")
        return

    if phase == "export":
        print(f"\n🚀 INITIALIZING DEPLOYMENT EXPORT FOR {model.__class__.__name__}")
        best_weights_path = state["inherit_weights"] 
        onnx_file = export_to_onnx(model, best_weights_path, run_dir, DEVICE)
        
        results_payload = {
            "model_architecture": model.__class__.__name__,
            "phase": phase,
            "completed_at": datetime.datetime.now().isoformat(),
            "deployment_artifact": onnx_file,
            "status": "Ready for Jetson TensorRT Compilation"
        }
        with open(os.path.join(run_dir, "results.json"), 'w') as f:
            json.dump(results_payload, f, indent=4)
        print("\n[SUCCESS] Pipeline Complete. The model is ready for hardware deployment.")
        return

    MAX_EPOCHS = 150 if phase == "baseline" else (300 if phase == "hero" else 200)
    PATIENCE = 25 if phase == "baseline" else 40

    print("\n" + "="*75)
    print(f"🚀 INITIALIZING RUN: {os.path.basename(run_dir)}")
    print(f"🧠 ARCHITECTURE: {model.__class__.__name__}")
    print(f"📊 PHASE: {phase.upper()} | EPOCHS: {MAX_EPOCHS} | PATIENCE: {PATIENCE}")
    print("="*75 + "\n")

    optimizer, scheduler, criterion, alpha, beta = build_phase_config(phase, model, MAX_EPOCHS, manager.model_dir, NUM_CLASSES)
    scaler = GradScaler(DEVICE.type)

    if state["is_resume"]:
        checkpoint = torch.load(os.path.join(run_dir, "latest_checkpoint.pt"))
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        scheduler.load_state_dict(checkpoint['scheduler_state'])
        scaler.load_state_dict(checkpoint['scaler_state'])
        print(f"[*] Resuming interrupted {phase} run from Epoch {state['start_epoch']}")
        
    elif state["inherit_weights"] is not None:
        model.load_state_dict(torch.load(state["inherit_weights"]))
        print(f"[*] Curriculum Learning: Inheriting weights from previous phase.")

    writer = SummaryWriter(log_dir=os.path.join(run_dir, "logs"))

    for epoch in range(state["start_epoch"], MAX_EPOCHS):
        model.train()
        train_loss, train_seg_loss, train_phys_loss = 0.0, 0.0, 0.0
        
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{MAX_EPOCHS} [Train]")
        for batch in loop:
            rgbd = batch['rgbd'].to(DEVICE)
            therm_masked = batch['therm_masked'].to(DEVICE)
            therm_target = batch['therm_target'].to(DEVICE)
            block_mask = batch['block_mask'].to(DEVICE)
            seg_mask = batch['seg_mask'].to(DEVICE)
            
            optimizer.zero_grad()
            
            with autocast(device_type=DEVICE.type):
                # Unpack the 3 outputs
                pred_seg, pred_therm, aux_therm_seg = model(rgbd, therm_masked)
                
                # Exam A: Anatomy (Primary Fusion Network)
                loss_seg = criterion(pred_seg, seg_mask)
                
                # Exam B: Physics (JEPA Target Prediction)
                loss_therm = masked_mse_loss(pred_therm, therm_target, block_mask)
                
                # Exam C: Thermal Expert (Auxiliary Supervision)
                loss_aux = criterion(aux_therm_seg, seg_mask)
                
                # The Tri-Objective Gradient Formulation
                loss = loss_seg + (alpha * loss_therm) + (beta * loss_aux)
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            train_seg_loss += loss_seg.item()
            train_phys_loss += loss_therm.item()
            loop.set_postfix(loss=loss.item())
            
        avg_train_loss = train_loss / len(train_loader)
        avg_train_seg_loss = train_seg_loss / len(train_loader)
        avg_train_phys_loss = train_phys_loss / len(train_loader)
        scheduler.step()
        
        model.eval()
        val_loss, val_miou_accum, batches = 0.0, 0.0, 0
        
        with torch.no_grad():
            for batch in eval_loader:
                rgbd = batch['rgbd'].to(DEVICE)
                therm_masked = batch['therm_masked'].to(DEVICE)
                therm_target = batch['therm_target'].to(DEVICE)
                block_mask = batch['block_mask'].to(DEVICE)
                seg_mask = batch['seg_mask'].to(DEVICE)
                
                with autocast(device_type=DEVICE.type):
                    pred_seg, pred_therm = model(rgbd, therm_masked)
                    loss_seg = criterion(pred_seg, seg_mask)
                    loss_therm = masked_mse_loss(pred_therm, therm_target, block_mask)
                    loss = loss_seg + (alpha * loss_therm)
                    
                val_loss += loss.item()
                val_miou_accum += compute_batch_miou(pred_seg, seg_mask, NUM_CLASSES, ignore_index=0)
                batches += 1
                
        avg_val_loss = val_loss / batches
        avg_val_miou = val_miou_accum / batches
        
        writer.add_scalars("Loss/Total", {'Train': avg_train_loss, 'Validation': avg_val_loss}, epoch + 1)
        writer.add_scalar("Loss/Anatomy_Seg", avg_train_seg_loss, epoch + 1)
        writer.add_scalar("Loss/Physics_MSE", avg_train_phys_loss, epoch + 1)
        writer.add_scalar("Metrics/Validation_mIoU", avg_val_miou, epoch + 1)
        writer.add_scalar("Hyperparameters/Learning_Rate", optimizer.param_groups[0]['lr'], epoch + 1)
        writer.flush() 
        
        print(f"Epoch {epoch+1} | Total Loss: {avg_train_loss:.4f} | Seg: {avg_train_seg_loss:.4f} | Phys: {avg_train_phys_loss:.4f} | Val mIoU: {avg_val_miou:.4f}")

        state["start_epoch"] = epoch + 1
        torch.save({
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'scheduler_state': scheduler.state_dict(),
            'scaler_state': scaler.state_dict(),
        }, os.path.join(run_dir, "latest_checkpoint.pt"))

        if avg_val_miou > state["best_miou"]:
            state["best_miou"] = avg_val_miou
            state["patience_counter"] = 0
            best_weights = model.state_dict()
            torch.save(best_weights, os.path.join(run_dir, "best_model.pt"))
            torch.save(best_weights, os.path.join(run_dir, "weights", f"epoch_{epoch+1}_mIoU_{avg_val_miou:.3f}.pt"))
        else:
            state["patience_counter"] += 1

        with open(os.path.join(run_dir, "state.json"), 'w') as f:
            json.dump(state, f, indent=4)

        if state["patience_counter"] >= PATIENCE:
            print(f"\n[!] Early Stopping triggered. Phase complete.")
            break

    writer.close()

    print("\n--- Generating Final Diagnostic & Grad-CAM Report ---")
    model.load_state_dict(torch.load(os.path.join(run_dir, "best_model.pt")))
    model.eval()
    
    grad_cam = SemanticGradCAM(model)
    final_val_miou, batches = 0.0, 0
    
    for i, batch in enumerate(tqdm(eval_loader, desc="Final Test Pass")):
        rgbd = batch['rgbd'].to(DEVICE)
        therm_masked = batch['therm_masked'].to(DEVICE)
        seg_mask = batch['seg_mask'].to(DEVICE)
        
        with autocast(device_type=DEVICE.type):
            logits, _ = model(rgbd, therm_masked)
            
        final_val_miou += compute_batch_miou(logits, seg_mask, NUM_CLASSES, ignore_index=0)
        batches += 1
        
        if i < 5:
            predictions = torch.argmax(logits, dim=1)
            for b in range(rgbd.size(0)):
                unique_classes = torch.unique(predictions[b])
                for cls in unique_classes:
                    if cls == 0: continue
                    rgbd_input = rgbd[b].unsqueeze(0)
                    therm_input = therm_masked[b].unsqueeze(0)
                    heatmap = grad_cam.generate_heatmap(rgbd_input, therm_input, cls.item())
                    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
                    explain_path = os.path.join(run_dir, "explainability", f"batch{i}_img{b}_class{cls.item()}_cam.png")
                    cv2.imwrite(explain_path, heatmap_colored)

    final_score = final_val_miou / batches
    
    results_payload = {
        "model_architecture": model.__class__.__name__,
        "initialization_params": model_kwargs,
        "phase": phase,
        "completed_at": datetime.datetime.now().isoformat(),
        "final_test_mIoU": final_score,
        "best_validation_mIoU": state["best_miou"],
        "epochs_trained": state["start_epoch"]
    }
    
    with open(os.path.join(run_dir, "results.json"), 'w') as f:
        json.dump(results_payload, f, indent=4)
        
    print(f"\n[SUCCESS] Phase {phase.upper()} recorded with final mIoU: {final_score:.4f}")
    print("Run `python train.py` again to automatically begin the next phase.\n")

if __name__ == '__main__':
    main()