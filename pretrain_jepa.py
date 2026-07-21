import os
import math
import random
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from models import MultimodalJEPA
from dataset_jepa import JEPAPretrainDataset
from config_utils import parse_with_config 

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def enforce_reproducibility(seed=42):
    """Locks the computational graph for rigorous ablation statistics."""
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def scale_invariant_depth_loss(pred_depth, target_depth, eps=1e-6):
    """
    Computes a scale-invariant error for depth tensors, enforcing geometric 
    structure consistency invariant to absolute distance scale changes.
    """
    log_diff = torch.log(pred_depth + eps) - torch.log(target_depth + eps)
    n = log_diff.numel()
    return torch.mean(log_diff ** 2) - (torch.sum(log_diff) ** 2) / (n ** 2)

def main():
    args, config = parse_with_config("Phase 1: MM-JEPA Self-Supervised Pre-Training")
    
    # --- Extracted Configuration Parameters ---
    cfg = config['phase1_pretraining']
    img_size = (config['dataset']['image_height'], config['dataset']['image_width'])
    dataset_name = config['dataset']['name']
    
    # Extract the target model architecture namespace for weight isolation
    # Defaults natively to the TMLPN_Downstream_v3 structural namespace
    model_name = getattr(args, 'model', config.get('metadata', {}).get('model', 'TMLPN_Downstream_v3'))
    
    # --- Isolate namespace and extract dynamic seed ---
    trial_name = cfg.get('trial_name', 'baseline')
    active_seed = cfg.get('seed', 42)
    enforce_reproducibility(active_seed)
    print(f"[*] Locked PyTorch computational graph to Seed: {active_seed}")

    # --- ABLATION STATE INJECTION ---
    ablation_cfg = cfg.get('ablations', {})
    enable_isolation = ablation_cfg.get('enable_modality_isolation', True)
    variance_type = ablation_cfg.get('variance_type', 'spatial')
    mask_strategy = ablation_cfg.get('mask_strategy', 'multi_block')
    
    # Extract V3 Fortification Flags
    enable_context = ablation_cfg.get('enable_context_consistency', True)
    enable_covariance = ablation_cfg.get('enable_covariance_penalty', True)

    print("\n" + "="*60)
    print("🔬 ABLATION STATE [PHASE 1: PRE-TRAINING (V3 PHYSICAL PRIORS)]")
    print("="*60)
    print(f"[*] Target Architecture Namespace      : {model_name}")
    print(f"[*] Modality Isolated Stem (1x1 Dirac) : {enable_isolation}")
    print(f"[*] Variance Regularization Topology   : {variance_type.upper()}")
    print(f"[*] Masking Strategy                   : {mask_strategy.upper()}")
    print(f"[*] Context Consistency Evaluated      : {enable_context}")
    print(f"[*] Covariance Penalty Evaluated       : {enable_covariance}")
    print("="*60 + "\n")

    # --- Agnostic Configuration Routing ---
    splits_root = os.path.join("data", "splits")
    
    dataset = JEPAPretrainDataset(
        dataset_name=dataset_name, 
        split="train",
        splits_root=splits_root, 
        image_size=img_size,
        mask_strategy=mask_strategy
    )
    
    dataloader = DataLoader(
        dataset, 
        batch_size=cfg['batch_size'], 
        shuffle=True, 
        num_workers=4, 
        pin_memory=True, 
        drop_last=True
    )
    
    model = MultimodalJEPA(backbone_name=cfg['backbone'], isolated_stem=enable_isolation).to(DEVICE)
    
    optimizer = optim.AdamW(model.parameters(), lr=cfg['optimizer']['lr'], weight_decay=cfg['optimizer']['weight_decay'])
    
    # Enforce strict isolated namespace for artifact generation matching the results/ tree
    weight_dir = os.path.join("weights", model_name, dataset_name, trial_name)
    os.makedirs(weight_dir, exist_ok=True)
    
    start_epoch = 0
    print(f"\n🚀 INITIATING PHASE 1: SELF-SUPERVISED MM-JEPA PRE-TRAINING ({cfg['backbone']} - {trial_name})")
    
    base_tau = cfg['jepa']['tau_base']
    total_steps = len(dataloader) * cfg['epochs']
    global_step = start_epoch * len(dataloader)
    
    for epoch in range(start_epoch, cfg['epochs']):
        model.train()
        epoch_loss = 0.0
        
        loop = tqdm(dataloader, desc=f"Epoch {epoch+1}/{cfg['epochs']} [Pre-Train]")
        for batch in loop:
            x_full = batch['x_full'].to(DEVICE)
            x_visible = batch['x_visible'].to(DEVICE)
            high_res_mask = batch['mask'].to(DEVICE)
            
            optimizer.zero_grad(set_to_none=True)
            
            z_pred, z_target, latent_mask = model(x_visible, x_full, high_res_mask)
            mask_exp = latent_mask.expand(-1, z_pred.shape[1], -1, -1)
            
            z_pred_norm = F.normalize(z_pred, dim=1)
            z_target_norm = F.normalize(z_target, dim=1)
            
            # =================================================================================
            # --- V3 OBJECTIVE UPDATE: CONDITIONAL ALIGNMENT, DECORRELATION & PHYSICS ---
            # =================================================================================
            
            # 1. Target Prediction Loss (Masked regions)
            loss_target = F.mse_loss(z_pred_norm[mask_exp == 1], z_target_norm[mask_exp == 1])
            
            # 2. Context Consistency Loss (Unmasked regions)
            if enable_context:
                loss_context = F.mse_loss(z_pred_norm[mask_exp == 0], z_target_norm[mask_exp == 0])
            else:
                loss_context = torch.tensor(0.0, device=DEVICE)
                
            # 3. Explicit Physical Constraint Loss: Scale-Invariant Depth & Radiometric Thermal Regularization
            raw_depth = x_full[:, 3:4, :, :]
            raw_therm = x_full[:, 4:5, :, :]
            
            stem = model.context_encoder.patch_embed1.proj
            if hasattr(stem, 'depth_scale'):
                d_mean = raw_depth.mean(dim=(2, 3), keepdim=True).clamp(min=1e-5)
                calibrated_depth = ((raw_depth / d_mean) * stem.depth_scale) + stem.depth_bias
                calibrated_therm = (raw_therm * torch.sigmoid(stem.therm_scale)) + stem.therm_bias
                
                # Scale-Invariant Depth Consistency Penalty
                loss_depth_phys = scale_invariant_depth_loss(calibrated_depth, raw_depth)
                # Radiometric Thermal Bounded Regularization
                loss_therm_phys = torch.mean((calibrated_therm - raw_therm) ** 2)
                loss_physics = 0.05 * (loss_depth_phys + loss_therm_phys)
            else:
                loss_physics = torch.tensor(0.0, device=DEVICE)
            
            # --- Fortified Variance & Covariance Regularization ---
            if variance_type == 'spatial':
                target_flat = z_target_norm.transpose(0, 1).reshape(z_target_norm.shape[1], -1)
                
                std_target = torch.sqrt(target_flat.var(dim=1) + 1e-04) 
                loss_var = torch.mean(F.relu(1.0 - std_target))
                
                if enable_covariance:
                    target_flat_centered = target_flat - target_flat.mean(dim=1, keepdim=True)
                    cov_matrix = (target_flat_centered @ target_flat_centered.T) / (target_flat.shape[1] - 1)
                    cov_matrix.fill_diagonal_(0.0)
                    loss_cov = (cov_matrix ** 2).sum() / target_flat.shape[0]
                else:
                    loss_cov = torch.tensor(0.0, device=DEVICE)
                
                loss_reg = loss_var + (0.05 * loss_cov)

            elif variance_type == 'batch':
                target_pooled = z_target_norm.mean(dim=(2, 3))
                
                std_target = torch.sqrt(target_pooled.var(dim=0) + 1e-04)
                loss_var = torch.mean(F.relu(1.0 - std_target))
                
                if enable_covariance:
                    target_centered = target_pooled - target_pooled.mean(dim=0, keepdim=True)
                    cov_matrix = (target_centered.T @ target_centered) / (target_centered.shape[0] - 1)
                    cov_matrix.fill_diagonal_(0.0)
                    loss_cov = (cov_matrix ** 2).sum() / target_pooled.shape[1]
                else:
                    loss_cov = torch.tensor(0.0, device=DEVICE)
                
                loss_reg = loss_var + (0.05 * loss_cov)
            else:
                loss_reg = torch.tensor(0.0, device=DEVICE)
            
            # Final Fortified Objective including V3 Physical Constraints
            loss = loss_target + (0.5 * loss_context) + (0.1 * loss_reg) + loss_physics
            
            loss.backward()
            optimizer.step()
            
            current_tau = 1.0 - (1.0 - base_tau) * (math.cos(math.pi * global_step / total_steps) + 1.0) / 2.0
            model.update_target_network(tau=current_tau)
            
            global_step += 1
            epoch_loss += loss.item()
            loop.set_postfix(loss=loss.item(), tau=f"{current_tau:.5f}")
            
        print(f"Epoch {epoch+1} | Latent MSE: {epoch_loss/len(dataloader):.4f}")
        
        checkpoint = {
            'epoch': epoch + 1,
            'context_encoder_state_dict': model.context_encoder.state_dict(),
            'target_encoder_state_dict': model.target_encoder.state_dict(),
            'predictor_state_dict': model.predictor.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'backbone': cfg['backbone']
        }
        torch.save(checkpoint, os.path.join(weight_dir, f"jepa_checkpoint_{cfg['backbone']}.pt"))
        torch.save(model.context_encoder.state_dict(), os.path.join(weight_dir, f"jepa_context_encoder_{cfg['backbone']}.pt"))

if __name__ == '__main__':
    main()