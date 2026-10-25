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
from dataset_jepa import JEPAPretrainDataset, seed_worker
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


def compute_vicreg_regularization(z_online, variance_type='spatial', enable_covariance=True, eps=1e-4):
    """
    Computes Variance and Covariance regularization on the unnormalized online
    predictor representations (z_pred) to prevent dimensional collapse with active gradients.
    Reference: Bardes et al., VICReg, ICLR 2022.
    """
    B, C, H, W = z_online.shape

    if variance_type == 'spatial':
        # Flatten across batch and spatial coordinates: [C, B * H * W]
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
        # Pool across spatial coordinates: [B, C]
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
    
    model_name = getattr(args, 'model', config.get('metadata', {}).get('model', 'TMLPN_Downstream_v3'))
    
    trial_name = cfg.get('trial_name', 'baseline')
    active_seed = cfg.get('seed', 42)
    enforce_reproducibility(active_seed)
    print(f"[*] Locked PyTorch computational graph to Seed: {active_seed}")

    ablation_cfg = cfg.get('ablations', {})
    enable_isolation = ablation_cfg.get('enable_modality_isolation', True)
    variance_type = ablation_cfg.get('variance_type', 'spatial')
    mask_strategy = ablation_cfg.get('mask_strategy', 'multi_block')
    
    enable_context = ablation_cfg.get('enable_context_consistency', True)
    enable_covariance = ablation_cfg.get('enable_covariance_penalty', True)
    enable_dirac = ablation_cfg.get('enable_dirac', True)
    enable_scse = ablation_cfg.get('enable_scse', True)
    enable_modality_dropout = ablation_cfg.get('enable_modality_dropout', True)
    
    modality_dropout_prob = cfg.get('modality_dropout_prob', 0.15) if enable_modality_dropout else 0.0

    print("\n" + "="*60)
    print("🔬 ABLATION STATE [PHASE 1: PRE-TRAINING (V3 PHYSICAL PRIORS)]")
    print("="*60)
    print(f"[*] Target Architecture Namespace      : {model_name}")
    print(f"[*] Modality Isolated Stem (1x1 Dirac) : {enable_isolation} (Dirac: {enable_dirac})")
    print(f"[*] Dynamic Cross-Modal Gating (scSE)  : {enable_scse}")
    print(f"[*] Variance Regularization Topology   : {variance_type.upper()}")
    print(f"[*] Masking Strategy                   : {mask_strategy.upper()}")
    print(f"[*] Context Consistency Evaluated      : {enable_context}")
    print(f"[*] Covariance Penalty Evaluated       : {enable_covariance}")
    print(f"[*] Modality Dropout Regularizer       : {enable_modality_dropout} (Rate: {modality_dropout_prob})")
    print("="*60 + "\n")

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
        pin_memory=True, 
        drop_last=True,
        worker_init_fn=seed_worker,
        generator=dl_generator
    )
    
    model = MultimodalJEPA(
        backbone_name=cfg['backbone'], 
        isolated_stem=enable_isolation, 
        enable_dirac=enable_dirac,
        enable_scse=enable_scse
    ).to(DEVICE)
    
    optimizer = optim.AdamW(model.parameters(), lr=cfg['optimizer']['lr'], weight_decay=cfg['optimizer']['weight_decay'])
    
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
            
            # 1. Compute Variance & Covariance Regularization on unnormalized online z_pred
            loss_reg = compute_vicreg_regularization(
                z_pred, 
                variance_type=variance_type, 
                enable_covariance=enable_covariance
            )

            # 2. L2-Normalize Latent Representations for Direction-Aligned Prediction Loss
            z_pred_norm = F.normalize(z_pred, dim=1, eps=1e-8)
            z_target_norm = F.normalize(z_target, dim=1, eps=1e-8)
            
            if (mask_exp == 1).any():
                loss_target = F.mse_loss(z_pred_norm[mask_exp == 1], z_target_norm[mask_exp == 1])
            else:
                loss_target = F.mse_loss(z_pred_norm, z_target_norm)
            
            if enable_context and (mask_exp == 0).any():
                loss_context = F.mse_loss(z_pred_norm[mask_exp == 0], z_target_norm[mask_exp == 0])
            else:
                loss_context = torch.tensor(0.0, device=DEVICE)
                
            # 3. Physical Calibration Parameter Anchoring Prior
            stem = model.context_encoder.patch_embed1.proj
            if hasattr(stem, 'depth_scale'):
                loss_depth_prior = ((stem.depth_scale - 1.0) ** 2).mean() + (stem.depth_bias ** 2).mean()
                loss_therm_prior = ((stem.therm_scale - 1.0) ** 2).mean() + (stem.therm_bias ** 2).mean()
                loss_physics = 0.01 * (loss_depth_prior + loss_therm_prior)
            else:
                loss_physics = torch.tensor(0.0, device=DEVICE)
            
            loss = loss_target + (0.5 * loss_context) + (0.1 * loss_reg) + loss_physics
            
            loss.backward()
            optimizer.step()
            
            current_tau = 1.0 - (1.0 - base_tau) * (math.cos(math.pi * global_step / total_steps) + 1.0) / 2.0
            model.update_target_network(tau=current_tau)
            
            global_step += 1
            epoch_loss += loss.item()
            loop.set_postfix(loss=loss.item(), tau=f"{current_tau:.5f}")
            
        print(f"Epoch {epoch+1} | Latent Objective Loss: {epoch_loss/len(dataloader):.4f}")
        
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