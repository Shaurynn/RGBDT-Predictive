import os
import math
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from models import MultimodalJEPA
from dataset_jepa import JEPAPretrainDataset
from config_utils import parse_with_config # Import the new utility

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    args, config = parse_with_config("Phase 1: MM-JEPA Self-Supervised Pre-Training")
    
    # --- Extracted Configuration Parameters ---
    cfg = config['phase1_pretraining']
    img_size = (config['dataset']['image_height'], config['dataset']['image_width'])
    dataset_name = config['dataset']['name']

    data_dir = os.path.join("dataset", dataset_name)
    dataset = JEPAPretrainDataset(data_dir=data_dir, image_size=img_size)
    dataloader = DataLoader(dataset, batch_size=cfg['batch_size'], shuffle=True, num_workers=4, pin_memory=True)
    
    model = MultimodalJEPA(backbone_name=cfg['backbone']).to(DEVICE)
    
    # Magic numbers 1e-4 and 0.04 replaced with config references
    optimizer = optim.AdamW(model.parameters(), lr=cfg['optimizer']['lr'], weight_decay=cfg['optimizer']['weight_decay'])
    
    weight_dir = os.path.join("weights", dataset_name)
    os.makedirs(weight_dir, exist_ok=True)
    
    start_epoch = 0
    # ... (Checkpoint loading logic remains unchanged)
    
    print(f"\n🚀 INITIATING PHASE 1: SELF-SUPERVISED MM-JEPA PRE-TRAINING ({cfg['backbone']})")
    
    # Magic tau=0.996 replaced
    base_tau = cfg['jepa']['tau_base']
    total_steps = len(dataloader) * cfg['epochs']
    global_step = start_epoch * len(dataloader)
    
    for epoch in range(start_epoch, cfg['epochs']):
        model.train()
        epoch_loss = 0.0
        
        loop = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs} [Pre-Train]")
        for batch in loop:
            x_full = batch['x_full'].to(DEVICE)
            x_visible = batch['x_visible'].to(DEVICE)
            high_res_mask = batch['mask'].to(DEVICE)
            
            optimizer.zero_grad(set_to_none=True)
            
            z_pred, z_target, latent_mask = model(x_visible, x_full, high_res_mask)
            mask_exp = latent_mask.expand(-1, z_pred.shape[1], -1, -1)
            
            # --- 1. L2 Feature Normalization ---
            # Projects embeddings onto a unit hypersphere to strictly bound the MSE magnitude,
            # ensuring gradient stability regardless of architectural channel depth.
            z_pred_norm = F.normalize(z_pred, dim=1)
            z_target_norm = F.normalize(z_target, dim=1)
            
            # --- 2. Dual-Objective MSE Loss ---
            # (Note: PyTorch's F.mse_loss inherently normalizes by element count via reduction='mean')
            loss_target = F.mse_loss(z_pred_norm[mask_exp == 1], z_target_norm[mask_exp == 1])
            loss_context = F.mse_loss(z_pred_norm[mask_exp == 0], z_target_norm[mask_exp == 0])
            
            # --- 3. Variance Hinge Regularization (VICReg style) ---
            # Flatten target spatial and batch dimensions to compute per-channel variance
            target_flat = z_target_norm.transpose(0, 1).reshape(z_target_norm.shape[1], -1)
            
            # Add epsilon for numerical stability before sqrt to prevent NaN gradients
            std_target = torch.sqrt(target_flat.var(dim=1) + 1e-04) 
            
            # Hinge penalty forcing standard deviation >= 1.0 to prevent dimensional collapse
            loss_var = torch.mean(F.relu(1.0 - std_target))
            
            # --- Total Loss Objective ---
            # Down-weight auxiliary losses to prevent overpowering the primary inference task
            loss = loss_target + (0.1 * loss_context) + (0.1 * loss_var)
            
            loss.backward()
            optimizer.step()
            
            # --- Dynamic EMA Cosine Schedule ---
            # Anneals tau from 0.996 to 1.0 to prevent early representational collapse 
            # while allowing the context encoder to stabilize.
            current_tau = 1.0 - (1.0 - base_tau) * (math.cos(math.pi * global_step / total_steps) + 1.0) / 2.0
            model.update_target_network(tau=current_tau)
            
            global_step += 1
            epoch_loss += loss.item()
            loop.set_postfix(loss=loss.item(), tau=f"{current_tau:.5f}")
            
            epoch_loss += loss.item()
            loop.set_postfix(mse=loss.item())
            
        print(f"Epoch {epoch+1} | Latent MSE: {epoch_loss/len(dataloader):.4f}")
        
        # Save checkpoints directly into the backbone subfolder
        checkpoint = {
            'epoch': epoch + 1,
            'context_encoder_state_dict': model.context_encoder.state_dict(),
            'target_encoder_state_dict': model.target_encoder.state_dict(),
            'predictor_state_dict': model.predictor.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'backbone': args.backbone
        }
        torch.save(checkpoint, os.path.join(weight_dir, f"jepa_checkpoint_{args.backbone}.pt"))
        torch.save(model.context_encoder.state_dict(), os.path.join(weight_dir, f"jepa_context_encoder_{args.backbone}.pt"))

if __name__ == '__main__':
    main()