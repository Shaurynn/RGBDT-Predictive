import os
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from models import MultimodalJEPA
from dataset_jepa import JEPAPretrainDataset
import argparse

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    parser = argparse.ArgumentParser(description="Phase 1: MM-JEPA Self-Supervised Pre-Training")
    parser.add_argument("--backbone", type=str, default="mit_b1", help="Vision Transformer backbone (e.g., mit_b1, mit_b4)")
    parser.add_argument("--data_dir", type=str, default="dataset/MM5", help="Path to MM5 dataset")
    parser.add_argument("--mask_ratio", type=float, default=0.60, help="Percentage of the 5-channel block to mask")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for pre-training")
    parser.add_argument("--epochs", type=int, default=100, help="Total pre-training epochs")
    args = parser.parse_args()

    dataset = JEPAPretrainDataset(data_dir=args.data_dir, mask_ratio=args.mask_ratio)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    
    model = MultimodalJEPA(backbone_name=args.backbone).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.04)
    
    os.makedirs("weights", exist_ok=True)
    
    print(f"\n🚀 INITIATING PHASE 1: SELF-SUPERVISED MM-JEPA PRE-TRAINING ({args.backbone})")
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        
        loop = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs} [Pre-Train]")
        for batch in loop:
            x_full = batch['x_full'].to(DEVICE)
            x_visible = batch['x_visible'].to(DEVICE)
            high_res_mask = batch['mask'].to(DEVICE)
            
            optimizer.zero_grad(set_to_none=True)
            
            # # 1. True JEPA Forward Pass
            # z_pred, z_target = model(x_visible, x_full)
            
            # # 2. Downsample mask to match latent dimensions
            # B, C, H, W = z_pred.shape
            # latent_mask = F.interpolate(high_res_mask, size=(H, W), mode='nearest').expand(-1, C, -1, -1)
            
            # # 3. Information Bottleneck: Compute MSE ONLY on the masked/predicted regions
            # loss = F.mse_loss(z_pred[latent_mask == 1], z_target[latent_mask == 1])
            z_pred, z_target, latent_mask = model(x_visible, x_full, high_res_mask)
            mask_exp = latent_mask.expand(-1, z_pred.shape[1], -1, -1)
            loss = F.mse_loss(z_pred[mask_exp == 1], z_target[mask_exp == 1])
            
            loss.backward()
            optimizer.step()
            
            # 4. Target Encoder Momentum Update
            model.update_target_network(tau=0.996)
            
            epoch_loss += loss.item()
            loop.set_postfix(mse=loss.item())
            
        print(f"Epoch {epoch+1} | Latent MSE: {epoch_loss/len(dataloader):.4f}")
        torch.save(model.context_encoder.state_dict(), f"weights/jepa_context_encoder_{args.backbone}.pt")

if __name__ == '__main__':
    main()