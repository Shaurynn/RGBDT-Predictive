import os
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from models import MultimodalJEPA
from dataset_jepa import JEPAPretrainDataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    dataset = JEPAPretrainDataset(data_dir="dataset/MM5", mask_ratio=0.60)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=4, pin_memory=True)
    
    model = MultimodalJEPA(backbone_name='mit_b1').to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.04)
    
    os.makedirs("weights", exist_ok=True)
    
    print("\n🚀 INITIATING PHASE 1: SELF-SUPERVISED MM-JEPA PRE-TRAINING")
    for epoch in range(100):
        model.train()
        epoch_loss = 0.0
        
        loop = tqdm(dataloader, desc=f"Epoch {epoch+1}/100 [Pre-Train]")
        for batch in loop:
            x_full = batch['x_full'].to(DEVICE)
            x_visible = batch['x_visible'].to(DEVICE)
            high_res_mask = batch['mask'].to(DEVICE)
            
            optimizer.zero_grad(set_to_none=True)
            
            # 1. True JEPA Forward Pass
            z_pred, z_target = model(x_visible, x_full)
            
            # 2. Downsample mask to match latent dimensions (e.g., 512x15x20)
            B, C, H, W = z_pred.shape
            latent_mask = F.interpolate(high_res_mask, size=(H, W), mode='nearest').expand(-1, C, -1, -1)
            
            # 3. Information Bottleneck: Compute MSE ONLY on the masked/predicted regions
            loss = F.mse_loss(z_pred[latent_mask == 1], z_target[latent_mask == 1])
            
            loss.backward()
            optimizer.step()
            
            # 4. Target Encoder Momentum Update
            model.update_target_network(tau=0.996)
            
            epoch_loss += loss.item()
            loop.set_postfix(mse=loss.item())
            
        print(f"Epoch {epoch+1} | Latent MSE: {epoch_loss/len(dataloader):.4f}")
        torch.save(model.context_encoder.state_dict(), f"weights/jepa_context_encoder.pt")

if __name__ == '__main__':
    main()