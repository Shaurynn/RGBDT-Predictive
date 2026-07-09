import os
import argparse
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from models import MultimodalJEPA
from dataset_jepa import JEPAPretrainDataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    parser = argparse.ArgumentParser(description="Phase 1: MM-JEPA Self-Supervised Pre-Training")
    parser.add_argument("--backbone", type=str, default="mit_b1", help="Vision Transformer backbone")
    parser.add_argument("--dataset", type=str, default="MM5", help="Path to MM5 dataset")
    parser.add_argument("--mask_ratio", type=float, default=0.60, help="Percentage of the 5-channel block to mask")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for pre-training")
    parser.add_argument("--epochs", type=int, default=100, help="Total pre-training epochs")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint file (.pt) to resume pretraining")
    args = parser.parse_args()

    data_dir = os.path.join("dataset", args.dataset)
    dataset = JEPAPretrainDataset(data_dir=data_dir, image_size=(480, 640))
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    
    model = MultimodalJEPA(backbone_name=args.backbone).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.04)
    
    # Create a backbone-specific subfolder for workspace hygiene
    weight_dir = os.path.join("weights", args.dataset)
    os.makedirs(weight_dir, exist_ok=True)
    
    start_epoch = 0
    if args.resume and os.path.exists(args.resume):
        print(f"[*] Resuming pretraining from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=DEVICE)
        model.context_encoder.load_state_dict(checkpoint['context_encoder_state_dict'])
        model.target_encoder.load_state_dict(checkpoint['target_encoder_state_dict'])
        model.predictor.load_state_dict(checkpoint['predictor_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        print(f"[*] Successfully restored state. Resuming at epoch {start_epoch}")
    
    print(f"\n🚀 INITIATING PHASE 1: SELF-SUPERVISED MM-JEPA PRE-TRAINING ({args.backbone})")
    for epoch in range(start_epoch, args.epochs):
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
            
            loss = F.mse_loss(z_pred[mask_exp == 1], z_target[mask_exp == 1])
            
            loss.backward()
            optimizer.step()
            model.update_target_network(tau=0.996)
            
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