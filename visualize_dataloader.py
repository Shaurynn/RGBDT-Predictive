import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from dataset import TriModalPredictiveDataset
import torchvision.transforms.functional as TF

def denormalize(tensor, mean, std):
    """Reverses the normalization for visualization purposes."""
    tensor = tensor.clone()
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return torch.clamp(tensor, 0, 1)

def run_diagnostic(output_dir="diagnostics"):
    """
    Pulls a single batch from the dataloader and generates a visual grid
    proving the JEPA masking and tensors are functioning correctly.
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"[*] Initializing Dataloader Diagnostic Tool...")
    
    # Initialize the dataset with a 30% mask ratio
    dataset = TriModalPredictiveDataset(data_dir='dataset/MM5', split='train', mask_ratio=0.30)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    batch = next(iter(dataloader))
    
    # Unpack the dictionary
    rgbd = batch['rgbd']                 # [B, 4, H, W]
    therm_masked = batch['therm_masked'] # [B, 1, H, W]
    therm_target = batch['therm_target'] # [B, 1, H, W]
    block_mask = batch['block_mask']     # [B, 1, H, W]
    seg_mask = batch['seg_mask']         # [B, H, W]
    
    # Known normalization stats from dataset.py
    rgb_mean, rgb_std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    therm_mean, therm_std = [0.500], [0.250]
    
    batch_size = rgbd.size(0)
    fig, axes = plt.subplots(batch_size, 5, figsize=(20, 4 * batch_size))
    
    col_titles = ["RGB", "Depth", "Pristine Thermal", "The Block Mask", "Network Input (Masked)"]
    for j in range(5):
        axes[0, j].set_title(col_titles[j], fontsize=14, fontweight='bold')

    for i in range(batch_size):
        # 1. Extract and denormalize RGB (Channels 0, 1, 2)
        rgb_vis = denormalize(rgbd[i, :3, :, :], rgb_mean, rgb_std)
        rgb_vis = rgb_vis.permute(1, 2, 0).numpy()
        
        # 2. Extract and denormalize Depth (Channel 3)
        depth_vis = denormalize(rgbd[i, 3:4, :, :], therm_mean, therm_std)
        depth_vis = depth_vis.squeeze().numpy()
        
        # 3. Denormalize Pristine Thermal Target
        therm_target_vis = denormalize(therm_target[i], therm_mean, therm_std).squeeze().numpy()
        
        # 4. Extract Block Mask (Binary, no denorm needed)
        mask_vis = block_mask[i].squeeze().numpy()
        
        # 5. Denormalize the actual input going to Backbone B
        therm_masked_vis = denormalize(therm_masked[i], therm_mean, therm_std).squeeze().numpy()
        
        # Plotting
        axes[i, 0].imshow(rgb_vis)
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(depth_vis, cmap='magma')
        axes[i, 1].axis('off')
        
        axes[i, 2].imshow(therm_target_vis, cmap='inferno')
        axes[i, 2].axis('off')
        
        axes[i, 3].imshow(mask_vis, cmap='gray')
        axes[i, 3].axis('off')
        
        axes[i, 4].imshow(therm_masked_vis, cmap='inferno')
        axes[i, 4].axis('off')

    plt.tight_layout()
    save_path = os.path.join(output_dir, "jepa_dataloader_sanity_check.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Unique values in masks: {torch.unique(seg_mask)}")
    print(f"[SUCCESS] Diagnostic grid saved to: {save_path}")

if __name__ == '__main__':
    run_diagnostic()