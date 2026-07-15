import os
import re
import glob
import torch
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import umap.umap_ as umap
from torch.utils.data import DataLoader
from dataset_jepa import DownstreamSegmentationDataset
import models
import warnings
warnings.filterwarnings("ignore", message=".*n_jobs value 1 overridden.*")

def parse_args():
    parser = argparse.ArgumentParser(description="TMLPN Batch Latent Visualization Engine")
    parser.add_argument("--model", type=str, default="TMLPN_Downstream_v2", help="Target model architecture directory")
    parser.add_argument("--dataset", type=str, default="MM5", help="Target dataset directory")
    parser.add_argument("--weights", type=str, default=None, help="Optional: Path to a specific best_model.pt to run in isolation")
    parser.add_argument("--backbone", type=str, default="mit_b5", help="Required only if using --weights to specify the backbone")
    parser.add_argument("--samples", type=int, default=8000, help="Max spatial tokens to sample per projection")
    return parser.parse_args()

def extract_modality_embeddings(model, dataloader, device, max_samples):
    model.eval()
    rgb_features_list, dt_features_list = [], []
    with torch.no_grad():
        for batch in dataloader:
            x_full = batch['x_full'].to(device)
            stem = model.context_encoder.patch_embed1.proj
            
            rgb_aligned = stem.rgb_proj(x_full[:, :3, :, :])
            dt_raw = stem.depth_therm_proj((x_full[:, 3:, :, :] * stem.dt_scale) + stem.dt_bias)
            dt_aligned = stem.dt_alignment(dt_raw)
            
            rgb_features_list.append(rgb_aligned.permute(0, 2, 3, 1).reshape(-1, rgb_aligned.shape[1]).cpu().numpy())
            dt_features_list.append(dt_aligned.permute(0, 2, 3, 1).reshape(-1, dt_aligned.shape[1]).cpu().numpy())
            
            if sum(len(f) for f in rgb_features_list) >= (max_samples // 2): break
                
    rgb_np, dt_np = np.concatenate(rgb_features_list, axis=0), np.concatenate(dt_features_list, axis=0)
    sample_size = min(len(rgb_np), max_samples // 2)
    rgb_idx = np.random.choice(len(rgb_np), sample_size, replace=False)
    dt_idx = np.random.choice(len(dt_np), sample_size, replace=False)
    
    return np.vstack((rgb_np[rgb_idx], dt_np[dt_idx])), np.array(['RGB Manifold'] * sample_size + ['Depth+Thermal Manifold'] * sample_size)

def extract_semantic_embeddings(model, dataloader, device, num_classes, max_samples):
    model.eval()
    semantic_features, semantic_labels = [], []
    with torch.no_grad():
        for batch in dataloader:
            x_full, seg_mask = batch['x_full'].to(device), batch['seg_mask'].to(device)
            _, features = model(x_full, return_features=True)
            c4_latent = features[-1] 
            
            mask_down = torch.nn.functional.interpolate(seg_mask.unsqueeze(1).float(), size=c4_latent.shape[2:], mode='nearest').squeeze(1).long()
            c4_flat, mask_flat = c4_latent.permute(0, 2, 3, 1).reshape(-1, c4_latent.shape[1]), mask_down.reshape(-1)
            
            valid_idx = (mask_flat != 255)
            semantic_features.append(c4_flat[valid_idx].cpu().numpy())
            semantic_labels.append(mask_flat[valid_idx].cpu().numpy())
            
            if sum(len(f) for f in semantic_features) >= max_samples: break
                
    f_np, l_np = np.concatenate(semantic_features, axis=0), np.concatenate(semantic_labels, axis=0)
    if len(f_np) > max_samples:
        indices = np.random.choice(len(f_np), max_samples, replace=False)
        return f_np[indices], l_np[indices]
    return f_np, l_np

def compute_and_plot(features, labels, title, ax_tsne, ax_umap, palette_type):
    tsne_results = TSNE(n_components=2, perplexity=45, random_state=42, max_iter=1000).fit_transform(features)
    umap_results = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine', random_state=42).fit_transform(features)
    
    df = pd.DataFrame({'TSNE_1': tsne_results[:, 0], 'TSNE_2': tsne_results[:, 1], 'UMAP_1': umap_results[:, 0], 'UMAP_2': umap_results[:, 1], 'Label': labels})
    palette = {"RGB Manifold": "#3498db", "Depth+Thermal Manifold": "#e74c3c"} if palette_type == "modality" else sns.color_palette("husl", len(np.unique(labels)))
        
    sns.scatterplot(data=df, x='TSNE_1', y='TSNE_2', hue='Label', palette=palette, s=12, alpha=0.7, ax=ax_tsne, edgecolor=None)
    ax_tsne.set_title(f"t-SNE: {title}", fontweight='bold'); ax_tsne.set_xticks([]); ax_tsne.set_yticks([]) 
    
    sns.scatterplot(data=df, x='UMAP_1', y='UMAP_2', hue='Label', palette=palette, s=12, alpha=0.7, ax=ax_umap, edgecolor=None)
    ax_umap.set_title(f"UMAP: {title}", fontweight='bold'); ax_umap.set_xticks([]); ax_umap.set_yticks([]) 

def main():
    args = parse_args()
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    splits_root = os.path.join("data", "splits")
    with open(os.path.join(splits_root, args.dataset, "classes.txt"), "r") as f:
        NUM_CLASSES = len([line for line in f if line.strip()])
        
    eval_dataset = DownstreamSegmentationDataset(dataset_name=args.dataset, split="eval", splits_root=splits_root, image_size=(480, 640))
    eval_loader = DataLoader(eval_dataset, batch_size=4, shuffle=True, num_workers=4)
    
    out_dir = os.path.join("results", args.model, args.dataset, "visualizations")
    os.makedirs(out_dir, exist_ok=True)

    runs_to_process = []
    if args.weights:
        runs_to_process.append((args.weights, args.backbone, "Custom_Isolated_Run"))
    else:
        base_dir = os.path.join("results", args.model, args.dataset)
        # Added the extra '/*' depth to reach inside the timestamped run folders
        weight_files = glob.glob(os.path.join(base_dir, "*", "*", "best_model.pt"))
        for wf in weight_files:
            run_name = os.path.basename(os.path.dirname(wf))
            parent_backbone_folder = os.path.basename(os.path.dirname(os.path.dirname(wf)))
            match = re.match(r"^(mit_b\d+)", parent_backbone_folder)
            backbone = match.group(1) if match else "mit_b1"
            runs_to_process.append((wf, backbone, run_name))

    if not runs_to_process:
        print(f"[-] No valid weights found for Model: {args.model} | Dataset: {args.dataset}")
        return

    print(f"\n[*] Booting Latent Visualization Engine (Found {len(runs_to_process)} target models)")
    
    for weights_path, backbone, run_name in runs_to_process:
        print("\n" + "="*70)
        print(f"[*] Processing: {run_name} (Backbone: {backbone})")
        print("="*70)
        
        model = models.TMLPN_Downstream_v2(num_classes=NUM_CLASSES, backbone_name=backbone, use_lora=False).to(DEVICE)
        model.load_state_dict(torch.load(weights_path, map_location=DEVICE), strict=False)
        
        modality_features, modality_labels = extract_modality_embeddings(model, eval_loader, DEVICE, args.samples)
        semantic_features, semantic_labels = extract_semantic_embeddings(model, eval_loader, DEVICE, NUM_CLASSES, args.samples)
        
        sns.set_theme(style="white", context="paper")
        fig, axes = plt.subplots(2, 2, figsize=(18, 16))
        
        compute_and_plot(modality_features, modality_labels, "Modality Alignment (Stem Output)", axes[0, 0], axes[0, 1], "modality")
        compute_and_plot(semantic_features, semantic_labels, "Semantic Separation (c4 Deep Features)", axes[1, 0], axes[1, 1], "semantic")
        
        plt.tight_layout(pad=3.0)
        save_path = os.path.join(out_dir, f"manifold_{run_name}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[+] Rendered projection exported to: {save_path}")

if __name__ == "__main__":
    main()