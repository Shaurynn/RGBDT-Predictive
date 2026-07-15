import os
import re
import glob
import torch
import argparse
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import umap.umap_ as umap
from torch.utils.data import DataLoader
from dataset_jepa import DownstreamSegmentationDataset
import models

# Suppress UMAP multi-threading warnings to maintain clean terminal output
warnings.filterwarnings("ignore", message=".*n_jobs value 1 overridden.*")

def parse_args():
    parser = argparse.ArgumentParser(description="TMLPN Batch Latent Visualization Engine (Dark Mode)")
    parser.add_argument("--model", type=str, default="TMLPN_Downstream_v2", help="Target model architecture directory")
    parser.add_argument("--dataset", type=str, default="MM5", help="Target dataset directory")
    parser.add_argument("--weights", type=str, default=None, help="Optional: Path to a specific model artifact to run in isolation")
    parser.add_argument("--backbone", type=str, default="mit_b5", help="Required only if using --weights to specify the backbone")
    parser.add_argument("--samples", type=int, default=8000, help="Max spatial tokens to sample per projection")
    return parser.parse_args()

def configure_dark_mode():
    """Injects a high-contrast dark aesthetic for spatial token visualization."""
    plt.style.use('dark_background')
    sns.set_theme(
        style="dark", 
        context="paper", 
        rc={
            "axes.facecolor": "#0d1117",       # Deep charcoal background
            "figure.facecolor": "#0d1117", 
            "grid.color": "#30363d",           # Subtle slate gridlines
            "axes.edgecolor": "#30363d",
            "text.color": "#c9d1d9",           # Soft white text
            "xtick.color": "#c9d1d9",
            "ytick.color": "#c9d1d9"
        }
    )

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

def extract_semantic_embeddings(model, dataloader, device, max_samples):
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

def compute_and_plot(features, labels, title, ax_tsne, ax_umap, palette_type, class_names=None):
    tsne_results = TSNE(n_components=2, perplexity=45, random_state=42, max_iter=1000).fit_transform(features)
    umap_results = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine', random_state=42).fit_transform(features)
    
    if palette_type == "semantic" and class_names is not None:
        unique_int_labels = sorted(np.unique(labels).astype(int))
        display_labels = [class_names[int(lbl)] if int(lbl) < len(class_names) else f"Class {lbl}" for lbl in labels]
        hue_order = [class_names[i] if i < len(class_names) else f"Class {i}" for i in unique_int_labels]
        palette = sns.color_palette("nipy_spectral", len(unique_int_labels))
    else:
        display_labels = labels
        hue_order = ["RGB Manifold", "Depth+Thermal Manifold"]
        # Adjusted modality colors to be highly visible against dark backgrounds
        palette = {"RGB Manifold": "#58a6ff", "Depth+Thermal Manifold": "#ff7b72"}

    df = pd.DataFrame({'TSNE_1': tsne_results[:, 0], 'TSNE_2': tsne_results[:, 1], 'UMAP_1': umap_results[:, 0], 'UMAP_2': umap_results[:, 1], 'Label': display_labels})
    
    # Render scatter plots without edge borders to prevent dark-mode blurring
    sns.scatterplot(data=df, x='TSNE_1', y='TSNE_2', hue='Label', hue_order=hue_order, palette=palette, s=15, alpha=0.85, ax=ax_tsne, edgecolor='none')
    ax_tsne.set_title(f"t-SNE: {title}", fontweight='bold', color='white')
    ax_tsne.set_xticks([]); ax_tsne.set_yticks([]) 
    
    sns.scatterplot(data=df, x='UMAP_1', y='UMAP_2', hue='Label', hue_order=hue_order, palette=palette, s=15, alpha=0.85, ax=ax_umap, edgecolor='none')
    ax_umap.set_title(f"UMAP: {title}", fontweight='bold', color='white')
    ax_umap.set_xticks([]); ax_umap.set_yticks([]) 
    
    if palette_type == "semantic":
        # Format legends for dark aesthetic
        for ax in [ax_tsne, ax_umap]:
            legend = ax.legend(title="Structural Class", bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., fontsize='small')
            plt.setp(legend.get_title(), color='white')
            plt.setp(legend.get_texts(), color='white')
            legend.get_frame().set_facecolor('#161b22')
            legend.get_frame().set_edgecolor('#30363d')

def main():
    args = parse_args()
    configure_dark_mode()
    
    if args.weights:
        norm_path = os.path.normpath(args.weights)
        path_parts = norm_path.split(os.sep)
        anchor = "results" if "results" in path_parts else ("weights" if "weights" in path_parts else None)
        
        if anchor:
            anchor_idx = path_parts.index(anchor)
            if len(path_parts) > anchor_idx + 2:
                args.model = path_parts[anchor_idx + 1]
                args.dataset = path_parts[anchor_idx + 2]
                print(f"[*] Isolated run detected. Auto-routed to Model: {args.model} | Dataset: {args.dataset}")
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    splits_root = os.path.join("data", "splits")
    try:
        with open(os.path.join(splits_root, args.dataset, "classes.txt"), "r") as f:
            # FIXED: Corrected Regex to preserve string spaces while dropping bracket tags
            class_names = [re.sub(r'\\s*', '', line).strip() for line in f if line.strip()]
            NUM_CLASSES = len(class_names)
    except FileNotFoundError:
        print(f"[-] CRITICAL: classes.txt not found. Cannot perform latent rendering for {args.dataset}")
        return
        
    eval_dataset = DownstreamSegmentationDataset(dataset_name=args.dataset, split="eval", splits_root=splits_root, image_size=(480, 640))
    eval_loader = DataLoader(eval_dataset, batch_size=4, shuffle=True, num_workers=4)
    
    out_dir = os.path.join("results", args.model, args.dataset, "visualizations")
    os.makedirs(out_dir, exist_ok=True)

    runs_to_process = []
    
    if args.weights:
        is_p1 = "jepa" in os.path.basename(args.weights)
        p1_w = args.weights if is_p1 else None
        p2_w = None if is_p1 else args.weights
        runs_to_process.append((p1_w, p2_w, args.backbone, "Custom_Isolated_Run"))
    else:
        base_dir = os.path.join("results", args.model, args.dataset)
        weight_files = glob.glob(os.path.join(base_dir, "*", "*", "best_model.pt"))
        for wf in weight_files:
            run_name = os.path.basename(os.path.dirname(wf))
            parent_backbone_folder = os.path.basename(os.path.dirname(os.path.dirname(wf)))
            
            match = re.match(r"^(mit_b\d+)(?:_(.*))?$", parent_backbone_folder)
            backbone = match.group(1) if match else "mit_b1"
            trial_name = match.group(2) if (match and match.group(2)) else "baseline"
            
            p1_weights = os.path.join("weights", args.model, args.dataset, trial_name, f"jepa_context_encoder_{backbone}.pt")
            runs_to_process.append((p1_weights, wf, backbone, run_name))

    if not runs_to_process:
        print(f"[-] No valid artifacts found for Model: {args.model} | Dataset: {args.dataset}")
        return

    print(f"\n[*] Booting Latent Visualization Engine (Found {len(runs_to_process)} target executions)")
    
    for p1_weights, p2_weights, backbone, run_name in runs_to_process:
        print("\n" + "="*70)
        print(f"[*] Processing: {run_name} (Backbone: {backbone})")
        print("="*70)
        
        has_p1 = p1_weights and os.path.exists(p1_weights)
        has_p2 = p2_weights and os.path.exists(p2_weights)
        
        cols = 4 if (has_p1 and has_p2) else 2
        # Use #0d1117 (Deep Charcoal) to match the dark aesthetic matrix
        fig, axes = plt.subplots(2, cols, figsize=(8*cols, 16), facecolor="#0d1117")
        if cols == 2: axes = axes.reshape(2, 2)
        
        try:
            if has_p1:
                print("    -> Extracting Phase 1 (Foundation) manifolds...")
                model_p1 = models.TMLPN_Downstream_v2(num_classes=NUM_CLASSES, backbone_name=backbone, use_lora=False).to(DEVICE)
                
                raw_checkpoint = torch.load(p1_weights, map_location=DEVICE)
                ce_weights = raw_checkpoint.get('context_encoder_state_dict', raw_checkpoint)
                model_p1.load_state_dict({f"context_encoder.{k}": v for k, v in ce_weights.items()}, strict=False)
                
                m_f1, m_l1 = extract_modality_embeddings(model_p1, eval_loader, DEVICE, args.samples)
                s_f1, s_l1 = extract_semantic_embeddings(model_p1, eval_loader, DEVICE, args.samples)
                
                col_offset = 0
                compute_and_plot(m_f1, m_l1, "Phase 1: Modality Alignment", axes[0, col_offset], axes[0, col_offset+1], "modality", class_names)
                compute_and_plot(s_f1, s_l1, "Phase 1: Semantic Foundation", axes[1, col_offset], axes[1, col_offset+1], "semantic", class_names)
                
            if has_p2:
                print("    -> Extracting Phase 2 (Fine-Tuned) manifolds...")
                model_p2 = models.TMLPN_Downstream_v2(num_classes=NUM_CLASSES, backbone_name=backbone, use_lora=False).to(DEVICE)
                model_p2.load_state_dict(torch.load(p2_weights, map_location=DEVICE), strict=False)
                
                m_f2, m_l2 = extract_modality_embeddings(model_p2, eval_loader, DEVICE, args.samples)
                s_f2, s_l2 = extract_semantic_embeddings(model_p2, eval_loader, DEVICE, args.samples)
                
                col_offset = 2 if (has_p1 and has_p2) else 0
                compute_and_plot(m_f2, m_l2, "Phase 2: Modality Alignment", axes[0, col_offset], axes[0, col_offset+1], "modality", class_names)
                compute_and_plot(s_f2, s_l2, "Phase 2: Semantic Separation", axes[1, col_offset], axes[1, col_offset+1], "semantic", class_names)
                
            plt.tight_layout(pad=3.0)
            save_path = os.path.join(out_dir, f"manifold_progression_{run_name}.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor="#0d1117")
            plt.close()
            print(f"[+] Rendered progression exported to: {save_path}")
            
        except Exception as e:
            print(f"    -> [!] Extraction failed for {run_name}: {e}")

if __name__ == "__main__":
    main()