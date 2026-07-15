import os
import re
import glob
import json
import argparse
import warnings
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.manifold import TSNE
import umap.umap_ as umap
from torch.utils.data import DataLoader

from dataset_jepa import DownstreamSegmentationDataset
import models

# Suppress UMAP multi-threading warnings to maintain clean terminal output
warnings.filterwarnings("ignore", message=".*n_jobs value 1 overridden.*")

def parse_args():
    parser = argparse.ArgumentParser(description="TMLPN Unified CVPR Evaluation & Latent Visualization Engine")
    parser.add_argument("--model", type=str, default="TMLPN_Downstream_v2", help="Target architecture directory (e.g., TMLPN_Downstream_v2)")
    parser.add_argument("--dataset", type=str, default="MM5", help="Target dataset name")
    parser.add_argument("--samples", type=int, default=8000, help="Max spatial tokens to sample per latent projection")
    parser.add_argument("--skip_latents", action="store_true", help="Flag to bypass the heavy UMAP/t-SNE rendering")
    return parser.parse_args()

# ====================================================================================
# --- PART 1: METRIC AGGREGATION & STATISTICAL PLOTTING ---
# ====================================================================================

def generate_metric_visualizations(primary_baselines, ablation_stats, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)

    if primary_baselines:
        plt.figure(figsize=(9, 6))
        trajectory_records = []
        for backbone, phases in primary_baselines.items():
            for phase in ["baseline", "hero", "microtune"]:
                if phase in phases:
                    trajectory_records.append({"Backbone": backbone, "Phase": phase.capitalize(), "mIoU": phases[phase]["mIoU"]})
        df_traj = pd.DataFrame(trajectory_records)
        if not df_traj.empty:
            ax = sns.lineplot(data=df_traj, x='Phase', y='mIoU', hue='Backbone', marker='o', linewidth=2.5, markersize=8, palette='viridis')
            ax.set_title("Architecture Progression Trajectory", fontweight='bold')
            ax.set_ylabel("Mean Intersection over Union (mIoU)")
            ax.set_xlabel("Training Phase")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "phase_progression_trajectory.png"), dpi=300)
            plt.close()

    if ablation_stats:
        plt.figure(figsize=(10, 6))
        ablation_records = [{"Variant": trial.replace("Ablation_", "").replace("_", " "), "Mean mIoU": stats_data["mean_mIoU"], "Std Dev": stats_data["std_dev"]} for trial, stats_data in ablation_stats.items()]
        df_abl = pd.DataFrame(ablation_records)
        ax = sns.barplot(x='Variant', y='Mean mIoU', data=df_abl, palette='mako')
        ax.errorbar(x=range(len(df_abl)), y=df_abl['Mean mIoU'], yerr=df_abl['Std Dev'], fmt='none', c='black', capsize=5, elinewidth=1.5)
        ax.set_title("Ablation Study Results (N Seeds)", fontweight='bold')
        ax.set_ylim(0, max(df_abl['Mean mIoU']) * 1.15)
        ax.set_ylabel("Mean mIoU")
        plt.xticks(rotation=15)
        for i, row in df_abl.iterrows():
            ax.annotate(f"{row['Mean mIoU']:.4f}\n±{row['Std Dev']:.4f}", (i, row['Mean mIoU'] + row['Std Dev']), ha='center', va='bottom', fontsize=10, xytext=(0, 8), textcoords='offset points')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "ablation_statistics.png"), dpi=300)
        plt.close()

# ====================================================================================
# --- PART 2: LATENT SPACE EXTRACTION & PROJECTION ---
# ====================================================================================

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

def compute_and_plot_latents(features, labels, title, ax_tsne, ax_umap, palette_type, class_names=None):
    tsne_results = TSNE(n_components=2, perplexity=45, random_state=42, max_iter=1000).fit_transform(features)
    umap_results = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine', random_state=42).fit_transform(features)
    
    if palette_type == "semantic" and class_names is not None:
        display_labels = [class_names[int(lbl)] if int(lbl) < len(class_names) else f"Class {lbl}" for lbl in labels]
    else:
        display_labels = labels

    df = pd.DataFrame({'TSNE_1': tsne_results[:, 0], 'TSNE_2': tsne_results[:, 1], 'UMAP_1': umap_results[:, 0], 'UMAP_2': umap_results[:, 1], 'Label': display_labels})
    
    if palette_type == "modality":
        palette = {"RGB Manifold": "#3498db", "Depth+Thermal Manifold": "#e74c3c"}
    else:
        palette = sns.color_palette("nipy_spectral", len(np.unique(display_labels)))
        
    sns.scatterplot(data=df, x='TSNE_1', y='TSNE_2', hue='Label', palette=palette, s=12, alpha=0.7, ax=ax_tsne, edgecolor=None)
    ax_tsne.set_title(f"t-SNE: {title}", fontweight='bold'); ax_tsne.set_xticks([]); ax_tsne.set_yticks([]) 
    if palette_type == "semantic":
        ax_tsne.legend(title="Structural Class", bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., fontsize='small')
    
    sns.scatterplot(data=df, x='UMAP_1', y='UMAP_2', hue='Label', palette=palette, s=12, alpha=0.7, ax=ax_umap, edgecolor=None)
    ax_umap.set_title(f"UMAP: {title}", fontweight='bold'); ax_umap.set_xticks([]); ax_umap.set_yticks([]) 
    if palette_type == "semantic":
        ax_umap.legend(title="Structural Class", bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., fontsize='small')

# ====================================================================================
# --- MAIN EVALUATION ENGINE ---
# ====================================================================================

def main():
    args = parse_args()
    base_dir = os.path.join("results", args.model, args.dataset)
    if not os.path.exists(base_dir):
        print(f"[-] CRITICAL: Target directory not found at {base_dir}")
        return
        
    print(f"\n[*] Booting Unified MLOps Evaluation Engine for {args.model} | {args.dataset}...")

    # ---------------------------------------------------------
    # STAGE 1: METRICS & STATISTICAL ANALYSIS
    # ---------------------------------------------------------
    primary_baselines, ablation_trials = {}, {}
    ablation_pattern = re.compile(r"^(mit_b\d+)_(.*)_seed(\d+)$")
    json_files = glob.glob(os.path.join(base_dir, "*", "*", "results.json"))

    for json_path in json_files:
        parts = json_path.split(os.sep)
        isolated_backbone = parts[-3]
        with open(json_path, 'r') as f:
            try: data = json.load(f)
            except json.JSONDecodeError: continue
        score, phase = data.get("best_mIoU", 0.0), data.get("phase", "unknown")
        t_loss, v_loss = data.get("final_train_loss", "N/A"), data.get("final_val_loss", "N/A")
        if phase in ["hpo", "export"]: continue
        
        match = ablation_pattern.match(isolated_backbone)
        if match:
            backbone, trial_base_name, seed = match.groups()
            if trial_base_name not in ablation_trials: ablation_trials[trial_base_name] = {}
            if score >= ablation_trials[trial_base_name].get(seed, {}).get("mIoU", 0.0):
                ablation_trials[trial_base_name][seed] = {"mIoU": score, "train_loss": t_loss, "val_loss": v_loss}
        else:
            if isolated_backbone not in primary_baselines: primary_baselines[isolated_backbone] = {}
            if score >= primary_baselines[isolated_backbone].get(phase, {}).get("mIoU", 0.0):
                primary_baselines[isolated_backbone][phase] = {"mIoU": score, "train_loss": t_loss, "val_loss": v_loss}

    analysis_payload = {"metadata": {"model": args.model, "dataset": args.dataset}, "primary_baselines": primary_baselines, "ablation_statistics": {}}
    control_key = "Control_Optimal"
    control_scores = [metrics["mIoU"] for seed, metrics in ablation_trials[control_key].items()] if control_key in ablation_trials else []

    for trial_name, seed_data in ablation_trials.items():
        scores = [metrics["mIoU"] for seed, metrics in seed_data.items()]
        stat_entry = {"seeds_evaluated": len(scores), "raw_scores": scores, "mean_mIoU": round(np.mean(scores), 4), "std_dev": round(np.std(scores), 4)}
        if trial_name == control_key:
            stat_entry["p_value_vs_control"] = stat_entry["is_statistically_significant"] = "Reference"
        elif len(control_scores) > 1 and len(scores) > 1:
            _, p_value = stats.ttest_ind(control_scores, scores, equal_var=False)
            stat_entry["p_value_vs_control"], stat_entry["is_statistically_significant"] = round(p_value, 4), bool(p_value < 0.05)
        else:
            stat_entry["p_value_vs_control"], stat_entry["is_statistically_significant"] = "Insufficient Data", False
        analysis_payload["ablation_statistics"][trial_name] = stat_entry

    with open(os.path.join(base_dir, "analysis.json"), 'w') as f: json.dump(analysis_payload, f, indent=4)
    vis_dir = os.path.join(base_dir, "analysis_plots")
    generate_metric_visualizations(primary_baselines, analysis_payload["ablation_statistics"], vis_dir)

    print("\n" + "="*85)
    print(f"📊 CVPR STATISTICAL SUMMARY: {args.model} | {args.dataset}")
    print("="*85 + "\n[ PRIMARY FLAGSHIP TRAJECTORIES ]\n" + f"{'Backbone':<15} | {'Phase':<12} | {'Max mIoU':<10} | {'Train Loss':<12} | {'Val Loss':<12}\n" + "-" * 70)
    for bb in sorted(primary_baselines.keys()):
        for phase in ["baseline", "hero", "microtune"]:
            if phase in primary_baselines[bb]:
                data = primary_baselines[bb][phase]
                print(f"{bb:<15} | {phase.capitalize():<12} | {data['mIoU']:.4f}     | {f'{data.get("train_loss")}'[:8]:<12} | {f'{data.get("val_loss")}'[:8]:<12}")
        print("-" * 70)
        
    print("\n[ ABLATION MATRIX STATISTICS (N={} Seeds) ]\n".format(len(control_scores)) + f"{'Ablation Variant':<30} | {'Mean ± Std':<15} | {'p-value':<12} | {'Sig (<0.05)'}\n" + "-" * 80)
    for trial, stats_data in analysis_payload["ablation_statistics"].items():
        mu_sigma = f"{stats_data['mean_mIoU']:.4f} ± {stats_data['std_dev']:.4f}"
        p_val, sig = stats_data.get("p_value_vs_control"), str(stats_data.get("is_statistically_significant"))
        print(f"{trial:<30} | {mu_sigma:<15} | {'---':<12} | Reference Baseline" if trial == control_key else f"{trial:<30} | {mu_sigma:<15} | {str(p_val):<12} | {sig}")

    # ---------------------------------------------------------
    # STAGE 2: LATENT SPACE EMBEDDING RENDERER
    # ---------------------------------------------------------
    if args.skip_latents:
        print("\n[*] Skipping heavy Latent Embedding renderings as requested.")
        return

    print("\n" + "="*85)
    print("🌌 INITIALIZING MANIFOLD PROJECTION ENGINE (UMAP / t-SNE)")
    print("="*85)

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    splits_root = os.path.join("data", "splits")
    try:
        with open(os.path.join(splits_root, args.dataset, "classes.txt"), "r") as f:
            class_names = [line.strip() for line in f if line.strip()]
            NUM_CLASSES = len(class_names)
    except FileNotFoundError:
        print(f"[-] WARNING: classes.txt not found. Cannot perform latent rendering for {args.dataset}")
        return

    eval_dataset = DownstreamSegmentationDataset(dataset_name=args.dataset, split="eval", splits_root=splits_root, image_size=(480, 640))
    eval_loader = DataLoader(eval_dataset, batch_size=4, shuffle=True, num_workers=4)

    ModelClass = getattr(models, args.model, None)
    if ModelClass is None:
        print(f"[-] CRITICAL: Architecture class '{args.model}' not found in models.py. Cannot extract latents.")
        return

    weight_files = glob.glob(os.path.join(base_dir, "*", "*", "best_model.pt"))
    if not weight_files:
        print("[-] No converged 'best_model.pt' artifacts found. Pipeline may still be training.")
        return

    for weights_path in weight_files:
        run_name = os.path.basename(os.path.dirname(weights_path))
        match = re.match(r"^(mit_b\d+)", os.path.basename(os.path.dirname(os.path.dirname(weights_path))))
        backbone = match.group(1) if match else "mit_b1"

        print(f"[*] Extracting latents for: {run_name} (Backbone: {backbone})...")
        
        try:
            model = ModelClass(num_classes=NUM_CLASSES, backbone_name=backbone, use_lora=False).to(DEVICE)
            model.load_state_dict(torch.load(weights_path, map_location=DEVICE), strict=False)
            
            modality_features, modality_labels = extract_modality_embeddings(model, eval_loader, DEVICE, args.samples)
            semantic_features, semantic_labels = extract_semantic_embeddings(model, eval_loader, DEVICE, args.samples)
            
            sns.set_theme(style="white", context="paper")
            # Increased figure width to comfortably accommodate the external legends
            fig, axes = plt.subplots(2, 2, figsize=(22, 16))
            
            compute_and_plot_latents(modality_features, modality_labels, "Modality Alignment (Stem Output)", axes[0, 0], axes[0, 1], "modality")
            compute_and_plot_latents(semantic_features, semantic_labels, "Semantic Separation (c4 Deep Features)", axes[1, 0], axes[1, 1], "semantic", class_names)
            
            plt.tight_layout(pad=3.0)
            save_path = os.path.join(vis_dir, f"manifold_{run_name}.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"    -> Render saved to: {save_path}")
        except Exception as e:
            print(f"    -> [!] Extraction failed for {run_name}: {e}")

    print("\n[+] CVPR Evaluation Suite Completed Successfully.")

if __name__ == "__main__":
    main()