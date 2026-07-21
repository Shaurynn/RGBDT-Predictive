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
from statsmodels.stats.multitest import multipletests
from sklearn.manifold import TSNE
import umap.umap_ as umap
from torch.utils.data import DataLoader

from dataset_jepa import DownstreamSegmentationDataset
import models

# Suppress UMAP multi-threading warnings to maintain clean terminal output
warnings.filterwarnings("ignore", message=".*n_jobs value 1 overridden.*")

def parse_args():
    parser = argparse.ArgumentParser(description="TMLPN Unified CVPR Evaluation & Latent Visualization Engine")
    parser.add_argument("--model", type=str, default="TMLPN_Downstream_v3", help="Target architecture namespace (e.g., TMLPN_Downstream_v3)")
    parser.add_argument("--dataset", type=str, default="MM5", help="Target dataset name")
    parser.add_argument("--samples", type=int, default=8000, help="Max spatial tokens to sample per latent projection")
    parser.add_argument("--skip_latents", action="store_true", help="Flag to bypass the heavy UMAP/t-SNE rendering")
    return parser.parse_args()

def configure_dark_mode():
    """Injects a high-contrast dark aesthetic strictly for spatial token visualization."""
    plt.style.use('dark_background')
    sns.set_theme(
        style="dark", 
        context="paper", 
        rc={
            "axes.facecolor": "#0d1117",       
            "figure.facecolor": "#0d1117", 
            "grid.color": "#30363d",           
            "axes.edgecolor": "#30363d",
            "text.color": "#c9d1d9",           
            "xtick.color": "#c9d1d9",
            "ytick.color": "#c9d1d9"
        }
    )

# ====================================================================================
# --- PART 1: METRIC AGGREGATION & STATISTICAL PLOTTING ---
# ====================================================================================

def generate_metric_visualizations(primary_baselines, ablation_stats, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    # Enforce standard light-mode whitegrid for CVPR statistical plots
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)

    if primary_baselines:
        plt.figure(figsize=(9, 6))
        trajectory_records = []
        for backbone, phases in primary_baselines.items():
            for phase in ["baseline", "hero", "microtune"]:
                if phase in phases:
                    # Append all seed instances; Seaborn automatically plots Mean with Error Bands
                    for seed, metrics in phases[phase].items():
                        trajectory_records.append({"Backbone": backbone, "Phase": phase.capitalize(), "mIoU": metrics["mIoU"]})
        
        df_traj = pd.DataFrame(trajectory_records)
        if not df_traj.empty:
            ax = sns.lineplot(data=df_traj, x='Phase', y='mIoU', hue='Backbone', marker='o', linewidth=2.5, markersize=8, palette='viridis')
            ax.set_title("Architecture Progression Trajectory (Multi-Seed)", fontweight='bold')
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
        unique_int_labels = sorted(np.unique(labels).astype(int))
        display_labels = [class_names[int(lbl)] if int(lbl) < len(class_names) else f"Class {lbl}" for lbl in labels]
        hue_order = [class_names[i] if i < len(class_names) else f"Class {i}" for i in unique_int_labels]
        palette = sns.color_palette("nipy_spectral", len(unique_int_labels))
    else:
        display_labels = labels
        hue_order = ["RGB Manifold", "Depth+Thermal Manifold"]
        palette = {"RGB Manifold": "#58a6ff", "Depth+Thermal Manifold": "#ff7b72"}

    df = pd.DataFrame({'TSNE_1': tsne_results[:, 0], 'TSNE_2': tsne_results[:, 1], 'UMAP_1': umap_results[:, 0], 'UMAP_2': umap_results[:, 1], 'Label': display_labels})
    
    sns.scatterplot(data=df, x='TSNE_1', y='TSNE_2', hue='Label', hue_order=hue_order, palette=palette, s=15, alpha=0.85, ax=ax_tsne, edgecolor='none')
    ax_tsne.set_title(f"t-SNE: {title}", fontweight='bold', color='white')
    ax_tsne.set_xticks([]); ax_tsne.set_yticks([]) 
    
    sns.scatterplot(data=df, x='UMAP_1', y='UMAP_2', hue='Label', hue_order=hue_order, palette=palette, s=15, alpha=0.85, ax=ax_umap, edgecolor='none')
    ax_umap.set_title(f"UMAP: {title}", fontweight='bold', color='white')
    ax_umap.set_xticks([]); ax_umap.set_yticks([]) 
    
    if palette_type == "semantic":
        for ax in [ax_tsne, ax_umap]:
            legend = ax.legend(title="Structural Class", bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., fontsize='small')
            plt.setp(legend.get_title(), color='white')
            plt.setp(legend.get_texts(), color='white')
            legend.get_frame().set_facecolor('#161b22')
            legend.get_frame().set_edgecolor('#30363d')

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
            
            # --- Multi-Seed Grouping Logic ---
            if trial_base_name == "baseline":
                if backbone not in primary_baselines: primary_baselines[backbone] = {}
                if phase not in primary_baselines[backbone]: primary_baselines[backbone][phase] = {}
                primary_baselines[backbone][phase][seed] = {"mIoU": score, "train_loss": t_loss, "val_loss": v_loss}
            else:
                if trial_base_name not in ablation_trials: ablation_trials[trial_base_name] = {}
                ablation_trials[trial_base_name][seed] = {"mIoU": score, "train_loss": t_loss, "val_loss": v_loss}
        else:
            # Fallback for old single-run baselines
            if isolated_backbone not in primary_baselines: primary_baselines[isolated_backbone] = {}
            if phase not in primary_baselines[isolated_backbone]: primary_baselines[isolated_backbone][phase] = {}
            primary_baselines[isolated_backbone][phase]["42"] = {"mIoU": score, "train_loss": t_loss, "val_loss": v_loss}

    analysis_payload = {"metadata": {"model": args.model, "dataset": args.dataset}, "primary_baselines": {}, "ablation_statistics": {}}
    
    # Process Primary Baselines Stats for Payload
    for bb, phases in primary_baselines.items():
        analysis_payload["primary_baselines"][bb] = {}
        for phase, seed_data in phases.items():
            scores = [m["mIoU"] for m in seed_data.values()]
            analysis_payload["primary_baselines"][bb][phase] = {
                "mean_mIoU": round(np.mean(scores), 4),
                "std_dev": round(np.std(scores), 4),
                "raw_scores": scores
            }

    control_key = "Control_Optimal"
    control_scores = [metrics["mIoU"] for seed, metrics in ablation_trials[control_key].items()] if control_key in ablation_trials else []

    # Prepare structures for Benjamini-Hochberg Correction
    raw_p_values = []
    trial_order = []

    for trial_name, seed_data in ablation_trials.items():
        scores = [metrics["mIoU"] for seed, metrics in seed_data.items()]
        stat_entry = {"seeds_evaluated": len(scores), "raw_scores": scores, "mean_mIoU": round(np.mean(scores), 4), "std_dev": round(np.std(scores), 4)}
        
        if trial_name == control_key:
            stat_entry["p_value_vs_control"] = "Reference"
            stat_entry["is_statistically_significant"] = "Reference"
        elif len(control_scores) > 1 and len(scores) > 1:
            _, p_value = stats.ttest_ind(control_scores, scores, equal_var=False)
            raw_p_values.append(p_value)
            trial_order.append(trial_name)
        else:
            stat_entry["p_value_vs_control"] = "Insufficient Data"
            stat_entry["is_statistically_significant"] = False
            
        analysis_payload["ablation_statistics"][trial_name] = stat_entry

    # Apply FDR (Benjamini-Hochberg) Correction
    if raw_p_values:
        reject, pvals_corrected, _, _ = multipletests(raw_p_values, alpha=0.05, method='fdr_bh')
        for i, trial_name in enumerate(trial_order):
            analysis_payload["ablation_statistics"][trial_name]["p_value_vs_control"] = round(pvals_corrected[i], 4)
            analysis_payload["ablation_statistics"][trial_name]["is_statistically_significant"] = bool(reject[i])

    with open(os.path.join(base_dir, "analysis.json"), 'w') as f: json.dump(analysis_payload, f, indent=4)
    vis_dir = os.path.join(base_dir, "analysis_plots")
    generate_metric_visualizations(primary_baselines, analysis_payload["ablation_statistics"], vis_dir)

    print("\n" + "="*85)
    print(f"📊 CVPR STATISTICAL SUMMARY: {args.model} | {args.dataset}")
    print("="*85 + "\n[ PRIMARY FLAGSHIP TRAJECTORIES ]\n" + f"{'Backbone':<15} | {'Phase':<12} | {'Mean ± Std':<15} | {'Train Loss':<12} | {'Val Loss':<12}\n" + "-" * 75)
    for bb in sorted(primary_baselines.keys()):
        for phase in ["baseline", "hero", "microtune"]:
            if phase in primary_baselines[bb]:
                data = analysis_payload["primary_baselines"][bb][phase]
                mu_sigma = f"{data['mean_mIoU']:.4f} ± {data['std_dev']:.4f}"
                
                # Average losses across seeds for clean reporting
                raw_t_loss = [m["train_loss"] for m in primary_baselines[bb][phase].values() if isinstance(m["train_loss"], (int, float))]
                raw_v_loss = [m["val_loss"] for m in primary_baselines[bb][phase].values() if isinstance(m["val_loss"], (int, float))]
                
                t_loss_str = f"{np.mean(raw_t_loss):.4f}" if raw_t_loss else "N/A"
                v_loss_str = f"{np.mean(raw_v_loss):.4f}" if raw_v_loss else "N/A"
                
                print(f"{bb:<15} | {phase.capitalize():<12} | {mu_sigma:<15} | {t_loss_str:<12} | {v_loss_str:<12}")
        print("-" * 75)
        
    print("\n[ ABLATION MATRIX STATISTICS (N={} Seeds) ]\n".format(len(control_scores)) + f"{'Ablation Variant':<30} | {'Mean ± Std':<15} | {'p-val (FDR)':<12} | {'Sig (<0.05)'}\n" + "-" * 80)
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
    
    # Inject Dark Mode directly before manifold plots
    configure_dark_mode()

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    splits_root = os.path.join("data", "splits")
    try:
        with open(os.path.join(splits_root, args.dataset, "classes.txt"), "r") as f:
            class_names = [re.sub(r'\\s*', '', line).strip() for line in f if line.strip()]
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

    runs_to_process = []
    for wf in weight_files:
        run_name = os.path.basename(os.path.dirname(wf))
        parent_backbone_folder = os.path.basename(os.path.dirname(os.path.dirname(wf)))
        
        # Filter: Avoid rendering identical latents for all 5 seeds. Restrict purely to Seed 42.
        if "seed" in parent_backbone_folder and "seed42" not in parent_backbone_folder:
            continue
            
        match = re.match(r"^(mit_b\d+)(?:_(.*))?$", parent_backbone_folder)
        backbone = match.group(1) if match else "mit_b1"
        trial_name = match.group(2) if (match and match.group(2)) else "baseline"
        
        p1_weights = os.path.join("weights", args.model, args.dataset, trial_name, f"jepa_context_encoder_{backbone}.pt")
        runs_to_process.append((p1_weights, wf, backbone, run_name))

    for p1_weights, p2_weights, backbone, run_name in runs_to_process:
        print(f"[*] Extracting latents for: {run_name} (Backbone: {backbone})...")
        
        has_p1 = p1_weights and os.path.exists(p1_weights)
        has_p2 = p2_weights and os.path.exists(p2_weights)
        
        cols = 4 if (has_p1 and has_p2) else 2
        fig, axes = plt.subplots(2, cols, figsize=(8*cols, 16), facecolor="#0d1117")
        if cols == 2: axes = axes.reshape(2, 2)
        
        try:
            if has_p1:
                print("    -> Extracting Phase 1 (Foundation) manifolds...")
                model_p1 = ModelClass(num_classes=NUM_CLASSES, backbone_name=backbone, use_lora=False).to(DEVICE)
                
                raw_checkpoint = torch.load(p1_weights, map_location=DEVICE)
                ce_weights = raw_checkpoint.get('context_encoder_state_dict', raw_checkpoint)
                model_p1.load_state_dict({f"context_encoder.{k}": v for k, v in ce_weights.items()}, strict=False)
                
                m_f1, m_l1 = extract_modality_embeddings(model_p1, eval_loader, DEVICE, args.samples)
                s_f1, s_l1 = extract_semantic_embeddings(model_p1, eval_loader, DEVICE, args.samples)
                
                col_offset = 0
                compute_and_plot_latents(m_f1, m_l1, "Phase 1: Modality Alignment", axes[0, col_offset], axes[0, col_offset+1], "modality", class_names)
                compute_and_plot_latents(s_f1, s_l1, "Phase 1: Semantic Foundation", axes[1, col_offset], axes[1, col_offset+1], "semantic", class_names)
                
            if has_p2:
                print("    -> Extracting Phase 2 (Fine-Tuned) manifolds...")
                model_p2 = ModelClass(num_classes=NUM_CLASSES, backbone_name=backbone, use_lora=False).to(DEVICE)
                model_p2.load_state_dict(torch.load(p2_weights, map_location=DEVICE), strict=False)
                
                m_f2, m_l2 = extract_modality_embeddings(model_p2, eval_loader, DEVICE, args.samples)
                s_f2, s_l2 = extract_semantic_embeddings(model_p2, eval_loader, DEVICE, args.samples)
                
                col_offset = 2 if (has_p1 and has_p2) else 0
                compute_and_plot_latents(m_f2, m_l2, "Phase 2: Modality Alignment", axes[0, col_offset], axes[0, col_offset+1], "modality", class_names)
                compute_and_plot_latents(s_f2, s_l2, "Phase 2: Semantic Separation", axes[1, col_offset], axes[1, col_offset+1], "semantic", class_names)
                
            plt.tight_layout(pad=3.0)
            save_path = os.path.join(vis_dir, f"manifold_progression_{run_name}.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor="#0d1117")
            plt.close()
            print(f"    -> Render saved to: {save_path}")
            
        except Exception as e:
            print(f"    -> [!] Extraction failed for {run_name}: {e}")

    print("\n[+] CVPR Evaluation Suite Completed Successfully.")

if __name__ == "__main__":
    main()