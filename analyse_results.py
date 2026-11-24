import os
import re
import glob
import json
import math
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

warnings.filterwarnings("ignore", message=".*n_jobs value 1 overridden.*")


def parse_args():
    parser = argparse.ArgumentParser(description="TMLPN Unified CVPR Evaluation & Latent Visualization Engine")
    parser.add_argument("--model", type=str, default="TMLPN_Downstream_v3", help="Target architecture namespace")
    parser.add_argument("--dataset", type=str, default="MM5", help="Target dataset name")
    parser.add_argument("--samples", type=int, default=8000, help="Max spatial tokens to sample per latent projection")
    parser.add_argument("--skip_latents", action="store_true", help="Flag to bypass heavy UMAP/t-SNE rendering")
    return parser.parse_args()


def sample_std(values: list) -> float:
    """Unbiased sample standard deviation (ddof=1), returning 0.0 for single-seed entries."""
    if len(values) <= 1:
        return 0.0
    return float(np.std(values, ddof=1))


def configure_dark_mode() -> None:
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

def generate_metric_visualizations(primary_baselines: dict, ablation_stats: dict, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)

    if primary_baselines:
        plt.figure(figsize=(9, 6))
        trajectory_records = []
        for backbone, phases in primary_baselines.items():
            for phase in ["baseline", "hero", "microtune"]:
                if phase in phases:
                    for _, metrics in phases[phase].items():
                        trajectory_records.append({
                            "Backbone": backbone,
                            "Phase": phase.capitalize(),
                            "mIoU": metrics["mIoU"]
                        })

        df_traj = pd.DataFrame(trajectory_records)
        if not df_traj.empty:
            ax = sns.lineplot(
                data=df_traj, x='Phase', y='mIoU', hue='Backbone',
                marker='o', linewidth=2.5, markersize=8, palette='viridis'
            )
            ax.set_title("Architecture Progression Trajectory (Multi-Seed)", fontweight='bold')
            ax.set_ylabel("Mean Intersection over Union (mIoU)")
            ax.set_xlabel("Training Phase")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "phase_progression_trajectory.png"), dpi=300)
            plt.close()

    if ablation_stats:
        plt.figure(figsize=(12, 6))
        ablation_records = [
            {
                "Variant": trial.replace("Ablation_", "").replace("_", " "),
                "Mean mIoU": stats_data["mean_mIoU"],
                "Std Dev": stats_data["std_dev"]
            }
            for trial, stats_data in ablation_stats.items()
        ]
        df_abl = pd.DataFrame(ablation_records)
        if not df_abl.empty:
            ax = sns.barplot(x='Variant', y='Mean mIoU', hue='Variant', data=df_abl, palette='mako', legend=False)
            ax.errorbar(
                x=range(len(df_abl)), y=df_abl['Mean mIoU'], yerr=df_abl['Std Dev'],
                fmt='none', c='black', capsize=5, elinewidth=1.5
            )
            ax.set_title("Ablation Study Results (Multi-Seed, Unbiased Std)", fontweight='bold')
            ymax = max(0.1, float((df_abl['Mean mIoU'] + df_abl['Std Dev']).max()) * 1.15)
            ax.set_ylim(0, ymax)
            ax.set_ylabel("Mean mIoU (Blind Test TTA)")
            plt.xticks(rotation=25, ha='right')
            for i, row in df_abl.iterrows():
                ax.annotate(
                    f"{row['Mean mIoU']:.4f}\n±{row['Std Dev']:.4f}",
                    (i, row['Mean mIoU'] + row['Std Dev']),
                    ha='center', va='bottom', fontsize=9, xytext=(0, 8), textcoords='offset points'
                )
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "ablation_statistics.png"), dpi=300)
            plt.close()


# ====================================================================================
# --- PART 2: UNIFORM FULL-TEST LATENT SPACE EXTRACTION & PROJECTION ---
# ====================================================================================

def extract_modality_embeddings(model, dataloader, device, max_samples: int, seed: int = 42):
    """
    Samples RGB and Depth+Thermal stem tokens uniformly across ALL batches of the test split
    (avoiding first-batch sampling bias) using the model's exact `forward_split` stem math.
    """
    model.eval()
    rng = np.random.default_rng(seed)
    target_per_modality = max_samples // 2
    num_batches = max(1, len(dataloader))
    per_batch_quota = max(1, math.ceil(target_per_modality / num_batches))

    rgb_chunks, dt_chunks = [], []
    with torch.no_grad():
        for batch in dataloader:
            x_full = batch['x_full'].to(device)
            stem = model.context_encoder.patch_embed1.proj

            if hasattr(stem, 'forward_split'):
                rgb_aligned, dt_aligned = stem.forward_split(x_full)
            elif hasattr(stem, 'rgb_proj'):
                rgb_aligned = stem.rgb_proj(x_full[:, :3, :, :])
                dt_raw = stem.depth_therm_proj(x_full[:, 3:, :, :])
                dt_aligned = stem.dt_alignment(dt_raw)
            else:
                unified = stem(x_full)
                rgb_aligned, dt_aligned = unified, unified

            rgb_flat = rgb_aligned.permute(0, 2, 3, 1).reshape(-1, rgb_aligned.shape[1]).cpu().numpy()
            dt_flat = dt_aligned.permute(0, 2, 3, 1).reshape(-1, dt_aligned.shape[1]).cpu().numpy()

            k = min(per_batch_quota, len(rgb_flat))
            rgb_chunks.append(rgb_flat[rng.choice(len(rgb_flat), k, replace=False)])
            dt_chunks.append(dt_flat[rng.choice(len(dt_flat), k, replace=False)])

    rgb_np = np.concatenate(rgb_chunks, axis=0)
    dt_np = np.concatenate(dt_chunks, axis=0)

    sample_size = min(len(rgb_np), len(dt_np), target_per_modality)
    rgb_idx = rng.choice(len(rgb_np), sample_size, replace=False)
    dt_idx = rng.choice(len(dt_np), sample_size, replace=False)

    features = np.vstack((rgb_np[rgb_idx], dt_np[dt_idx]))
    labels = np.array(['RGB Manifold'] * sample_size + ['Depth+Thermal Manifold'] * sample_size)
    return features, labels


def extract_semantic_embeddings(model, dataloader, device, max_samples: int, seed: int = 42):
    """
    Samples Stage-4 (c4) semantic embeddings uniformly across ALL batches of the test split.
    """
    model.eval()
    rng = np.random.default_rng(seed)
    num_batches = max(1, len(dataloader))
    per_batch_quota = max(1, math.ceil(max_samples / num_batches))

    feat_chunks, label_chunks = [], []
    with torch.no_grad():
        for batch in dataloader:
            x_full = batch['x_full'].to(device)
            seg_mask = batch['seg_mask'].to(device)

            _, features = model(x_full, return_features=True)
            c4_latent = features[-1]

            mask_down = torch.nn.functional.interpolate(
                seg_mask.unsqueeze(1).float(),
                size=c4_latent.shape[2:],
                mode='nearest'
            ).squeeze(1).long()

            c4_flat = c4_latent.permute(0, 2, 3, 1).reshape(-1, c4_latent.shape[1])
            mask_flat = mask_down.reshape(-1)

            valid_idx = (mask_flat != 255)
            f_valid = c4_flat[valid_idx].cpu().numpy()
            l_valid = mask_flat[valid_idx].cpu().numpy()
            if len(f_valid) == 0:
                continue

            k = min(per_batch_quota, len(f_valid))
            chosen = rng.choice(len(f_valid), k, replace=False)
            feat_chunks.append(f_valid[chosen])
            label_chunks.append(l_valid[chosen])

    f_np = np.concatenate(feat_chunks, axis=0)
    l_np = np.concatenate(label_chunks, axis=0)
    if len(f_np) > max_samples:
        indices = rng.choice(len(f_np), max_samples, replace=False)
        return f_np[indices], l_np[indices]
    return f_np, l_np


def compute_and_plot_latents(features, labels, title, ax_tsne, ax_umap, palette_type, class_names=None):
    perplexity = min(45, max(5, (len(features) - 1) // 3))
    tsne_results = TSNE(n_components=2, perplexity=perplexity, random_state=42, max_iter=1000).fit_transform(features)
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

    df = pd.DataFrame({
        'TSNE_1': tsne_results[:, 0],
        'TSNE_2': tsne_results[:, 1],
        'UMAP_1': umap_results[:, 0],
        'UMAP_2': umap_results[:, 1],
        'Label': display_labels
    })

    sns.scatterplot(
        data=df, x='TSNE_1', y='TSNE_2', hue='Label', hue_order=hue_order,
        palette=palette, s=15, alpha=0.85, ax=ax_tsne, edgecolor='none'
    )
    ax_tsne.set_title(f"t-SNE: {title}", fontweight='bold', color='white')
    ax_tsne.set_xticks([])
    ax_tsne.set_yticks([])

    sns.scatterplot(
        data=df, x='UMAP_1', y='UMAP_2', hue='Label', hue_order=hue_order,
        palette=palette, s=15, alpha=0.85, ax=ax_umap, edgecolor='none'
    )
    ax_umap.set_title(f"UMAP: {title}", fontweight='bold', color='white')
    ax_umap.set_xticks([])
    ax_umap.set_yticks([])

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
    ablation_backbone_detected = None

    # Backbone-agnostic pattern supporting MiT (mit_b1..b5) and external baselines (e.g., resnet50)
    run_folder_pattern = re.compile(r"^(.+?)_(baseline|Control_Optimal|Ablation_.+|LoRA_.+)_seed(\d+)$")
    json_files = sorted(glob.glob(os.path.join(base_dir, "*", "*", "results.json")))

    for json_path in json_files:
        parts = json_path.split(os.sep)
        isolated_backbone = parts[-3]
        with open(json_path, 'r') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                continue

        phase = data.get("phase", "unknown").lower()
        if phase in ["hpo", "export"]:
            continue

        score = float(data.get("best_mIoU", 0.0))
        val_score = data.get("val_mIoU_checkpoint", None)
        t_loss = data.get("final_train_loss", "N/A")
        v_loss = data.get("final_val_loss", "N/A")
        record = {"mIoU": score, "val_mIoU": val_score, "train_loss": t_loss, "val_loss": v_loss}

        match = run_folder_pattern.match(isolated_backbone)
        if match:
            backbone, trial_base_name, seed = match.groups()
            if trial_base_name == "baseline":
                primary_baselines.setdefault(backbone, {}).setdefault(phase, {})[seed] = record
            elif phase == "microtune":
                ablation_backbone_detected = backbone
                ablation_trials.setdefault(trial_base_name, {})[seed] = record
        else:
            primary_baselines.setdefault(isolated_backbone, {}).setdefault(phase, {})["42"] = record

    # Fall back to the ablation backbone's baseline microtune seeds if Control_Optimal was not mirrored
    control_key = "Control_Optimal"
    if (
        control_key not in ablation_trials
        and ablation_backbone_detected in primary_baselines
        and "microtune" in primary_baselines[ablation_backbone_detected]
    ):
        ablation_trials[control_key] = primary_baselines[ablation_backbone_detected]["microtune"]

    analysis_payload = {
        "metadata": {"model": args.model, "dataset": args.dataset, "std_ddof": 1},
        "primary_baselines": {},
        "ablation_statistics": {}
    }

    for bb, phases in primary_baselines.items():
        analysis_payload["primary_baselines"][bb] = {}
        for phase, seed_data in phases.items():
            scores = [m["mIoU"] for m in seed_data.values()]
            val_scores = [m["val_mIoU"] for m in seed_data.values() if isinstance(m["val_mIoU"], (int, float))]
            raw_t_loss = [m["train_loss"] for m in seed_data.values() if isinstance(m["train_loss"], (int, float))]
            raw_v_loss = [m["val_loss"] for m in seed_data.values() if isinstance(m["val_loss"], (int, float))]
            analysis_payload["primary_baselines"][bb][phase] = {
                "mean_mIoU": round(float(np.mean(scores)), 4),
                "std_dev": round(sample_std(scores), 4),
                "mean_val_mIoU": round(float(np.mean(val_scores)), 4) if val_scores else None,
                "mean_train_loss": round(float(np.mean(raw_t_loss)), 4) if raw_t_loss else None,
                "mean_val_loss": round(float(np.mean(raw_v_loss)), 4) if raw_v_loss else None,
                "raw_scores": scores
            }

    control_seed_map = ablation_trials.get(control_key, {})
    control_scores = [m["mIoU"] for m in control_seed_map.values()]

    raw_p_values = []
    trial_order = []

    for trial_name, seed_data in ablation_trials.items():
        scores = [m["mIoU"] for m in seed_data.values()]
        raw_t_loss = [m["train_loss"] for m in seed_data.values() if isinstance(m["train_loss"], (int, float))]
        raw_v_loss = [m["val_loss"] for m in seed_data.values() if isinstance(m["val_loss"], (int, float))]
        stat_entry = {
            "seeds_evaluated": len(scores),
            "raw_scores": scores,
            "mean_mIoU": round(float(np.mean(scores)), 4),
            "std_dev": round(sample_std(scores), 4),
            "mean_train_loss": round(float(np.mean(raw_t_loss)), 4) if raw_t_loss else None,
            "mean_val_loss": round(float(np.mean(raw_v_loss)), 4) if raw_v_loss else None
        }

        if trial_name == control_key:
            stat_entry["test_type"] = "Reference"
            stat_entry["p_value_vs_control"] = "Reference"
            stat_entry["is_statistically_significant"] = "Reference"
        else:
            common_seeds = sorted(set(control_seed_map.keys()) & set(seed_data.keys()))
            if len(common_seeds) > 1:
                c_vals = [control_seed_map[s]["mIoU"] for s in common_seeds]
                t_vals = [seed_data[s]["mIoU"] for s in common_seeds]
                _, p_value = stats.ttest_rel(c_vals, t_vals)
                stat_entry["test_type"] = "paired_t"
            elif len(control_scores) > 1 and len(scores) > 1:
                _, p_value = stats.ttest_ind(control_scores, scores, equal_var=False)
                stat_entry["test_type"] = "welch_t"
            else:
                p_value = None
                stat_entry["test_type"] = "none"

            if p_value is not None:
                if math.isnan(p_value):
                    p_value = 1.0
                raw_p_values.append(float(p_value))
                trial_order.append(trial_name)
            else:
                stat_entry["p_value_vs_control"] = "Insufficient Data"
                stat_entry["is_statistically_significant"] = False

        analysis_payload["ablation_statistics"][trial_name] = stat_entry

    # Apply Benjamini-Hochberg False Discovery Rate (FDR) Correction
    if raw_p_values:
        reject, pvals_corrected, _, _ = multipletests(raw_p_values, alpha=0.05, method='fdr_bh')
        for i, trial_name in enumerate(trial_order):
            analysis_payload["ablation_statistics"][trial_name]["p_value_vs_control"] = round(float(pvals_corrected[i]), 4)
            analysis_payload["ablation_statistics"][trial_name]["is_statistically_significant"] = bool(reject[i])

    with open(os.path.join(base_dir, "analysis.json"), 'w') as f:
        json.dump(analysis_payload, f, indent=4)

    vis_dir = os.path.join(base_dir, "analysis_plots")
    generate_metric_visualizations(primary_baselines, analysis_payload["ablation_statistics"], vis_dir)

    print("\n" + "=" * 90)
    print(f"📊 CVPR STATISTICAL SUMMARY: {args.model} | {args.dataset} (Unbiased Sample Std, ddof=1)")
    print("=" * 90)
    print(f"{'Backbone':<15} | {'Phase':<12} | {'Mean ± Std':<17} | {'Val mIoU':<10} | {'Train Loss':<10} | {'Val Loss':<10}")
    print("-" * 90)
    for bb in sorted(primary_baselines.keys()):
        for phase in ["baseline", "hero", "microtune"]:
            if phase in primary_baselines[bb]:
                d = analysis_payload["primary_baselines"][bb][phase]
                mu_sigma = f"{d['mean_mIoU']:.4f} ± {d['std_dev']:.4f}"
                v_miou = f"{d['mean_val_mIoU']:.4f}" if d['mean_val_mIoU'] is not None else "N/A"
                t_loss = f"{d['mean_train_loss']:.4f}" if d['mean_train_loss'] is not None else "N/A"
                v_loss = f"{d['mean_val_loss']:.4f}" if d['mean_val_loss'] is not None else "N/A"
                print(f"{bb:<15} | {phase.capitalize():<12} | {mu_sigma:<17} | {v_miou:<10} | {t_loss:<10} | {v_loss:<10}")
        print("-" * 90)

    print(f"\n[ ABLATION MATRIX STATISTICS (N={len(control_scores)} Seeds, Paired-t + BH-FDR) ]")
    print(f"{'Ablation Variant':<30} | {'Mean ± Std':<17} | {'Test':<10} | {'p-val (FDR)':<12} | {'Sig (<0.05)'}")
    print("-" * 90)
    for trial, s_data in analysis_payload["ablation_statistics"].items():
        mu_sigma = f"{s_data['mean_mIoU']:.4f} ± {s_data['std_dev']:.4f}"
        if trial == control_key:
            print(f"{trial:<30} | {mu_sigma:<17} | {'Ref':<10} | {'---':<12} | Reference Baseline")
        else:
            print(
                f"{trial:<30} | {mu_sigma:<17} | {s_data.get('test_type', 'N/A'):<10} | "
                f"{str(s_data.get('p_value_vs_control')):<12} | {s_data.get('is_statistically_significant')}"
            )

    # ---------------------------------------------------------
    # STAGE 2: LATENT SPACE EMBEDDING RENDERER
    # ---------------------------------------------------------
    if args.skip_latents:
        print("\n[*] Skipping heavy Latent Embedding renderings as requested.")
        return

    print("\n" + "=" * 90)
    print("🌌 INITIALIZING MANIFOLD PROJECTION ENGINE (FULL-TEST UNIFORM UMAP / t-SNE)")
    print("=" * 90)

    configure_dark_mode()

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    splits_root = os.path.join("data", "splits")
    try:
        with open(os.path.join(splits_root, args.dataset, "classes.txt"), "r") as f:
            class_names = [line.strip() for line in f if line.strip()]
            NUM_CLASSES = len(class_names)
    except FileNotFoundError:
        print(f"[-] WARNING: classes.txt not found. Cannot perform latent rendering for {args.dataset}")
        return

    eval_dataset = DownstreamSegmentationDataset(
        dataset_name=args.dataset,
        split="test",
        splits_root=splits_root,
        image_size=(480, 640),
        enable_augmentation=False
    )
    eval_loader = DataLoader(eval_dataset, batch_size=4, shuffle=False, num_workers=4)

    ModelClass = getattr(models, args.model, None)
    if ModelClass is None:
        print(f"[-] CRITICAL: Architecture class '{args.model}' not found in models.py.")
        return

    weight_files = sorted(glob.glob(os.path.join(base_dir, "*", "*", "best_model.pt")))
    if not weight_files:
        print("[-] No converged 'best_model.pt' artifacts found. Pipeline may still be training.")
        return

    runs_to_process = []
    for wf in weight_files:
        run_name = os.path.basename(os.path.dirname(wf))
        parent_backbone_folder = os.path.basename(os.path.dirname(os.path.dirname(wf)))

        if "seed" in parent_backbone_folder and "seed42" not in parent_backbone_folder:
            continue

        match = run_folder_pattern.match(parent_backbone_folder)
        if match:
            backbone, trial_base, seed = match.groups()
            trial_name = f"{trial_base}_seed{seed}"
        else:
            backbone = parent_backbone_folder
            trial_name = "baseline"

        p1_weights = os.path.join("weights", args.model, args.dataset, trial_name, f"jepa_context_encoder_{backbone}.pt")
        if not os.path.exists(p1_weights):
            fallback_p1 = os.path.join("weights", args.model, args.dataset, "baseline_seed42", f"jepa_context_encoder_{backbone}.pt")
            if os.path.exists(fallback_p1):
                p1_weights = fallback_p1
        runs_to_process.append((p1_weights, wf, backbone, parent_backbone_folder, run_name))

    for p1_weights, p2_weights, backbone, folder_tag, run_name in runs_to_process:
        print(f"[*] Extracting latents for: {folder_tag}/{run_name} (Backbone: {backbone})...")

        has_p1 = p1_weights and os.path.exists(p1_weights)
        has_p2 = p2_weights and os.path.exists(p2_weights)

        cols = 4 if (has_p1 and has_p2) else 2
        fig, axes = plt.subplots(2, cols, figsize=(8 * cols, 16), facecolor="#0d1117")
        if cols == 2:
            axes = axes.reshape(2, 2)

        try:
            if has_p1:
                print("    -> Extracting Phase 1 (Foundation) manifolds...")
                raw_ckpt = torch.load(p1_weights, map_location=DEVICE)
                ce_weights = raw_ckpt.get('context_encoder_state_dict', raw_ckpt)
                prefixed_ce_weights = {f"context_encoder.{k}": v for k, v in ce_weights.items()}
                model_p1 = models.instantiate_model_from_checkpoint(
                    ModelClass, NUM_CLASSES, backbone, prefixed_ce_weights, DEVICE, strict=False
                )

                m_f1, m_l1 = extract_modality_embeddings(model_p1, eval_loader, DEVICE, args.samples)
                s_f1, s_l1 = extract_semantic_embeddings(model_p1, eval_loader, DEVICE, args.samples)

                compute_and_plot_latents(m_f1, m_l1, "Phase 1: Modality Alignment", axes[0, 0], axes[0, 1], "modality", class_names)
                compute_and_plot_latents(s_f1, s_l1, "Phase 1: Semantic Foundation", axes[1, 0], axes[1, 1], "semantic", class_names)

            if has_p2:
                print("    -> Extracting Phase 2 (Fine-Tuned) manifolds...")
                p2_state_dict = torch.load(p2_weights, map_location=DEVICE)
                model_p2 = models.instantiate_model_from_checkpoint(
                    ModelClass, NUM_CLASSES, backbone, p2_state_dict, DEVICE, strict=False
                )

                m_f2, m_l2 = extract_modality_embeddings(model_p2, eval_loader, DEVICE, args.samples)
                s_f2, s_l2 = extract_semantic_embeddings(model_p2, eval_loader, DEVICE, args.samples)

                col_offset = 2 if (has_p1 and has_p2) else 0
                compute_and_plot_latents(m_f2, m_l2, "Phase 2: Modality Alignment", axes[0, col_offset], axes[0, col_offset + 1], "modality", class_names)
                compute_and_plot_latents(s_f2, s_l2, "Phase 2: Semantic Separation", axes[1, col_offset], axes[1, col_offset + 1], "semantic", class_names)

            plt.tight_layout(pad=3.0)
            save_path = os.path.join(vis_dir, f"manifold_progression_{folder_tag}_{run_name}.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor="#0d1117")
            plt.close()
            print(f"    -> Render saved to: {save_path}")

        except Exception as e:
            print(f"    -> [!] Extraction failed for {folder_tag}/{run_name}: {e}")
            plt.close(fig)

    print("\n[+] CVPR Evaluation Suite Completed Successfully.")


if __name__ == "__main__":
    main()