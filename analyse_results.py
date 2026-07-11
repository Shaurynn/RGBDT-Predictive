import os
import re
import glob
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def parse_args():
    parser = argparse.ArgumentParser(description="MLOps CVPR Results Extraction & Statistical Analysis")
    parser.add_argument("--model", type=str, default="TMLPN_Downstream", help="The name of the architecture")
    parser.add_argument("--dataset", type=str, default="MM5", help="The dataset name")
    return parser.parse_args()

def generate_visualizations(primary_baselines, ablation_stats, output_dir):
    """Generates CVPR-ready visualizations for Phase Trajectories and Ablation Statistics."""
    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)

    # --- Plot 1: Phase Progression Trajectory (Line Plot) ---
    if primary_baselines:
        plt.figure(figsize=(9, 6))
        
        trajectory_records = []
        phase_order = ["baseline", "hero", "microtune"]
        
        for backbone, phases in primary_baselines.items():
            for phase in phase_order:
                if phase in phases:
                    trajectory_records.append({
                        "Backbone": backbone,
                        "Phase": phase.capitalize(),
                        "mIoU": phases[phase]["mIoU"]
                    })
                    
        df_traj = pd.DataFrame(trajectory_records)
        
        if not df_traj.empty:
            ax = sns.lineplot(data=df_traj, x='Phase', y='mIoU', hue='Backbone', 
                              marker='o', linewidth=2.5, markersize=8, palette='viridis')
            ax.set_title("Architecture Progression Trajectory", fontweight='bold')
            ax.set_ylabel("Mean Intersection over Union (mIoU)")
            ax.set_xlabel("Training Phase")
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "phase_progression_trajectory.png"), dpi=300)
            plt.close()
            print(f"[*] Generated: {os.path.join(output_dir, 'phase_progression_trajectory.png')}")

    # --- Plot 2: Ablation Matrix (Mean ± Std Dev) ---
    if ablation_stats:
        plt.figure(figsize=(10, 6))
        
        ablation_records = []
        for trial, stats_data in ablation_stats.items():
            ablation_records.append({
                "Variant": trial.replace("Ablation_", "").replace("_", " "),
                "Mean mIoU": stats_data["mean_mIoU"],
                "Std Dev": stats_data["std_dev"]
            })
            
        df_abl = pd.DataFrame(ablation_records)
        
        ax = sns.barplot(x='Variant', y='Mean mIoU', data=df_abl, palette='mako')
        
        # Inject precise error bars based on the calculated seeds
        ax.errorbar(x=range(len(df_abl)), y=df_abl['Mean mIoU'], yerr=df_abl['Std Dev'], 
                    fmt='none', c='black', capsize=5, elinewidth=1.5)

        ax.set_title("Ablation Study Results (N Seeds)", fontweight='bold')
        ax.set_ylim(0, max(df_abl['Mean mIoU']) * 1.15)
        ax.set_ylabel("Mean mIoU")
        plt.xticks(rotation=15)
        
        for i, row in df_abl.iterrows():
            ax.annotate(f"{row['Mean mIoU']:.4f}\n±{row['Std Dev']:.4f}", 
                        (i, row['Mean mIoU'] + row['Std Dev']),
                        ha='center', va='bottom', fontsize=10, xytext=(0, 8), textcoords='offset points')
            
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "ablation_statistics.png"), dpi=300)
        plt.close()
        print(f"[*] Generated: {os.path.join(output_dir, 'ablation_statistics.png')}")


def analyse_results(model_name, dataset_name):
    print(f"\n[*] Booting MLOps Analysis Engine for {model_name} | {dataset_name}...")
    
    base_dir = os.path.join("results", model_name, dataset_name)
    
    if not os.path.exists(base_dir):
        print(f"[-] CRITICAL: Directory not found at {base_dir}")
        return

    primary_baselines = {}
    ablation_trials = {}
    ablation_pattern = re.compile(r"^(mit_b\d+)_(.*)_seed(\d+)$")

    search_path = os.path.join(base_dir, "*", "*", "results.json")
    json_files = glob.glob(search_path)
    
    if not json_files:
        print("[-] WARNING: No results.json files found. Has the pipeline finished running?")
        return

    for json_path in json_files:
        parts = json_path.split(os.sep)
        isolated_backbone = parts[-3]
        
        with open(json_path, 'r') as f:
            try:
                data = json.load(f)
                score = data.get("best_mIoU", 0.0)
                phase = data.get("phase", "unknown")
                t_loss = data.get("final_train_loss", "N/A")
                v_loss = data.get("final_val_loss", "N/A")
            except json.JSONDecodeError:
                continue

        # Strictly ignore HPO and Export phases for metric tabulation
        if phase in ["hpo", "export"]:
            continue

        match = ablation_pattern.match(isolated_backbone)
        
        if match:
            # Ablations only execute the baseline phase
            backbone, trial_base_name, seed = match.groups()
            if trial_base_name not in ablation_trials:
                ablation_trials[trial_base_name] = {}
                
            current_max = ablation_trials[trial_base_name].get(seed, {}).get("mIoU", 0.0)
            if score >= current_max:
                ablation_trials[trial_base_name][seed] = {"mIoU": score, "train_loss": t_loss, "val_loss": v_loss}
        else:
            # Primary Flagships (Tracked across multiple phases)
            backbone = isolated_backbone
            if backbone not in primary_baselines:
                primary_baselines[backbone] = {}
                
            current_max = primary_baselines[backbone].get(phase, {}).get("mIoU", 0.0)
            if score >= current_max:
                primary_baselines[backbone][phase] = {
                    "mIoU": score, "train_loss": t_loss, "val_loss": v_loss
                }

    # ====================================================================================
    # --- STATISTICAL ENGINE ---
    # ====================================================================================
    analysis_payload = {"metadata": {"model": model_name, "dataset": dataset_name}, "primary_baselines": primary_baselines, "ablation_statistics": {}}

    control_key = "Control_Optimal"
    control_scores = []
    if control_key in ablation_trials:
        control_scores = [metrics["mIoU"] for seed, metrics in ablation_trials[control_key].items()]

    for trial_name, seed_data in ablation_trials.items():
        scores = [metrics["mIoU"] for seed, metrics in seed_data.items()]
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        
        stat_entry = {"seeds_evaluated": len(scores), "raw_scores": scores, "mean_mIoU": round(mean_score, 4), "std_dev": round(std_score, 4)}

        # Determine statistical significance
        if trial_name == control_key:
            stat_entry["p_value_vs_control"] = "Reference"
            stat_entry["is_statistically_significant"] = "Reference"
        elif len(control_scores) > 1 and len(scores) > 1:
            t_stat, p_value = stats.ttest_ind(control_scores, scores, equal_var=False)
            stat_entry["p_value_vs_control"] = round(p_value, 4)
            stat_entry["is_statistically_significant"] = bool(p_value < 0.05)
        else:
            stat_entry["p_value_vs_control"] = "Insufficient Data"
            stat_entry["is_statistically_significant"] = False
            
        analysis_payload["ablation_statistics"][trial_name] = stat_entry

    output_path = os.path.join(base_dir, "analysis.json")
    with open(output_path, 'w') as f:
        json.dump(analysis_payload, f, indent=4)

    vis_dir = os.path.join(base_dir, "analysis_plots")
    generate_visualizations(primary_baselines, analysis_payload["ablation_statistics"], vis_dir)

    # ====================================================================================
    # --- TERMINAL TABULATION ---
    # ====================================================================================
    print("\n" + "="*85)
    print(f"📊 MLOPS PIPELINE RESULTS: {model_name} | {dataset_name}")
    print("="*85)
    
    print("\n[ PRIMARY FLAGSHIP TRAJECTORIES ]")
    print(f"{'Backbone':<15} | {'Phase':<12} | {'Max mIoU':<10} | {'Train Loss':<12} | {'Val Loss':<12}")
    print("-" * 70)
    
    phase_order = ["baseline", "hero", "microtune"]
    for bb in sorted(primary_baselines.keys()):
        for phase in phase_order:
            if phase in primary_baselines[bb]:
                data = primary_baselines[bb][phase]
                t_loss = f"{data['train_loss']:.4f}" if isinstance(data['train_loss'], float) else str(data['train_loss'])
                v_loss = f"{data['val_loss']:.4f}" if isinstance(data['val_loss'], float) else str(data['val_loss'])
                print(f"{bb:<15} | {phase.capitalize():<12} | {data['mIoU']:.4f}     | {t_loss:<12} | {v_loss:<12}")
        print("-" * 70)

    print("\n[ ABLATION MATRIX STATISTICS (N={} Seeds) ]".format(len(control_scores) if control_scores else "0"))
    print(f"{'Ablation Variant':<30} | {'Mean ± Std':<15} | {'p-value':<12} | {'Sig (<0.05)'}")
    print("-" * 80)
    
    for trial, stats_data in analysis_payload["ablation_statistics"].items():
        mu_sigma = f"{stats_data['mean_mIoU']:.4f} ± {stats_data['std_dev']:.4f}"
        p_val = stats_data.get("p_value_vs_control")
        sig = str(stats_data.get("is_statistically_significant"))
        
        # Clean formatting for the control reference
        if trial == control_key:
            print(f"{trial:<30} | {mu_sigma:<15} | {'---':<12} | Reference Baseline")
        else:
            print(f"{trial:<30} | {mu_sigma:<15} | {str(p_val):<12} | {sig}")

    print("\n" + "="*85)
    print(f"[+] Statistical analysis exported to: {output_path}")
    print(f"[+] Rendered image artifacts saved to:  {vis_dir}")

if __name__ == "__main__":
    args = parse_args()
    analyse_results(args.model, args.dataset)