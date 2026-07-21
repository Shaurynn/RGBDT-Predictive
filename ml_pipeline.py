import os
import glob
import yaml
import json
import shutil
import smtplib
import subprocess
import traceback
import argparse
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dotenv import load_dotenv
load_dotenv()

# ====================================================================================
# --- NOTIFICATION ENGINE ---
# ====================================================================================
def send_alert_email(subject, error_message):
    smtp_server = os.getenv("SMTP_SERVER", "smtp.gmail.com")
    smtp_port = int(os.getenv("SMTP_PORT", 587))
    sender_email = os.getenv("SMTP_SENDER")
    sender_password = os.getenv("SMTP_PASSWORD")
    recipient_email = os.getenv("ALERT_RECIPIENT")

    if not all([sender_email, sender_password, recipient_email]):
        print("[!] WARNING: SMTP credentials missing in environment. Cannot send email alert.")
        return

    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = recipient_email
    msg['Subject'] = f"[MLOps Pipeline Alert] {subject}"
    msg.attach(MIMEText(f"The automated training pipeline has encountered a fatal error.\n\n{error_message}", 'plain'))

    try:
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            server.send_message(msg)
        print(f"[+] Alert email successfully dispatched to {recipient_email}")
    except Exception as e:
        print(f"[-] CRITICAL: Failed to send email alert: {e}")

# ====================================================================================
# --- CONFIGURATION MANAGER ---
# ====================================================================================
def inject_configuration(yaml_path, backbone_name, ablation_state, trial_name="baseline", seed=42, model_name="TMLPN_Downstream_v3"):
    with open(yaml_path, 'r') as f: config = yaml.safe_load(f)

    if 'metadata' not in config:
        config['metadata'] = {}
    config['metadata']['model'] = model_name

    config['phase1_pretraining']['backbone'] = backbone_name
    config['phase2_downstream']['backbone'] = backbone_name
    config['phase1_pretraining']['ablations'] = ablation_state
    config['phase2_downstream']['ablations'] = ablation_state
    config['phase1_pretraining']['trial_name'] = trial_name
    config['phase2_downstream']['trial_name'] = trial_name
    config['phase1_pretraining']['seed'] = seed
    config['phase2_downstream']['seed'] = seed

    with open(yaml_path, 'w') as f: yaml.dump(config, f, default_flow_style=False, sort_keys=False)

def execute_command(command, step_description):
    print(f"\n>> [EXECUTING] {step_description}")
    result = subprocess.run(command, text=True)
    if result.returncode != 0: raise RuntimeError(f"Command failed with exit code {result.returncode}")

def get_completed_downstream_phases(dataset_name, backbone_name, trial_name="baseline", model_name="TMLPN_Downstream_v3"):
    isolated_backbone = f"{backbone_name}_{trial_name}" if trial_name != "baseline" else backbone_name
    model_dir = os.path.join("results", model_name, dataset_name, isolated_backbone)
    
    if not os.path.exists(model_dir): return 0
    runs = sorted(glob.glob(os.path.join(model_dir, "*_*")))
    return sum(1 for run in runs if os.path.exists(os.path.join(run, "results.json")))

# ====================================================================================
# --- AUTOMATIC ARCHITECTURE EVALUATOR ---
# ====================================================================================
def identify_optimal_backbone(dataset_name, backbones, seeds, model_name="TMLPN_Downstream_v3"):
    """
    Parses the results.json files across all baseline seeds to determine the absolute 
    best performing architecture based on mean mIoU, dynamically routing the pipeline.
    """
    print("\n" + "="*70)
    print(f"📊 AUTO-EVALUATING BASELINES FOR OPTIMAL BACKBONE SELECTION ({model_name})")
    print("="*70)
    
    best_backbone = backbones[0]
    highest_mean_miou = 0.0

    for backbone in backbones:
        seed_mious = []
        for seed in seeds:
            trial_name = f"baseline_seed{seed}"
            isolated_backbone = f"{backbone}_{trial_name}"
            model_dir = os.path.join("results", model_name, dataset_name, isolated_backbone)
            json_files = glob.glob(os.path.join(model_dir, "*_*", "results.json"))
            
            max_seed_miou = 0.0
            for jf in json_files:
                try:
                    with open(jf, 'r') as f:
                        data = json.load(f)
                        score = data.get("best_mIoU", 0.0)
                        if score > max_seed_miou:
                            max_seed_miou = score
                except Exception:
                    pass
            
            if max_seed_miou > 0:
                seed_mious.append(max_seed_miou)
        
        if seed_mious:
            mean_miou = sum(seed_mious) / len(seed_mious)
            print(f"[*] {backbone:<10} | Evaluated {len(seed_mious)}/5 seeds | Mean mIoU: {mean_miou:.4f}")
            if mean_miou > highest_mean_miou:
                highest_mean_miou = mean_miou
                best_backbone = backbone
        else:
            print(f"[*] {backbone:<10} | No completed baseline data found.")

    print(f"\n[+] OPTIMAL BACKBONE SELECTED: {best_backbone} (mIoU: {highest_mean_miou:.4f})")
    return best_backbone

# ====================================================================================
# --- MASTER ORCHESTRATOR ---
# ====================================================================================
def run_pipeline(dataset_name, backbones, model_name="TMLPN_Downstream_v3"):
    # --- DATA PROTECTION SAFETY CHECK ---
    legacy_dir = os.path.join("results", "TMLPN_Downstream")
    backup_dir = os.path.join("results", "TMLPN_Downstream_v1")
    if os.path.exists(legacy_dir):
        print(f"[*] Detected legacy results. Automatically securing directory to {backup_dir} ...")
        os.rename(legacy_dir, backup_dir)

    config_path = "config.yaml"
    backup_path = "config.yaml.backup"
    
    academic_seeds = [42, 1024, 2048, 4096, 8192] 
    
    control_optimal = {
        "enable_modality_isolation": True,
        "variance_type": "spatial",
        "gdl_type": "global_anchored",
        "enable_kd": True,
        "mask_strategy": "multi_block",
        "enable_covariance_penalty": True,
        "enable_context_consistency": True
    }

    ablation_matrix = {
        "Control_Optimal": control_optimal,
        "Ablation_NaiveFusion": {**control_optimal, "enable_modality_isolation": False},
        "Ablation_BatchVariance": {**control_optimal, "variance_type": "batch"},
        "Ablation_NoVariance": {**control_optimal, "variance_type": "none"}, 
        "Ablation_NoKD": {**control_optimal, "enable_kd": False},            
        "Ablation_RandomMasking": {**control_optimal, "mask_strategy": "random"},
        "Ablation_NoCovariance": {**control_optimal, "enable_covariance_penalty": False},
        "Ablation_NoContext": {**control_optimal, "enable_context_consistency": False}
    }

    print(f"[*] Initiating {model_name} MLOps Orchestration Pipeline for Dataset: {dataset_name}")
    shutil.copy(config_path, backup_path)

    try:
        # --- PART 1: PRIMARY BASELINE CYCLES (Multi-Seed) ---
        for seed in academic_seeds:
            for backbone in backbones:
                trial_name = f"baseline_seed{seed}"
                
                print("\n" + "="*70)
                print(f"🚀 INITIATING CYCLE: Backbone [{backbone}] | Seed: {seed}")
                print("="*70)
                
                inject_configuration(config_path, backbone, control_optimal, trial_name=trial_name, seed=seed, model_name=model_name)
                
                jepa_weights = os.path.join("weights", model_name, dataset_name, trial_name, f"jepa_checkpoint_{backbone}.pt")
                
                if not os.path.exists(jepa_weights): 
                    execute_command(["uv", "run", "pretrain_jepa.py"], f"Phase 1: MM-JEPA ({backbone} | Seed {seed})")
                else: 
                    print(f"[>>] Phase 1 Pre-training already completed for {backbone} (Seed {seed}). Skipping.")

                completed_phases = get_completed_downstream_phases(dataset_name, backbone, trial_name=trial_name, model_name=model_name)
                remaining_calls = 5 - completed_phases
                if remaining_calls == 0: 
                    print(f"[>>] All 5 Downstream phases already completed for {backbone} (Seed {seed}). Skipping.")
                else:
                    for i in range(remaining_calls):
                        current_phase_index = completed_phases + i + 1
                        execute_command(["uv", "run", "train_downstream.py"], f"Phase 2 Call [{current_phase_index}/5] ({backbone} | Seed {seed})")

        # --- AUTO-SELECT OPTIMAL BACKBONE ---
        optimal_backbone = identify_optimal_backbone(dataset_name, backbones, academic_seeds, model_name=model_name)

        # --- PART 2: ABLATION STUDIES ---
        print("\n" + "="*70)
        print(f"🔬 INITIATING PART 2: ABLATION STUDIES (Target: {optimal_backbone} | N=5)")
        print("="*70)

        for seed in academic_seeds:
            for trial_base_name, ablation_state in ablation_matrix.items():
                trial_name = f"{trial_base_name}_seed{seed}"
                inject_configuration(config_path, optimal_backbone, ablation_state, trial_name=trial_name, seed=seed, model_name=model_name)
                
                jepa_weights = os.path.join("weights", model_name, dataset_name, trial_name, f"jepa_checkpoint_{optimal_backbone}.pt")
                
                if not os.path.exists(jepa_weights): 
                    execute_command(["uv", "run", "pretrain_jepa.py"], f"Ablation Phase 1: {trial_name}")
                else: 
                    print(f"[>>] Ablation Phase 1 completed for {trial_name}. Skipping.")

                if get_completed_downstream_phases(dataset_name, optimal_backbone, trial_name=trial_name, model_name=model_name) == 0:
                    execute_command(["uv", "run", "train_downstream.py"], f"Ablation Phase 2: {trial_name}")
                else: 
                    print(f"[>>] Ablation Phase 2 completed for {trial_name}. Skipping.")
                    
        # --- PART 3: V3 COMPONENT ISOLATION ---
        v3_isolation_matrix = {
            "Ablation_NoLoRA": {**control_optimal, "use_lora": False},
            "Ablation_NoLLRD": {**control_optimal, "llrd_decay": 1.0},
            "Ablation_NoFeatureKD": {**control_optimal, "enable_kd": False}
        }
        
        print("\n" + "="*70)
        print(f"🧬 INITIATING PART 3: V3 COMPONENT ISOLATION (Target: {optimal_backbone} | N=5)")
        print("="*70)
        
        for seed in academic_seeds:
            for trial_base_name, ablation_state in v3_isolation_matrix.items():
                trial_name = f"{trial_base_name}_seed{seed}"
                inject_configuration(config_path, optimal_backbone, ablation_state, trial_name=trial_name, seed=seed, model_name=model_name)
                
                jepa_weights = os.path.join("weights", model_name, dataset_name, trial_name, f"jepa_checkpoint_{optimal_backbone}.pt")
                
                if not os.path.exists(jepa_weights): 
                    execute_command(["uv", "run", "pretrain_jepa.py"], f"Isolation Phase 1: {trial_name}")
                else: 
                    print(f"[>>] Isolation Phase 1 completed for {trial_name}. Skipping.")

                if get_completed_downstream_phases(dataset_name, optimal_backbone, trial_name=trial_name, model_name=model_name) == 0:
                    execute_command(["uv", "run", "train_downstream.py"], f"Isolation Phase 2: {trial_name}")
                else: 
                    print(f"[>>] Isolation Phase 2 completed for {trial_name}. Skipping.")

        print("\n[+] SUCCESS: Entire execution matrix completed.")

    except Exception as e:
        error_trace = traceback.format_exc()
        print(f"\n[-] CRITICAL PIPELINE FAILURE:\n{error_trace}")
        send_alert_email(subject="Pipeline Halt", error_message=error_trace)
    finally:
        if os.path.exists(backup_path): 
            shutil.move(backup_path, config_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MLOps Training Orchestrator")
    parser.add_argument("--dataset", type=str, default="MM5", help="Target dataset name")
    parser.add_argument("--model", type=str, default="TMLPN_Downstream_v3", help="Target architecture namespace")
    parser.add_argument("--backbones", nargs="+", default=["mit_b1", "mit_b2", "mit_b3", "mit_b4", "mit_b5"], help="List of backbones to execute")
    args = parser.parse_args()
    
    run_pipeline(args.dataset, args.backbones, model_name=args.model)