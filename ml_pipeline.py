import os
import glob
import yaml
import shutil
import smtplib
import subprocess
import traceback
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
def inject_configuration(yaml_path, backbone_name, ablation_state, trial_name="baseline", seed=42):
    """
    Injects the backbone, ablation state, namespace identifier, AND the statistical seed.
    """
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)

    # Core routing
    config['phase1_pretraining']['backbone'] = backbone_name
    config['phase2_downstream']['backbone'] = backbone_name
    config['phase1_pretraining']['ablations'] = ablation_state
    config['phase2_downstream']['ablations'] = ablation_state
    
    # Namespace isolation
    config['phase1_pretraining']['trial_name'] = trial_name
    config['phase2_downstream']['trial_name'] = trial_name
    
    # Statistical Seed Injection
    config['phase1_pretraining']['seed'] = seed
    config['phase2_downstream']['seed'] = seed

    with open(yaml_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

def execute_command(command, step_description):
    print(f"\n>> [EXECUTING] {step_description}")
    result = subprocess.run(command, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {result.returncode}:\n{' '.join(command)}")

def get_completed_downstream_phases(dataset_name, backbone_name, trial_name="baseline"):
    isolated_backbone = f"{backbone_name}_{trial_name}" if trial_name != "baseline" else backbone_name
    model_dir = os.path.join("results", "TMLPN_Downstream", dataset_name, isolated_backbone)
    
    if not os.path.exists(model_dir):
        return 0
        
    runs = sorted(glob.glob(os.path.join(model_dir, "*_*")))
    completed_count = sum(1 for run in runs if os.path.exists(os.path.join(run, "results.json")))
    return completed_count

def run_pipeline():
    config_path = "config.yaml"
    backup_path = "config.yaml.backup"
    
    dataset_name = "MM5" 
    backbones = ["mit_b3", "mit_b4", "mit_b5"]
    
    # N=3 CVPR Minimum Viable Standard
    academic_seeds = [42, 1024, 2048] 
    
    control_optimal = {
        "enable_modality_isolation": True,
        "variance_type": "spatial",
        "gdl_type": "global_anchored",
        "enable_kd": True
    }

    # The ablation target matrix includes the control itself to establish the statistical baseline
    ablation_matrix = {
        "Control_Optimal": control_optimal,
        "Ablation_NaiveFusion": {**control_optimal, "enable_modality_isolation": False},
        "Ablation_BatchVariance": {**control_optimal, "variance_type": "batch"},
    }

    print("[*] Initiating MLOps Orchestration Pipeline...")
    shutil.copy(config_path, backup_path)

    try:
        # =====================================================================
        # PART 1: PRIMARY BASELINE CYCLES (Canonical Seed 42)
        # =====================================================================
        for backbone in backbones:
            print("\n" + "="*70)
            print(f"🚀 INITIATING CYCLE: Backbone [{backbone}] | Canonical Seed: 42")
            print("="*70)
            
            inject_configuration(config_path, backbone, control_optimal, trial_name="baseline", seed=42)

            # Phase 1
            jepa_weights = os.path.join("weights", dataset_name, "baseline", f"jepa_checkpoint_{backbone}.pt")
            if not os.path.exists(jepa_weights):
                execute_command(["uv", "run", "pretrain_jepa.py"], f"Phase 1: MM-JEPA ({backbone})")
            else:
                print(f"[>>] Phase 1 Pre-training already completed for {backbone}. Skipping.")

            # Phase 2
            completed_phases = get_completed_downstream_phases(dataset_name, backbone, trial_name="baseline")
            remaining_calls = 5 - completed_phases
            if remaining_calls == 0:
                print(f"[>>] All 5 Downstream phases already completed for {backbone}. Skipping.")
            else:
                for i in range(remaining_calls):
                    current_phase_index = completed_phases + i + 1
                    execute_command(["uv", "run", "train_downstream.py"], f"Phase 2 Call [{current_phase_index}/5] ({backbone})")

        # =====================================================================
        # PART 2: ABLATION STUDIES (N=3 Seeds)
        # =====================================================================
        ablation_target_backbone = "mit_b3" 

        print("\n" + "="*70)
        print("🔬 INITIATING ABLATION STUDIES (N=3 Statistical Matrix)")
        print("="*70)

        for seed in academic_seeds:
            for trial_base_name, ablation_state in ablation_matrix.items():
                
                # Create a highly specific namespace mapping to this seed and configuration
                trial_name = f"{trial_base_name}_seed{seed}"
                
                inject_configuration(config_path, ablation_target_backbone, ablation_state, trial_name=trial_name, seed=seed)
                
                # Phase 1 Check
                jepa_weights = os.path.join("weights", dataset_name, trial_name, f"jepa_checkpoint_{ablation_target_backbone}.pt")
                if not os.path.exists(jepa_weights):
                    execute_command(["uv", "run", "pretrain_jepa.py"], f"Ablation Phase 1: {trial_name}")
                else:
                    print(f"[>>] Ablation Phase 1 completed for {trial_name}. Skipping.")

                # Phase 2 Check (Only Baseline evaluation for ablations)
                completed_phases = get_completed_downstream_phases(dataset_name, ablation_target_backbone, trial_name=trial_name)
                if completed_phases == 0:
                    execute_command(["uv", "run", "train_downstream.py"], f"Ablation Phase 2 (Baseline): {trial_name}")
                else:
                    print(f"[>>] Ablation Phase 2 completed for {trial_name}. Skipping.")

        print("\n[+] SUCCESS: Entire execution matrix completed.")

    except Exception as e:
        error_trace = traceback.format_exc()
        print(f"\n[-] CRITICAL PIPELINE FAILURE:\n{error_trace}")
        send_alert_email(subject="Pipeline Halt", error_message=error_trace)
    finally:
        if os.path.exists(backup_path):
            shutil.move(backup_path, config_path)

if __name__ == "__main__":
    run_pipeline()