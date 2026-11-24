import os
import glob
import yaml
import json
import copy
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

def send_alert_email(subject: str, error_message: str) -> None:
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
# --- IMMUTABLE CONFIGURATION BUILDER ---
# ====================================================================================

def write_trial_configuration(base_config: dict, out_dir: str, backbone_name: str,
                              ablation_state: dict, trial_name: str = "baseline",
                              seed: int = 42, model_name: str = "TMLPN_Downstream_v3") -> str:
    """
    Materializes an isolated YAML config per trial so master config.yaml is never mutated
    in place, ensuring crash/SIGKILL safety.
    """
    config = copy.deepcopy(base_config)
    config.setdefault('metadata', {})['model'] = model_name

    config['phase1_pretraining']['backbone'] = backbone_name
    config['phase2_downstream']['backbone'] = backbone_name

    config['phase1_pretraining']['ablations'] = dict(ablation_state)
    config['phase2_downstream']['ablations'] = dict(ablation_state)
    config['phase2_downstream']['use_lora'] = bool(ablation_state.get('use_lora', False))
    config['phase2_downstream']['llrd_decay'] = float(ablation_state.get('llrd_decay', 0.85))

    config['phase2_downstream'].setdefault('lora', {})
    if "lora_r" in ablation_state:
        r_val = int(ablation_state["lora_r"])
        config['phase2_downstream']['lora']['r'] = r_val
        config['phase2_downstream']['lora']['alpha'] = r_val * 2
    else:
        config['phase2_downstream']['lora']['r'] = 8
        config['phase2_downstream']['lora']['alpha'] = 16

    config['phase1_pretraining']['trial_name'] = trial_name
    config['phase2_downstream']['trial_name'] = trial_name
    config['phase1_pretraining']['seed'] = seed
    config['phase2_downstream']['seed'] = seed

    os.makedirs(out_dir, exist_ok=True)
    trial_cfg_path = os.path.join(out_dir, f"{model_name}_{backbone_name}_{trial_name}.yaml")
    tmp_path = f"{trial_cfg_path}.tmp"
    with open(tmp_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    os.replace(tmp_path, trial_cfg_path)
    return trial_cfg_path


def execute_command(command: list, step_description: str) -> None:
    print(f"\n>> [EXECUTING] {step_description}")
    result = subprocess.run(command, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {result.returncode}: {' '.join(command)}")


def get_completed_downstream_phases(dataset_name: str, backbone_name: str,
                                    trial_name: str = "baseline",
                                    model_name: str = "TMLPN_Downstream_v3") -> int:
    isolated_backbone = f"{backbone_name}_{trial_name}" if trial_name != "baseline" else backbone_name
    model_dir = os.path.join("results", model_name, dataset_name, isolated_backbone)
    if not os.path.exists(model_dir):
        return 0
    runs = sorted(glob.glob(os.path.join(model_dir, "*_*")))
    return sum(1 for run in runs if os.path.exists(os.path.join(run, "results.json")))


def is_phase1_equivalent(ablation_state: dict, reference_state: dict) -> bool:
    phase1_keys = [
        "enable_modality_isolation",
        "variance_type",
        "mask_strategy",
        "enable_covariance_penalty",
        "enable_context_consistency",
        "enable_dirac",
        "enable_scse",
        "enable_modality_dropout"
    ]
    return all(ablation_state.get(k) == reference_state.get(k) for k in phase1_keys)


def reuse_phase1_foundation_if_compatible(model_name: str, dataset_name: str, backbone: str,
                                          seed: int, trial_name: str,
                                          ablation_state: dict, reference_state: dict) -> bool:
    """
    Reuses converged Phase 1 encoder weights from baseline_seed{seed} when an ablation
    only modifies Phase 2 downstream parameters. Strictly checks the final exported
    context encoder artifact so incomplete checkpoints are never treated as converged.
    """
    target_dir = os.path.join("weights", model_name, dataset_name, trial_name)
    target_ckpt = os.path.join(target_dir, f"jepa_checkpoint_{backbone}.pt")
    target_enc = os.path.join(target_dir, f"jepa_context_encoder_{backbone}.pt")

    if os.path.exists(target_enc):
        return True

    if not is_phase1_equivalent(ablation_state, reference_state):
        return False

    source_dir = os.path.join("weights", model_name, dataset_name, f"baseline_seed{seed}")
    source_ckpt = os.path.join(source_dir, f"jepa_checkpoint_{backbone}.pt")
    source_enc = os.path.join(source_dir, f"jepa_context_encoder_{backbone}.pt")

    if os.path.exists(source_enc):
        os.makedirs(target_dir, exist_ok=True)
        shutil.copy2(source_enc, target_enc)
        if os.path.exists(source_ckpt):
            shutil.copy2(source_ckpt, target_ckpt)
        print(f"[>>] Reused compatible Phase 1 foundation weights from baseline_seed{seed} for {trial_name}.")
        return True

    return False


def sync_baseline_to_control_optimal(model_name: str, dataset_name: str,
                                     optimal_backbone: str, seeds: list) -> None:
    """
    Mirrors Part 1 converged baseline_seed{seed} results on optimal_backbone into
    Control_Optimal_seed{seed} so Part 2 does not re-execute 25 identical downstream runs.
    """
    for seed in seeds:
        src_dir = os.path.join("results", model_name, dataset_name, f"{optimal_backbone}_baseline_seed{seed}")
        dst_dir = os.path.join("results", model_name, dataset_name, f"{optimal_backbone}_Control_Optimal_seed{seed}")
        if os.path.exists(src_dir) and not os.path.exists(dst_dir):
            shutil.copytree(src_dir, dst_dir)
            print(f"[>>] Mirrored Part 1 baseline results into Control_Optimal_seed{seed} for {optimal_backbone}.")


# ====================================================================================
# --- AUTOMATIC ARCHITECTURE EVALUATOR (VALIDATION-ONLY, ZERO TEST LEAKAGE) ---
# ====================================================================================

def identify_optimal_backbone(dataset_name: str, backbones: list, seeds: list,
                              model_name: str = "TMLPN_Downstream_v3") -> str:
    """
    Selects the highest-performing backbone strictly using validation mIoU
    (`val_mIoU_checkpoint` from val.csv) across all baseline seeds so test.csv
    remains completely blind to architecture selection.
    """
    print("\n" + "=" * 70)
    print(f"⚙️ AUTO-EVALUATING BASELINES ON VALIDATION SPLIT ({model_name})")
    print("=" * 70)

    best_backbone = backbones[0]
    highest_mean_val_miou = -1.0

    for backbone in backbones:
        seed_val_mious = []
        for seed in seeds:
            trial_name = f"baseline_seed{seed}"
            isolated_backbone = f"{backbone}_{trial_name}"
            model_dir = os.path.join("results", model_name, dataset_name, isolated_backbone)
            json_files = sorted(glob.glob(os.path.join(model_dir, "*_*", "results.json")))

            microtune_val = None
            fallback_val = -1.0
            for jf in json_files:
                try:
                    with open(jf, 'r') as f:
                        data = json.load(f)
                    phase = data.get("phase", "").lower()
                    val_score = data.get("val_mIoU_checkpoint", None)
                    if val_score is None:
                        continue
                    if phase == "microtune":
                        microtune_val = float(val_score)
                    elif phase in ["baseline", "hero"] and float(val_score) > fallback_val:
                        fallback_val = float(val_score)
                except Exception:
                    pass

            selected = microtune_val if microtune_val is not None else fallback_val
            if selected >= 0.0:
                seed_val_mious.append(selected)

        if seed_val_mious:
            mean_val = sum(seed_val_mious) / len(seed_val_mious)
            print(f"[*] {backbone:<10} | Evaluated {len(seed_val_mious)}/{len(seeds)} seeds | Mean Val mIoU: {mean_val:.4f}")
            if mean_val > highest_mean_val_miou:
                highest_mean_val_miou = mean_val
                best_backbone = backbone
        else:
            print(f"[*] {backbone:<10} | No completed validation data found.")

    print(f"\n[+] OPTIMAL BACKBONE SELECTED (VAL-ONLY): {best_backbone} (Val mIoU: {highest_mean_val_miou:.4f})")
    return best_backbone


# ====================================================================================
# --- MASTER ORCHESTRATOR ---
# ====================================================================================

def run_pipeline(dataset_name: str, backbones: list, model_name: str = "TMLPN_Downstream_v3",
                 config_path: str = "config.yaml") -> None:
    legacy_dir = os.path.join("results", "TMLPN_Downstream")
    backup_dir = os.path.join("results", "TMLPN_Downstream_v1")
    if os.path.exists(legacy_dir) and not os.path.exists(backup_dir):
        print(f"[*] Detected legacy results. Securing directory to {backup_dir} ...")
        os.rename(legacy_dir, backup_dir)

    with open(config_path, 'r') as f:
        base_config = yaml.safe_load(f)

    teacher_bb = base_config.get('phase2_downstream', {}).get('teacher_backbone', 'mit_b4')

    # Order teacher backbone first within each seed cycle so student KD runs have converged teacher weights
    ordered_backbones = []
    if teacher_bb in backbones:
        ordered_backbones.append(teacher_bb)
    for bb in backbones:
        if bb not in ordered_backbones:
            ordered_backbones.append(bb)

    academic_seeds = [42, 1024, 2048, 4096, 8192]
    trial_cfg_dir = os.path.join(".pipeline_configs", model_name, dataset_name)

    control_optimal = {
        "enable_modality_isolation": True,
        "variance_type": "spatial",
        "gdl_type": "global_anchored",
        "enable_kd": True,
        "mask_strategy": "random",
        "enable_covariance_penalty": True,
        "enable_context_consistency": True,
        "use_lora": False,
        "llrd_decay": 0.85,
        "enable_dirac": True,
        "dynamic_lora_rank": True,
        "adaptive_llrd_decay": True,
        "enable_scse": True,
        "enable_modality_dropout": True,
        "enable_boundary_loss": True
    }

    ablation_matrix = {
        "Ablation_NaiveFusion": {**control_optimal, "enable_modality_isolation": False},
        "Ablation_NoDirac": {**control_optimal, "enable_dirac": False},
        "Ablation_NoSCSE": {**control_optimal, "enable_scse": False},
        "Ablation_NoModalityDropout": {**control_optimal, "enable_modality_dropout": False},
        "Ablation_NoBoundaryLoss": {**control_optimal, "enable_boundary_loss": False},
        "Ablation_BatchVariance": {**control_optimal, "variance_type": "batch"},
        "Ablation_NoVariance": {**control_optimal, "variance_type": "none"},
        "Ablation_NoKD": {**control_optimal, "enable_kd": False},
        "Ablation_MultiBlockMasking": {**control_optimal, "mask_strategy": "multi_block"},
        "Ablation_NoCovariance": {**control_optimal, "enable_covariance_penalty": False},
        "Ablation_NoContext": {**control_optimal, "enable_context_consistency": False}
    }

    v3_isolation_matrix = {
        "Ablation_WithLoRA": {**control_optimal, "use_lora": True},
        "Ablation_NoLLRD": {**control_optimal, "llrd_decay": 1.0, "adaptive_llrd_decay": False},
        "Ablation_WithLoRA_NoLLRD": {**control_optimal, "use_lora": True, "llrd_decay": 1.0, "adaptive_llrd_decay": False},
        "Ablation_Vanilla_V1": {
            **control_optimal,
            "use_lora": False,
            "llrd_decay": 1.0,
            "adaptive_llrd_decay": False,
            "enable_kd": False,
            "enable_scse": False,
            "enable_boundary_loss": False
        },
        # Explicitly disable dynamic_lora_rank so target ranks are not silently halved on mit_b3/b4/b5
        "LoRA_Rank_4": {**control_optimal, "use_lora": True, "lora_r": 4, "dynamic_lora_rank": False},
        "LoRA_Rank_16": {**control_optimal, "use_lora": True, "lora_r": 16, "dynamic_lora_rank": False},
        "LoRA_Rank_32": {**control_optimal, "use_lora": True, "lora_r": 32, "dynamic_lora_rank": False},
        "Ablation_NoDynamicLoRA": {**control_optimal, "use_lora": True, "dynamic_lora_rank": False},
        "Ablation_NoAdaptiveLLRD": {**control_optimal, "adaptive_llrd_decay": False}
    }

    print(f"[*] Initiating {model_name} MLOps Orchestration Pipeline for Dataset: {dataset_name}")

    try:
        # --- PART 1: PRIMARY BASELINE CYCLES (Multi-Seed, Teacher-First Ordering) ---
        for seed in academic_seeds:
            for backbone in ordered_backbones:
                trial_name = f"baseline_seed{seed}"

                print("\n" + "=" * 70)
                print(f"⚙️ INITIATING CYCLE: Backbone [{backbone}] | Seed: {seed}")
                print("=" * 70)

                trial_cfg = write_trial_configuration(
                    base_config, trial_cfg_dir, backbone, control_optimal,
                    trial_name=trial_name, seed=seed, model_name=model_name
                )

                # Strictly check exported converged context encoder (written only after full Phase 1 completion)
                jepa_encoder = os.path.join("weights", model_name, dataset_name, trial_name, f"jepa_context_encoder_{backbone}.pt")
                if not os.path.exists(jepa_encoder):
                    execute_command(
                        ["uv", "run", "pretrain_jepa.py", "--config", trial_cfg],
                        f"Phase 1: MM-JEPA ({backbone} | Seed {seed})"
                    )
                else:
                    print(f"[>>] Phase 1 Pre-training converged for {backbone} (Seed {seed}). Skipping.")

                completed_phases = get_completed_downstream_phases(dataset_name, backbone, trial_name=trial_name, model_name=model_name)
                remaining_calls = max(0, 5 - completed_phases)
                if remaining_calls == 0:
                    print(f"[>>] All 5 Downstream phases completed for {backbone} (Seed {seed}). Skipping.")
                else:
                    for i in range(remaining_calls):
                        current_phase_index = completed_phases + i + 1
                        execute_command(
                            ["uv", "run", "train_downstream.py", "--config", trial_cfg],
                            f"Phase 2 Call [{current_phase_index}/5] ({backbone} | Seed {seed})"
                        )

        # --- AUTO-SELECT OPTIMAL BACKBONE (VALIDATION-ONLY) ---
        optimal_backbone = identify_optimal_backbone(dataset_name, backbones, academic_seeds, model_name=model_name)
        sync_baseline_to_control_optimal(model_name, dataset_name, optimal_backbone, academic_seeds)

        if optimal_backbone == teacher_bb:
            print(f"[!] NOTICE: Optimal backbone ({optimal_backbone}) equals teacher_backbone ({teacher_bb}). "
                  f"KD is self-skipped; Ablation_NoKD will match Control_Optimal.")

        # --- PART 2: ABLATION STUDIES ---
        print("\n" + "=" * 70)
        print(f"⚙️ INITIATING PART 2: ABLATION STUDIES (Target: {optimal_backbone} | N=5)")
        print("=" * 70)

        for seed in academic_seeds:
            for trial_base_name, ablation_state in ablation_matrix.items():
                trial_name = f"{trial_base_name}_seed{seed}"
                trial_cfg = write_trial_configuration(
                    base_config, trial_cfg_dir, optimal_backbone, ablation_state,
                    trial_name=trial_name, seed=seed, model_name=model_name
                )

                reused = reuse_phase1_foundation_if_compatible(
                    model_name, dataset_name, optimal_backbone, seed,
                    trial_name, ablation_state, control_optimal
                )
                if not reused:
                    execute_command(
                        ["uv", "run", "pretrain_jepa.py", "--config", trial_cfg],
                        f"Ablation Phase 1: {trial_name}"
                    )

                completed_phases = get_completed_downstream_phases(dataset_name, optimal_backbone, trial_name=trial_name, model_name=model_name)
                remaining_calls = max(0, 5 - completed_phases)
                if remaining_calls == 0:
                    print(f"[>>] Ablation Phase 2 completed for {trial_name}. Skipping.")
                else:
                    for i in range(remaining_calls):
                        current_phase_index = completed_phases + i + 1
                        execute_command(
                            ["uv", "run", "train_downstream.py", "--config", trial_cfg],
                            f"Ablation Phase 2 Call [{current_phase_index}/5] ({trial_name})"
                        )

        # --- PART 3: V3 COMPONENT ISOLATION ---
        print("\n" + "=" * 70)
        print(f"⚙️ INITIATING PART 3: V3 COMPONENT ISOLATION (Target: {optimal_backbone} | N=5)")
        print("=" * 70)

        for seed in academic_seeds:
            for trial_base_name, ablation_state in v3_isolation_matrix.items():
                if optimal_backbone not in {'mit_b3', 'mit_b4', 'mit_b5'} and trial_base_name in {
                    "Ablation_NoDynamicLoRA", "Ablation_NoAdaptiveLLRD"
                }:
                    print(f"[>>] Skipping {trial_base_name}_seed{seed}: only active on high-capacity backbones (mit_b3/b4/b5).")
                    continue

                trial_name = f"{trial_base_name}_seed{seed}"
                trial_cfg = write_trial_configuration(
                    base_config, trial_cfg_dir, optimal_backbone, ablation_state,
                    trial_name=trial_name, seed=seed, model_name=model_name
                )

                reused = reuse_phase1_foundation_if_compatible(
                    model_name, dataset_name, optimal_backbone, seed,
                    trial_name, ablation_state, control_optimal
                )
                if not reused:
                    execute_command(
                        ["uv", "run", "pretrain_jepa.py", "--config", trial_cfg],
                        f"Isolation Phase 1: {trial_name}"
                    )

                completed_phases = get_completed_downstream_phases(dataset_name, optimal_backbone, trial_name=trial_name, model_name=model_name)
                remaining_calls = max(0, 5 - completed_phases)
                if remaining_calls == 0:
                    print(f"[>>] Isolation Phase 2 completed for {trial_name}. Skipping.")
                else:
                    for i in range(remaining_calls):
                        current_phase_index = completed_phases + i + 1
                        execute_command(
                            ["uv", "run", "train_downstream.py", "--config", trial_cfg],
                            f"Isolation Phase 2 Call [{current_phase_index}/5] ({trial_name})"
                        )

        print("\n[+] SUCCESS: Entire execution matrix completed.")

    except Exception:
        error_trace = traceback.format_exc()
        print(f"\n[-] CRITICAL PIPELINE FAILURE:\n{error_trace}")
        send_alert_email(subject="Pipeline Halt", error_message=error_trace)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MLOps Training Orchestrator")
    parser.add_argument("--dataset", type=str, default="MM5", help="Target dataset name")
    parser.add_argument("--model", type=str, default="TMLPN_Downstream_v3", help="Target architecture namespace")
    parser.add_argument("--config", type=str, default="config.yaml", help="Master configuration YAML path")
    parser.add_argument("--backbones", nargs="+", default=["mit_b1", "mit_b2", "mit_b3", "mit_b4", "mit_b5"], help="List of backbones to execute")
    args = parser.parse_args()

    run_pipeline(args.dataset, args.backbones, model_name=args.model, config_path=args.config)