import os
import glob
import yaml
import json
import copy
import shutil
import smtplib
import datetime
import subprocess
import traceback
import argparse
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dotenv import load_dotenv

load_dotenv()

# Mirrors train_downstream.PHASE_SEQUENCE without importing PyTorch/CUDA in the orchestrator process
EXPECTED_DOWNSTREAM_PHASES = ("baseline", "hpo", "hero", "microtune", "export")

PHASE1_KEYS = (
    "enable_modality_isolation",
    "variance_type",
    "mask_strategy",
    "enable_covariance_penalty",
    "enable_context_consistency",
    "enable_dirac",
    "enable_scse",
    "enable_modality_dropout",
)

PHASE2_ONLY_KEYS = (
    "gdl_type",
    "enable_kd",
    "use_lora",
    "lora_r",
    "llrd_decay",
    "dynamic_lora_rank",
    "adaptive_llrd_decay",
    "enable_boundary_loss",
)

KNOWN_ABLATION_KEYS = frozenset(PHASE1_KEYS) | frozenset(PHASE2_ONLY_KEYS)


# ====================================================================================
# --- NOTIFICATION & DURABLE ERROR LOGGING ---
# ====================================================================================

def log_pipeline_error(subject: str, error_message: str, log_path: str = "pipeline_errors.log") -> None:
    timestamp = datetime.datetime.now().isoformat(timespec="seconds")
    try:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"\n{'=' * 72}\n[{timestamp}] {subject}\n{'=' * 72}\n{error_message}\n")
        print(f"[*] Fatal error persisted to {log_path}")
    except Exception as e:
        print(f"[-] CRITICAL: Failed to write error log {log_path}: {e}")


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
        with smtplib.SMTP(smtp_server, smtp_port, timeout=30) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            server.send_message(msg)
        print(f"[+] Alert email successfully dispatched to {recipient_email}")
    except Exception as e:
        print(f"[-] CRITICAL: Failed to send email alert: {e}")


# ====================================================================================
# --- IMMUTABLE CONFIGURATION BUILDER & DETERMINISM ENV ---
# ====================================================================================

def normalize_ablation_for_model(ablation_state: dict, model_name: str) -> dict:
    """
    Ensures Phase 1 and Phase 2 structural compatibility across architecture versions.
    Legacy V1/V2 downstream models hardcode enable_dirac=False and enable_scse=False.
    """
    normalized = dict(ablation_state)
    unknown_keys = set(normalized.keys()) - KNOWN_ABLATION_KEYS
    if unknown_keys:
        raise ValueError(
            f"[-] CRITICAL: Unknown ablation key(s) detected (would be silently ignored): {sorted(unknown_keys)}"
        )
    if model_name in {"TMLPN_Downstream_v1", "TMLPN_Downstream_v2"}:
        normalized["enable_dirac"] = False
        normalized["enable_scse"] = False
    return normalized


def write_trial_configuration(base_config: dict, out_dir: str, backbone_name: str,
                              ablation_state: dict, trial_name: str = "baseline",
                              seed: int = 42, model_name: str = "TMLPN_Downstream_v3") -> str:
    """
    Materializes an isolated YAML config per trial so master config.yaml is never mutated
    in place, ensuring crash/SIGKILL safety.
    """
    normalized_ablation = normalize_ablation_for_model(ablation_state, model_name)
    config = copy.deepcopy(base_config)
    config.setdefault('metadata', {})['model'] = model_name

    config['phase1_pretraining']['backbone'] = backbone_name
    config['phase2_downstream']['backbone'] = backbone_name

    config['phase1_pretraining']['ablations'] = dict(normalized_ablation)
    config['phase2_downstream']['ablations'] = dict(normalized_ablation)
    config['phase2_downstream']['use_lora'] = bool(normalized_ablation.get('use_lora', False))
    config['phase2_downstream']['llrd_decay'] = float(normalized_ablation.get('llrd_decay', 0.85))

    config['phase2_downstream'].setdefault('lora', {})
    if "lora_r" in normalized_ablation:
        r_val = int(normalized_ablation["lora_r"])
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


def build_deterministic_env(seed: int) -> dict:
    """
    Injects process-level determinism variables BEFORE spawning child Python interpreters
    (setting PYTHONHASHSEED inside an already-running process is a no-op).
    """
    env = os.environ.copy()
    env["PYTHONHASHSEED"] = str(seed)
    env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    return env


def execute_command(command: list, step_description: str, seed: int = 42, timeout: int = None) -> None:
    print(f"\n>> [EXECUTING] {step_description}")
    result = subprocess.run(command, text=True, env=build_deterministic_env(seed), timeout=timeout)
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
    """
    Strictly verifies that every Phase 1 key is explicitly present in both dictionaries
    and holds an identical value before allowing Phase 1 foundation weight reuse.
    """
    return all(
        (k in ablation_state) and (k in reference_state) and (ablation_state[k] == reference_state[k])
        for k in PHASE1_KEYS
    )


def reuse_phase1_foundation_if_compatible(model_name: str, dataset_name: str, backbone: str,
                                          seed: int, trial_name: str,
                                          ablation_state: dict, reference_state: dict) -> bool:
    """
    Copies converged Phase 1 encoder weights from baseline_seed{seed} when an ablation
    only modifies Phase 2 downstream parameters. Strictly checks the final exported
    context encoder artifact so incomplete checkpoints are never treated as converged.
    """
    norm_ablation = normalize_ablation_for_model(ablation_state, model_name)
    norm_reference = normalize_ablation_for_model(reference_state, model_name)

    target_dir = os.path.join("weights", model_name, dataset_name, trial_name)
    target_ckpt = os.path.join(target_dir, f"jepa_checkpoint_{backbone}.pt")
    target_enc = os.path.join(target_dir, f"jepa_context_encoder_{backbone}.pt")

    if os.path.exists(target_enc):
        return True

    if not is_phase1_equivalent(norm_ablation, norm_reference):
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


def record_skipped_ablation(model_name: str, dataset_name: str, trial_base_name: str,
                            backbone: str, reason: str) -> None:
    """Persists skipped ablation metadata so analyse_results.py documents and excludes it from BH-FDR."""
    base_dir = os.path.join("results", model_name, dataset_name)
    os.makedirs(base_dir, exist_ok=True)
    skip_path = os.path.join(base_dir, "skipped_ablations.json")

    payload = {}
    if os.path.exists(skip_path):
        try:
            with open(skip_path, "r") as f:
                payload = json.load(f)
        except json.JSONDecodeError:
            payload = {}

    payload[trial_base_name] = {
        "backbone": backbone,
        "skipped_reason": reason,
    }
    tmp_path = f"{skip_path}.tmp"
    with open(tmp_path, "w") as f:
        json.dump(payload, f, indent=4)
    os.replace(tmp_path, skip_path)


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
            json_files = glob.glob(os.path.join(model_dir, "*_*", "results.json"))
            # Sort explicitly by modification time (chronological) with directory name tie-breaker
            json_files.sort(key=lambda p: (os.path.getmtime(p), p))

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
                 config_path: str = "config.yaml", step_timeout: int = None) -> None:
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
    num_phases = len(EXPECTED_DOWNSTREAM_PHASES)
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
        # Explicitly disable dynamic_lora_rank so target ranks are never halved on mit_b3/b4/b5
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

                jepa_encoder = os.path.join("weights", model_name, dataset_name, trial_name, f"jepa_context_encoder_{backbone}.pt")
                if not os.path.exists(jepa_encoder):
                    execute_command(
                        ["uv", "run", "pretrain_jepa.py", "--config", trial_cfg],
                        f"Phase 1: MM-JEPA ({backbone} | Seed {seed})",
                        seed=seed,
                        timeout=step_timeout
                    )
                else:
                    print(f"[>>] Phase 1 Pre-training converged for {backbone} (Seed {seed}). Skipping.")

                completed_phases = get_completed_downstream_phases(dataset_name, backbone, trial_name=trial_name, model_name=model_name)
                remaining_calls = max(0, num_phases - completed_phases)
                if remaining_calls == 0:
                    print(f"[>>] All {num_phases} Downstream phases completed for {backbone} (Seed {seed}). Skipping.")
                else:
                    for i in range(remaining_calls):
                        current_phase_index = completed_phases + i + 1
                        execute_command(
                            ["uv", "run", "train_downstream.py", "--config", trial_cfg],
                            f"Phase 2 Call [{current_phase_index}/{num_phases}] ({backbone} | Seed {seed})",
                            seed=seed,
                            timeout=step_timeout
                        )

        # --- AUTO-SELECT OPTIMAL BACKBONE (VALIDATION-ONLY) ---
        # Note: analyse_results.py automatically aliases Part 1 baseline_seed* on optimal_backbone
        # as Control_Optimal without duplicating checkpoint folders on disk.
        optimal_backbone = identify_optimal_backbone(dataset_name, backbones, academic_seeds, model_name=model_name)

        # --- PART 2: ABLATION STUDIES ---
        print("\n" + "=" * 70)
        print(f"⚙️ INITIATING PART 2: ABLATION STUDIES (Target: {optimal_backbone} | N={len(academic_seeds)})")
        print("=" * 70)

        for trial_base_name, ablation_state in ablation_matrix.items():
            # Prevent degenerate self-distillation trial from diluting BH-FDR p-values
            if trial_base_name == "Ablation_NoKD" and optimal_backbone == teacher_bb:
                reason = f"self_distillation (optimal_backbone == teacher_backbone == {teacher_bb})"
                print(f"[>>] Skipping {trial_base_name}: {reason}.")
                record_skipped_ablation(model_name, dataset_name, trial_base_name, optimal_backbone, reason)
                continue

            for seed in academic_seeds:
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
                        f"Ablation Phase 1: {trial_name}",
                        seed=seed,
                        timeout=step_timeout
                    )

                completed_phases = get_completed_downstream_phases(dataset_name, optimal_backbone, trial_name=trial_name, model_name=model_name)
                remaining_calls = max(0, num_phases - completed_phases)
                if remaining_calls == 0:
                    print(f"[>>] Ablation Phase 2 completed for {trial_name}. Skipping.")
                else:
                    for i in range(remaining_calls):
                        current_phase_index = completed_phases + i + 1
                        execute_command(
                            ["uv", "run", "train_downstream.py", "--config", trial_cfg],
                            f"Ablation Phase 2 Call [{current_phase_index}/{num_phases}] ({trial_name})",
                            seed=seed,
                            timeout=step_timeout
                        )

        # --- PART 3: V3 COMPONENT ISOLATION ---
        print("\n" + "=" * 70)
        print(f"⚙️ INITIATING PART 3: V3 COMPONENT ISOLATION (Target: {optimal_backbone} | N={len(academic_seeds)})")
        print("=" * 70)

        for trial_base_name, ablation_state in v3_isolation_matrix.items():
            if optimal_backbone not in {'mit_b3', 'mit_b4', 'mit_b5'} and trial_base_name in {
                "Ablation_NoDynamicLoRA", "Ablation_NoAdaptiveLLRD"
            }:
                reason = f"heuristic_inactive_on_{optimal_backbone} (only active for mit_b3/b4/b5)"
                print(f"[>>] Skipping {trial_base_name}: {reason}.")
                record_skipped_ablation(model_name, dataset_name, trial_base_name, optimal_backbone, reason)
                continue

            for seed in academic_seeds:
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
                        f"Isolation Phase 1: {trial_name}",
                        seed=seed,
                        timeout=step_timeout
                    )

                completed_phases = get_completed_downstream_phases(dataset_name, optimal_backbone, trial_name=trial_name, model_name=model_name)
                remaining_calls = max(0, num_phases - completed_phases)
                if remaining_calls == 0:
                    print(f"[>>] Isolation Phase 2 completed for {trial_name}. Skipping.")
                else:
                    for i in range(remaining_calls):
                        current_phase_index = completed_phases + i + 1
                        execute_command(
                            ["uv", "run", "train_downstream.py", "--config", trial_cfg],
                            f"Isolation Phase 2 Call [{current_phase_index}/{num_phases}] ({trial_name})",
                            seed=seed,
                            timeout=step_timeout
                        )

        print("\n[+] SUCCESS: Entire execution matrix completed.")

    except Exception:
        error_trace = traceback.format_exc()
        print(f"\n[-] CRITICAL PIPELINE FAILURE:\n{error_trace}")
        log_pipeline_error(subject=f"Pipeline Halt ({model_name} | {dataset_name})", error_message=error_trace)
        send_alert_email(subject="Pipeline Halt", error_message=error_trace)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MLOps Training Orchestrator")
    parser.add_argument("--dataset", type=str, default="MM5", help="Target dataset name")
    parser.add_argument("--model", type=str, default="TMLPN_Downstream_v3", help="Target architecture namespace")
    parser.add_argument("--config", type=str, default="config.yaml", help="Master configuration YAML path")
    parser.add_argument("--backbones", nargs="+", default=["mit_b1", "mit_b2", "mit_b3", "mit_b4", "mit_b5"], help="List of backbones to execute")
    parser.add_argument("--step_timeout", type=int, default=None, help="Optional per-subprocess timeout in seconds (default: None)")
    args = parser.parse_args()

    run_pipeline(
        args.dataset,
        args.backbones,
        model_name=args.model,
        config_path=args.config,
        step_timeout=args.step_timeout
    )