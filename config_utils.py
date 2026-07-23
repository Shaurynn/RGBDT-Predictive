import yaml
import argparse

def load_config(config_path="config.yaml"):
    """Loads the YAML configuration file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def parse_with_config(description):
    """Parses CLI arguments, allowing the config path to be specified alongside dynamic ablation overrides."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to configuration YAML")
    
    # --- V3 Structural Overrides for rapid ablation testing via ml_pipeline.py ---
    parser.add_argument("--patience", type=int, default=None, help="Override early stopping patience")
    parser.add_argument("--grad_accum", type=int, default=None, help="Override gradient accumulation steps")
    parser.add_argument("--dynamic_lora", action="store_true", help="Force enable dynamic LoRA ranking")
    parser.add_argument("--adaptive_llrd", action="store_true", help="Force enable adaptive LLRD scaling")
    
    args = parser.parse_args()
    config = load_config(args.config)
    
    # Inject CLI overrides into the parsed dictionary dictionary at runtime
    if args.patience is not None:
        if 'phase2_downstream' in config:
            config['phase2_downstream']['early_stopping_patience'] = args.patience
            
    if args.grad_accum is not None:
        if 'phase2_downstream' in config:
            config['phase2_downstream']['gradient_accumulation_steps'] = args.grad_accum
            
    if args.dynamic_lora:
        if 'phase2_downstream' in config and 'ablations' in config['phase2_downstream']:
            config['phase2_downstream']['ablations']['dynamic_lora_rank'] = True
            
    if args.adaptive_llrd:
        if 'phase2_downstream' in config and 'ablations' in config['phase2_downstream']:
            config['phase2_downstream']['ablations']['adaptive_llrd_decay'] = True
            
    return args, config