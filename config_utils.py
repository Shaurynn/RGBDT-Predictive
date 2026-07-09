import yaml
import argparse

def load_config(config_path="config.yaml"):
    """Loads the YAML configuration file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def parse_with_config(description):
    """Parses CLI arguments, allowing the config path to be specified."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to configuration YAML")
    # You can still add specific CLI overrides here if needed for cluster scheduling
    args = parser.parse_args()
    
    config = load_config(args.config)
    return args, config