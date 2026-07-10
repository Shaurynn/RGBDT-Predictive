import os
import glob
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split

def parse_dataset_structure(data_root, dataset_name):
    """
    Extensible routing logic to parse different dataset folder structures.
    Returns a list of dictionaries containing exact paths for all modalities.
    """
    data = []
    
    # Generic generic traversal (e.g., finding the anchor RGB image)
    # Can be expanded with elif dataset_name == 'other_dataset':
    search_pattern = os.path.join(data_root, "**", "*_rgb.png")
    anchor_files = glob.glob(search_pattern, recursive=True)
    
    for rgb_path in anchor_files:
        # Resolve modalities explicitly at the generation phase
        depth_path = rgb_path.replace('_rgb.png', '_depth.png')
        thermal_path = rgb_path.replace('_rgb.png', '_thermal.png')
        mask_path = rgb_path.replace('_rgb.png', '_mask.png') # If downstream
        
        # Verify multi-modal integrity
        if os.path.exists(depth_path) and os.path.exists(thermal_path):
            class_name = os.path.basename(os.path.dirname(rgb_path))
            data.append({
                "dataset": dataset_name,
                "class_label": class_name,
                "rgb_path": rgb_path,
                "depth_path": depth_path,
                "thermal_path": thermal_path,
                "mask_path": mask_path if os.path.exists(mask_path) else None
            })
            
    return data

def main():
    parser = argparse.ArgumentParser(description="Generate deterministic dataset splits.")
    # ONLY the dataset name is required now, matching the execution config
    parser.add_argument("--dataset", type=str, required=True, help="Name of the dataset (e.g., MM5)")
    args = parser.parse_args()

    # Inherit the exact same routing convention used in pretrain_jepa.py
    data_root = os.path.join("dataset", args.dataset)
    
    if not os.path.exists(data_root):
        raise FileNotFoundError(
            f"[-] CRITICAL: Data directory not found at {data_root}. "
            f"Please ensure your dataset is placed or symlinked correctly."
        )

    output_dir = os.path.join("data", "splits", args.dataset)
    os.makedirs(output_dir, exist_ok=True)
    print(f"[*] Compiling agnostic manifest for dataset: {args.dataset} from {data_root}")

    # Pass the standardized data_root to the parser
    data_records = parse_dataset_structure(data_root, args.dataset)
    df = pd.DataFrame(data_records)
    
    if len(df) == 0:
        raise ValueError(f"[-] No valid 3-stream multi-modal data found in {data_root}")

    # Deterministic split generation
    train_df, temp_df = train_test_split(df, test_size=0.30, stratify=df['class_label'], random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.50, stratify=temp_df['class_label'], random_state=42)

    train_df.to_csv(os.path.join(output_dir, "train.csv"), index=False)
    val_df.to_csv(os.path.join(output_dir, "val.csv"), index=False)
    test_df.to_csv(os.path.join(output_dir, "test.csv"), index=False)
    
    print(f"[+] Static splits frozen to {output_dir}")

if __name__ == "__main__":
    main()