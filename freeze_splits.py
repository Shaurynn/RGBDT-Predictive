import os
import argparse
import shutil
import pandas as pd

def adapt_canonical_splits(data_root, dataset_name, seed=42):
    """
    Reads the canonical dataset splits and maps them to explicit modality subdirectories,
    preserving the official benchmark distribution for fair baseline comparisons.
    Splits the canonical evaluation set strictly into 50% validation and 50% test sets.
    """
    canonical_train_path = os.path.join(data_root, "train_dataset.csv")
    canonical_eval_path = os.path.join(data_root, "eval_dataset.csv")
    
    if not (os.path.exists(canonical_train_path) and os.path.exists(canonical_eval_path)):
        raise FileNotFoundError(
            f"[-] CRITICAL: Canonical benchmark splits (`train_dataset.csv`, `eval_dataset.csv`) "
            f"not found in root dataset directory: {data_root}"
        )

    # Define the strict dataset subdirectory structure
    subdirs = {
        "rgb": os.path.join(data_root, "RGB"),
        "depth": os.path.join(data_root, "Depth"),
        "thermal": os.path.join(data_root, "Thermal"),
        "mask": os.path.join(data_root, "Class_Annotations")
    }
    
    # Verify strict folder structure exists
    for name, path in subdirs.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"[-] CRITICAL: Required subdirectory missing: {path}")

    print("[*] Detected official canonical splits and strict modality subdirectories.")
    
    # Read the canonical splits. 
    train_df_raw = pd.read_csv(canonical_train_path)
    eval_df_raw = pd.read_csv(canonical_eval_path)
    
    def process_split(df):
        records = []
        for _, row in df.iterrows():
            # EXPLICIT FIX: Force string casting to override Pandas numeric inference
            filename = str(row.get('filename', row.iloc[0])).strip()
            
            # ENTERPRISE GUARD: Ensure filename contains a valid extension, appending .png if omitted
            if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
                filename = f"{filename}.png"
            
            # Safely extract class label (fallback to second column if 'class_label' header is missing)
            if 'class_label' in row:
                class_label = str(row['class_label'])
            else:
                class_label = str(row.iloc[1]) if len(row) > 1 else 'unknown'
            
            records.append({
                "dataset": dataset_name,
                "class_label": class_label,
                "rgb_path": os.path.join(subdirs["rgb"], filename),
                "depth_path": os.path.join(subdirs["depth"], filename),
                "thermal_path": os.path.join(subdirs["thermal"], filename),
                "mask_path": os.path.join(subdirs["mask"], filename)
            })
        return pd.DataFrame(records)

    train_df = process_split(train_df_raw)
    processed_eval_df = process_split(eval_df_raw)
    
    # Programmatically split the evaluation dataset 50/50 using a locked seed
    val_df = processed_eval_df.sample(frac=0.5, random_state=seed)
    test_df = processed_eval_df.drop(val_df.index)

    return train_df, val_df, test_df

def main():
    parser = argparse.ArgumentParser(description="Adapt canonical dataset splits into the agnostic framework.")
    parser.add_argument("--dataset", type=str, required=True, help="Name of the dataset (e.g., MM5)")
    args = parser.parse_args()

    data_root = os.path.join("dataset", args.dataset)
    
    if not os.path.exists(data_root):
        raise FileNotFoundError(
            f"[-] CRITICAL: Data directory not found at {data_root}. "
            f"Ensure metadata.json, classes.txt, and canonical CSVs are present."
        )

    output_dir = os.path.join("data", "splits", args.dataset)
    os.makedirs(output_dir, exist_ok=True)
    print(f"[*] Adapting agnostic manifest for dataset: {args.dataset} from {data_root}")

    # Process and preserve the canonical splits with a strict 3-way split
    train_df, val_df, test_df = adapt_canonical_splits(data_root, args.dataset, seed=42)
    
    # Serialize to the locked routing directory
    train_df.to_csv(os.path.join(output_dir, "train.csv"), index=False)
    val_df.to_csv(os.path.join(output_dir, "val.csv"), index=False)
    test_df.to_csv(os.path.join(output_dir, "test.csv"), index=False)
    shutil.copyfile(os.path.join(data_root, "classes.txt"), os.path.join(output_dir, "classes.txt"))
    
    print(f"[+] Canonical splits adapted and frozen to {output_dir}")
    print(f"    Train: {len(train_df)} samples")
    print(f"    Val:   {len(val_df)} samples")
    print(f"    Test:  {len(test_df)} samples")

if __name__ == "__main__":
    main()