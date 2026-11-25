import os
import re
import json
import shutil
import argparse
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, StratifiedShuffleSplit


GROUP_COLUMN_CANDIDATES = ["scene_id", "sequence_id", "capture_id", "group_id", "video_id", "session_id"]


def derive_rarest_mask_class(df: pd.DataFrame, ignore_index: int = 255) -> pd.Series:
    """
    Scans ground-truth segmentation masks and labels each frame by the globally rarest
    foreground class (> 0) present in that frame. Falls back to 'bg_only' if empty.
    """
    print("[*] Deriving per-image stratification labels from rarest foreground defect class in masks...")
    frame_classes = []
    global_frame_freq = {}

    for mask_path in df["mask_path"]:
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"[-] CRITICAL: Unable to read annotation mask at {mask_path}")
        unique_cls = [int(c) for c in np.unique(mask) if c != ignore_index]
        fg_cls = [c for c in unique_cls if c > 0]
        frame_classes.append(fg_cls)
        for c in fg_cls:
            global_frame_freq[c] = global_frame_freq.get(c, 0) + 1

    rarest_labels = []
    for fg_cls in frame_classes:
        if not fg_cls:
            rarest_labels.append("bg_only")
        else:
            rarest = min(fg_cls, key=lambda c: (global_frame_freq[c], c))
            rarest_labels.append(f"mask_cls_{rarest}")

    return pd.Series(rarest_labels, index=df.index)


def extract_groups(raw_df: pd.DataFrame, processed_df: pd.DataFrame,
                   group_col: str = None, group_regex: str = None):
    """Resolves scene/sequence group IDs from an explicit CSV column or filename regex."""
    resolved_col = group_col
    if resolved_col is None:
        for candidate in GROUP_COLUMN_CANDIDATES:
            if candidate in raw_df.columns:
                resolved_col = candidate
                break

    if resolved_col is not None:
        if resolved_col not in raw_df.columns:
            raise ValueError(f"[-] CRITICAL: Specified --group_col '{resolved_col}' not found in eval_dataset.csv")
        groups = raw_df[resolved_col].astype(str).str.strip()
        groups.index = processed_df.index
        print(f"[*] Grouping val/test split by CSV column '{resolved_col}' ({groups.nunique()} unique groups).")
        return groups, f"column:{resolved_col}"

    if group_regex:
        pattern = re.compile(group_regex)
        extracted = []
        unmatched = 0
        for rgb_path in processed_df["rgb_path"]:
            stem = os.path.splitext(os.path.basename(rgb_path))[0]
            match = pattern.match(stem)
            if match and match.groups():
                extracted.append(match.group(1))
            else:
                extracted.append(stem)
                unmatched += 1

        if unmatched > 0:
            pct = 100.0 * unmatched / max(1, len(extracted))
            print(f"[!] WARNING: --group_regex '{group_regex}' failed to match {unmatched}/{len(extracted)} "
                  f"({pct:.1f}%) filenames; unmatched files fall back to singleton stem groups.")
        groups = pd.Series(extracted, index=processed_df.index)
        print(f"[*] Grouping val/test split by regex '{group_regex}' ({groups.nunique()} unique groups).")
        return groups, f"regex:{group_regex}"

    return None, None


def stratified_half_split(df: pd.DataFrame, stratify_labels: pd.Series,
                          groups: pd.Series = None, seed: int = 42):
    """
    Performs a deterministic 50/50 split of df into (val_df, test_df).
    - Uses StratifiedGroupKFold(n_splits=2) when scene/sequence groups exist (>= 2 groups).
      With n_splits=2, folds 0 and 1 are complementary group-disjoint partitions determined by `seed`.
    - Uses StratifiedShuffleSplit(test_size=0.5) with deterministic singleton handling otherwise.
    """
    n_samples = len(df)
    if n_samples < 2:
        raise ValueError(f"[-] CRITICAL: Need at least 2 evaluation samples to create val/test splits (got {n_samples}).")

    if groups is not None and groups.nunique() >= 2 and groups.nunique() < n_samples:
        sgkf = StratifiedGroupKFold(n_splits=2, shuffle=True, random_state=seed)
        val_idx, test_idx = next(sgkf.split(df, stratify_labels, groups))
        val_df = df.iloc[val_idx].copy()
        test_df = df.iloc[test_idx].copy()
        val_groups = set(groups.iloc[val_idx])
        test_groups = set(groups.iloc[test_idx])
        assert val_groups.isdisjoint(test_groups), "[-] FATAL: Group leakage detected between val and test!"
        return val_df, test_df, "StratifiedGroupKFold(n_splits=2)"

    if groups is not None and groups.nunique() == n_samples:
        print("[!] NOTICE: Every sample has a unique group ID; falling back to pure class stratification.")

    counts = stratify_labels.value_counts()
    singleton_classes = set(counts[counts < 2].index)

    multi_mask = ~stratify_labels.isin(singleton_classes)
    df_multi = df[multi_mask]
    labels_multi = stratify_labels[multi_mask]

    val_indices, test_indices = [], []

    if len(df_multi) >= 2 and labels_multi.nunique() >= 1:
        if labels_multi.nunique() == 1:
            shuffled = df_multi.sample(frac=1.0, random_state=seed)
            mid = len(shuffled) // 2
            val_indices.extend(shuffled.index[:mid].tolist())
            test_indices.extend(shuffled.index[mid:].tolist())
            strategy = "DeterministicShuffle(single_class)"
        else:
            sss = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=seed)
            rel_val, rel_test = next(sss.split(df_multi, labels_multi))
            val_indices.extend(df_multi.index[rel_val].tolist())
            test_indices.extend(df_multi.index[rel_test].tolist())
            strategy = "StratifiedShuffleSplit(test_size=0.5)"
    else:
        strategy = "SingletonAlternation"

    df_single = df[~multi_mask]
    if len(df_single) > 0:
        print(f"[!] WARNING: {len(df_single)} sample(s) belong to singleton strata ({sorted(singleton_classes)}). "
              f"Alternating deterministically between val and test.")
        rng = np.random.default_rng(seed)
        shuffled_single_idx = rng.permutation(df_single.index.to_numpy()).tolist()
        for idx in shuffled_single_idx:
            if len(val_indices) <= len(test_indices):
                val_indices.append(idx)
            else:
                test_indices.append(idx)

    val_df = df.loc[sorted(val_indices)].copy()
    test_df = df.loc[sorted(test_indices)].copy()
    return val_df, test_df, strategy


def adapt_canonical_splits(data_root: str, dataset_name: str, seed: int = 42,
                           group_col: str = None, group_regex: str = None,
                           stratify_by_mask: bool = False, verify_files: bool = True,
                           allow_canonical_overlap: bool = False):
    canonical_train_path = os.path.join(data_root, "train_dataset.csv")
    canonical_eval_path = os.path.join(data_root, "eval_dataset.csv")

    if not (os.path.exists(canonical_train_path) and os.path.exists(canonical_eval_path)):
        raise FileNotFoundError(
            f"[-] CRITICAL: Canonical benchmark splits (`train_dataset.csv`, `eval_dataset.csv`) "
            f"not found in root dataset directory: {data_root}"
        )

    subdirs = {
        "rgb": os.path.join(data_root, "RGB"),
        "depth": os.path.join(data_root, "Depth"),
        "thermal": os.path.join(data_root, "Thermal"),
        "mask": os.path.join(data_root, "Class_Annotations")
    }
    for path in subdirs.values():
        if not os.path.exists(path):
            raise FileNotFoundError(f"[-] CRITICAL: Required subdirectory missing: {path}")

    print("[*] Detected official canonical splits and strict modality subdirectories.")

    train_df_raw = pd.read_csv(canonical_train_path)
    eval_df_raw = pd.read_csv(canonical_eval_path)

    def process_split(df):
        records = []
        for _, row in df.iterrows():
            filename = str(row.get('filename', row.iloc[0])).strip()
            if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
                filename = f"{filename}.png"

            if 'class_label' in row and pd.notna(row['class_label']):
                class_label = str(row['class_label']).strip()
            elif len(row) > 1 and pd.notna(row.iloc[1]):
                class_label = str(row.iloc[1]).strip()
            else:
                class_label = 'unknown'

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

    if verify_files:
        for split_name, split_df in [("train", train_df), ("eval", processed_eval_df)]:
            for col in ["rgb_path", "depth_path", "thermal_path", "mask_path"]:
                missing = [p for p in split_df[col] if not os.path.exists(p)]
                if missing:
                    raise FileNotFoundError(
                        f"[-] CRITICAL: {len(missing)} missing files in {split_name} ({col}), e.g. {missing[0]}"
                    )

    unique_csv_labels = processed_eval_df["class_label"].nunique()
    if stratify_by_mask or unique_csv_labels <= 1:
        if not stratify_by_mask:
            print("[*] `class_label` in eval_dataset.csv has <= 1 unique value; auto-enabling mask rare-class stratification.")
        stratify_labels = derive_rarest_mask_class(processed_eval_df)
        stratify_source = "mask_rarest_foreground_class"
    else:
        stratify_labels = processed_eval_df["class_label"]
        stratify_source = "csv_class_label"

    groups, group_source = extract_groups(eval_df_raw, processed_eval_df, group_col=group_col, group_regex=group_regex)
    if groups is not None:
        processed_eval_df["group_id"] = groups

    val_df, test_df, split_strategy = stratified_half_split(
        processed_eval_df, stratify_labels=stratify_labels, groups=groups, seed=seed
    )

    train_files = set(train_df["rgb_path"])
    val_files = set(val_df["rgb_path"])
    test_files = set(test_df["rgb_path"])
    if not val_files.isdisjoint(test_files):
        raise RuntimeError("[-] FATAL: Overlap detected between val.csv and test.csv!")

    overlap_train_eval = sorted(train_files & (val_files | test_files))
    if overlap_train_eval and not allow_canonical_overlap:
        raise RuntimeError(
            f"[-] FATAL DATA LEAKAGE: {len(overlap_train_eval)} sample(s) appear in BOTH "
            f"train_dataset.csv and eval_dataset.csv (e.g., {overlap_train_eval[0]}). "
            f"Deduplicate canonical splits or pass --allow_canonical_overlap explicitly."
        )

    val_counts = stratify_labels.loc[val_df.index].value_counts().to_dict()
    test_counts = stratify_labels.loc[test_df.index].value_counts().to_dict()
    all_strata = sorted(set(val_counts.keys()) | set(test_counts.keys()))

    print("\n" + "=" * 68)
    print(f"📊 STRATIFIED 50/50 SPLIT AUDIT ({split_strategy} | {stratify_source})")
    print("=" * 68)
    print(f"{'Stratum':<28} | {'Val':<8} | {'Test':<8} | {'Val %'}")
    print("-" * 68)
    strata_summary = {}
    for s in all_strata:
        v_c = int(val_counts.get(s, 0))
        t_c = int(test_counts.get(s, 0))
        pct = (100.0 * v_c / max(1, v_c + t_c))
        print(f"{str(s):<28} | {v_c:<8} | {t_c:<8} | {pct:5.1f}%")
        strata_summary[str(s)] = {"val": v_c, "test": t_c}
        if v_c == 0 or t_c == 0:
            print(f"    [!] WARNING: Stratum '{s}' is absent from {'val' if v_c == 0 else 'test'}.")
    print("=" * 68 + "\n")

    audit_report = {
        "dataset": dataset_name,
        "seed": seed,
        "split_strategy": split_strategy,
        "stratify_source": stratify_source,
        "group_source": group_source,
        "counts": {"train": len(train_df), "val": len(val_df), "test": len(test_df)},
        "val_test_group_overlap": 0 if groups is not None else None,
        "train_eval_overlap_count": len(overlap_train_eval),
        "leakage_detected": len(overlap_train_eval) > 0,
        "strata_distribution": strata_summary
    }

    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True), audit_report


def main():
    parser = argparse.ArgumentParser(description="Adapt and stratify canonical dataset splits into locked 3-way manifests.")
    parser.add_argument("--dataset", type=str, required=True, help="Name of the dataset (e.g., MM5)")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic random seed for val/test split")
    parser.add_argument("--group_col", type=str, default=None, help="Optional CSV column name for scene/sequence grouping")
    parser.add_argument("--group_regex", type=str, default=None, help="Optional regex with 1 capture group to extract sequence ID from filename")
    parser.add_argument("--stratify_by_mask", action="store_true", help="Force stratification by rarest foreground defect class in Class_Annotations masks")
    parser.add_argument("--skip_file_check", action="store_true", help="Skip disk existence validation for image paths")
    parser.add_argument("--allow_canonical_overlap", action="store_true", help="Allow canonical train/eval overlap with explicit audit flag instead of failing")
    args = parser.parse_args()

    data_root = os.path.join("dataset", args.dataset)
    if not os.path.exists(data_root):
        raise FileNotFoundError(
            f"[-] CRITICAL: Data directory not found at {data_root}. "
            f"Ensure metadata.json, classes.txt, and canonical CSVs are present."
        )

    output_dir = os.path.join("data", "splits", args.dataset)
    os.makedirs(output_dir, exist_ok=True)
    print(f"[*] Adapting stratified manifest for dataset: {args.dataset} from {data_root}")

    train_df, val_df, test_df, audit_report = adapt_canonical_splits(
        data_root=data_root,
        dataset_name=args.dataset,
        seed=args.seed,
        group_col=args.group_col,
        group_regex=args.group_regex,
        stratify_by_mask=args.stratify_by_mask,
        verify_files=not args.skip_file_check,
        allow_canonical_overlap=args.allow_canonical_overlap
    )

    train_df.to_csv(os.path.join(output_dir, "train.csv"), index=False)
    val_df.to_csv(os.path.join(output_dir, "val.csv"), index=False)
    test_df.to_csv(os.path.join(output_dir, "test.csv"), index=False)
    shutil.copyfile(os.path.join(data_root, "classes.txt"), os.path.join(output_dir, "classes.txt"))

    with open(os.path.join(output_dir, "split_audit.json"), "w") as f:
        json.dump(audit_report, f, indent=4)

    print(f"[+] Stratified canonical splits frozen to {output_dir}")
    print(f"    Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")


if __name__ == "__main__":
    main()