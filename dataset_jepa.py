import os
import cv2
import csv
import json
import math
import torch
import numpy as np
import pandas as pd
from PIL import Image
from typing import Tuple, List, Dict, Any
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

class PredictiveMultimodalDataset(Dataset):
    def __init__(self, dataset_name, split="train", splits_root="data/splits", transform=None):
        """
        Universal, agnostic dataloader for the MM-JEPA architecture.
        Routes explicitly based on the --dataset flag paradigm.
        """
        self.dataset_name = dataset_name
        self.split = split
        self.transform = transform
        
        # Route to the dataset-specific locked CSV directory
        csv_path = os.path.join(splits_root, dataset_name, f"{split}.csv")
        
        if not os.path.exists(csv_path):
            raise FileNotFoundError(
                f"[-] CRITICAL: Reproducibility artifact missing. "
                f"No static split found for dataset '{dataset_name}' at {csv_path}. "
                f"Run `python freeze_splits.py --dataset {dataset_name}` first."
            )
            
        self.manifest = pd.read_csv(csv_path)
        print(f"[*] Instantiated agnostic loader for '{dataset_name}' [{split}]: {len(self.manifest)} samples.")

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, idx):
        row = self.manifest.iloc[idx]
        
        # Agnostic loading: Paths are explicitly provided by the dataframe
        rgb = Image.open(row['rgb_path']).convert('RGB')
        depth = Image.open(row['depth_path']).convert('L')
        thermal = Image.open(row['thermal_path']).convert('L')
        
        # Package for the ModalityIsolatedPatchEmbed stem
        sample = {
            'rgb': rgb,
            'depth': depth,
            'thermal': thermal,
            'class_label': row['class_label']
        }
        
        if self.transform:
            sample = self.transform(sample)
            
        return sample
    
class BaseRGBDTDataset(Dataset):
    """
    Abstract base class for RGB-Depth-Thermal datasets.
    Centralizes robust file I/O, metadata parsing, statistical caching, and exception handling.
    Supports both legacy `data_dir` and agnostic `dataset_name` parameters.
    """
    def __init__(self, data_dir: str = None, dataset_name: str = None, split: str = 'train', splits_root: str = "data/splits", image_size: Tuple[int, int] = (480, 640)):
        if data_dir is None and dataset_name is not None:
            self.data_dir = os.path.join("dataset", dataset_name)
        elif data_dir is not None:
            self.data_dir = data_dir
        else:
            raise ValueError("[!] CRITICAL: Either 'data_dir' or 'dataset_name' must be provided.")
            
        self.split = split
        self.image_size = image_size
        
        # --- Eradicate Magic Numbers via Metadata Manifest ---
        self.metadata = self._load_metadata()
        self.depth_scale = float(self.metadata.get('depth_max_mm', 10000.0))
        self.therm_scale = float(self.metadata.get('thermal_max_raw', 65535.0))

        csv_path = os.path.join(self.data_dir, f'{split}_dataset.csv')
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"[!] CRITICAL: Dataset split manifest missing at {csv_path}")
            
        self.image_files: List[str] = []
        with open(csv_path, 'r', encoding='utf-8-sig') as f:
            reader = csv.reader(f)
            next(reader) 
            for row in reader:
                if row: 
                    self.image_files.append(row[0].replace('_rgb.png', '').replace('.png', '').replace('.jpg', ''))
                    
        self.mean, self.std = self._get_or_compute_stats()

    def __len__(self) -> int: 
        return len(self.image_files)
        
    def _load_metadata(self) -> Dict[str, Any]:
        """Loads physical sensor limits from dataset configuration."""
        meta_path = os.path.join(self.data_dir, 'metadata.json')
        if os.path.exists(meta_path):
            with open(meta_path, 'r') as f:
                return json.load(f)
        print("[!] WARNING: metadata.json not found. Falling back to default sensor scales.")
        return {}

    def _safe_imread(self, path: str, flags: int) -> np.ndarray:
        """Exception handling for missing or corrupted physical files."""
        img = cv2.imread(path, flags)
        if img is None:
            raise ValueError(f"[!] I/O Error: Unable to read image or file corrupted at {path}")
        return img
        
    def _load_multimodal_tensors(self, base_name: str) -> torch.Tensor:
        """Centralized image loading and basic pre-processing."""
        rgb = cv2.cvtColor(self._safe_imread(os.path.join(self.data_dir, 'RGB', f"{base_name}.png"), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        depth = self._safe_imread(os.path.join(self.data_dir, 'Depth', f"{base_name}.png"), cv2.IMREAD_ANYDEPTH)
        therm = self._safe_imread(os.path.join(self.data_dir, 'Thermal', f"{base_name}.png"), cv2.IMREAD_ANYDEPTH)
        
        rgb = cv2.resize(rgb, (self.image_size[1], self.image_size[0]))
        depth = cv2.resize(depth, (self.image_size[1], self.image_size[0]), interpolation=cv2.INTER_NEAREST)
        therm = cv2.resize(therm, (self.image_size[1], self.image_size[0]), interpolation=cv2.INTER_NEAREST)
        
        rgb_t = TF.to_tensor(rgb)
        depth_t = torch.from_numpy(depth.astype(np.float32)).unsqueeze(0) / self.depth_scale
        therm_t = torch.from_numpy(therm.astype(np.float32)).unsqueeze(0) / self.therm_scale
        
        return torch.cat([rgb_t, depth_t, therm_t], dim=0)

    def _get_or_compute_stats(self) -> Tuple[List[float], List[float]]:
        """Safely computes and caches empirical mean and std using float64 precision."""
        cache_path = os.path.join(self.data_dir, 'dataset_stats.json')
        
        if os.path.exists(cache_path):
            with open(cache_path, 'r') as f:
                stats = json.load(f)
            return stats['mean'], stats['std']
            
        if self.split != 'train':
            raise RuntimeError(f"[!] CRITICAL: Missing '{cache_path}'. Initialize 'train' split first.")
            
        print("\n[*] Initializing Cache. Computing precise dataset statistics...")
        channels_sum = torch.zeros(5, dtype=torch.float64)
        channels_sq_sum = torch.zeros(5, dtype=torch.float64)
        num_pixels = 0

        for base_name in self.image_files:
            x_5ch = self._load_multimodal_tensors(base_name).to(torch.float64)
            channels_sum += x_5ch.sum(dim=(1, 2))
            channels_sq_sum += (x_5ch ** 2).sum(dim=(1, 2))
            num_pixels += (self.image_size[0] * self.image_size[1])

        mean = channels_sum / num_pixels
        variance = torch.clamp((channels_sq_sum / num_pixels) - (mean ** 2), min=1e-8)
        std = torch.sqrt(variance)
        
        mean_list, std_list = mean.tolist(), std.tolist()
        with open(cache_path, 'w') as f:
            json.dump({'mean': mean_list, 'std': std_list}, f, indent=4)
            
        return mean_list, std_list


class JEPAPretrainDataset(BaseRGBDTDataset):
    """Phase 1 Dataset utilizing isolated base components with dynamic mask routing."""
    
    def __init__(self, data_dir: str = None, dataset_name: str = None, split: str = 'train', splits_root: str = "data/splits", image_size: Tuple[int, int] = (480, 640), mask_strategy: str = "multi_block"):
        super().__init__(data_dir=data_dir, dataset_name=dataset_name, split=split, splits_root=splits_root, image_size=image_size)
        self.mask_strategy = mask_strategy

    def _generate_multiblock_mask(self, h: int, w: int, num_blocks: int = 4) -> torch.Tensor:
        mask = torch.zeros((1, h, w), dtype=torch.float32)
        for _ in range(num_blocks):
            scale = np.random.uniform(0.15, 0.20)
            aspect_ratio = np.random.uniform(0.75, 1.5)
            target_area = scale * h * w
            
            exact_w, exact_h = math.sqrt(target_area * aspect_ratio), math.sqrt(target_area / aspect_ratio)
            w_f, h_f = max(1, int(math.floor(exact_w))), max(1, int(math.floor(exact_h)))
            w_c, h_c = max(1, int(math.ceil(exact_w))), max(1, int(math.ceil(exact_h)))
            
            if abs((w_f * h_f) - target_area) <= abs((w_c * h_c) - target_area):
                block_w, block_h = w_f, h_f
            else:
                block_w, block_h = w_c, h_c
                
            block_w, block_h = min(block_w, w - 1), min(block_h, h - 1)
            top, left = np.random.randint(0, h - block_h + 1), np.random.randint(0, w - block_w + 1)
            mask[:, top:top + block_h, left:left + block_w] = 1.0
            
        return mask

    def _generate_random_patch_mask(self, h: int, w: int, patch_size: int = 16, mask_ratio: float = 0.60) -> torch.Tensor:
        """Ablation Variant: Drops completely random disjointed patches instead of semantic blocks."""
        mask = torch.zeros((1, h, w), dtype=torch.float32)
        num_patches_h, num_patches_w = h // patch_size, w // patch_size
        total_patches = num_patches_h * num_patches_w
        num_mask_patches = int(total_patches * mask_ratio)
        
        # Randomly select patches to mask
        mask_indices = np.random.choice(total_patches, num_mask_patches, replace=False)
        
        for idx in mask_indices:
            row = (idx // num_patches_w) * patch_size
            col = (idx % num_patches_w) * patch_size
            mask[:, row:row+patch_size, col:col+patch_size] = 1.0
            
        return mask

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        base_name = self.image_files[idx]
        x_full = self._load_multimodal_tensors(base_name)
        
        x_normalized = TF.normalize(x_full, mean=self.mean, std=self.std)
        
        # Route the mask topology based on the orchestrator's injected state
        if self.mask_strategy == "random":
            mask = self._generate_random_patch_mask(self.image_size[0], self.image_size[1])
        else:
            mask = self._generate_multiblock_mask(self.image_size[0], self.image_size[1])
            
        x_visible = x_normalized * (1.0 - mask)
        
        return {'x_full': x_normalized, 'x_visible': x_visible, 'mask': mask}


class DownstreamSegmentationDataset(BaseRGBDTDataset):
    """Phase 2 Dataset utilizing isolated base components."""
    
    def __init__(self, data_dir: str = None, dataset_name: str = None, split: str = 'train', splits_root: str = "data/splits", image_size: Tuple[int, int] = (480, 640)):
        super().__init__(data_dir=data_dir, dataset_name=dataset_name, split=split, splits_root=splits_root, image_size=image_size)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        base_name = self.image_files[idx]
        x_full = self._load_multimodal_tensors(base_name)
        
        # Safe GT Load
        gt_path = os.path.join(self.data_dir, 'Class_Annotations', f"{base_name}.png")
        gt_mask = self._safe_imread(gt_path, cv2.IMREAD_GRAYSCALE)
        gt_mask = cv2.resize(gt_mask, (self.image_size[1], self.image_size[0]), interpolation=cv2.INTER_NEAREST)
        
        x_normalized = TF.normalize(x_full, mean=self.mean, std=self.std)
        gt_t = torch.as_tensor(gt_mask, dtype=torch.long)
        gt_t[gt_t == 255] = 255 

        return {'x_full': x_normalized, 'seg_mask': gt_t}