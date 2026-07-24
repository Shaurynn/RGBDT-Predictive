import os
import cv2
import csv
import json
import math
import random
import torch
import numpy as np
import pandas as pd
from PIL import Image
from typing import Tuple, List, Dict, Any
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

from torchvision.transforms import InterpolationMode

class PredictiveMultimodalDataset(Dataset):
    def __init__(self, dataset_name, split="train", splits_root="data/splits", transform=None):
        self.dataset_name = dataset_name
        self.split = split
        self.transform = transform
        
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
        
        rgb = Image.open(row['rgb_path']).convert('RGB')
        depth = Image.open(row['depth_path']).convert('L')
        thermal = Image.open(row['thermal_path']).convert('L')
        
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
    def __init__(self, data_dir: str = None, dataset_name: str = None, split: str = 'train', splits_root: str = "data/splits", image_size: Tuple[int, int] = (480, 640)):
        if dataset_name is not None:
            self.dataset_name = dataset_name
            self.data_dir = os.path.join("dataset", dataset_name)
        elif data_dir is not None:
            self.data_dir = data_dir
            self.dataset_name = os.path.basename(data_dir.rstrip('/\\'))
        else:
            raise ValueError("[!] CRITICAL: Either 'data_dir' or 'dataset_name' must be provided.")
            
        self.split = split
        self.image_size = image_size
        self.splits_root = splits_root
        
        self.metadata = self._load_metadata()
        self.depth_scale = float(self.metadata.get('depth_max_mm', 10000.0))
        self.therm_scale = float(self.metadata.get('thermal_max_raw', 65535.0))

        csv_path = os.path.join(self.splits_root, self.dataset_name, f'{split}.csv')
        if not os.path.exists(csv_path):
            raise FileNotFoundError(
                f"[!] CRITICAL: Dataset split manifest missing at {csv_path}. "
                f"Run `python freeze_splits.py --dataset {self.dataset_name}` first."
            )
            
        self.manifest = pd.read_csv(csv_path)
        self.image_files: List[str] = []
        for _, row in self.manifest.iterrows():
            if 'rgb_path' in row:
                rgb_filename = os.path.basename(str(row['rgb_path']))
                base_name = rgb_filename.replace('_rgb.png', '').replace('.png', '').replace('.jpg', '')
            else:
                base_name = str(row.iloc[0])
            self.image_files.append(base_name)
                    
        self.mean, self.std = self._get_or_compute_stats()

    def __len__(self) -> int: 
        return len(self.manifest)
        
    def _load_metadata(self) -> Dict[str, Any]:
        meta_path = os.path.join(self.data_dir, 'metadata.json')
        if os.path.exists(meta_path):
            with open(meta_path, 'r') as f:
                return json.load(f)
        print("[!] WARNING: metadata.json not found. Falling back to default sensor scales.")
        return {}

    def _safe_imread(self, path: str, flags: int) -> np.ndarray:
        img = cv2.imread(path, flags)
        if img is None:
            raise ValueError(f"[!] I/O Error: Unable to read image or file corrupted at {path}")
        return img
        
    def _load_multimodal_tensors(self, idx: int, hflip: bool = False) -> torch.Tensor:
        row = self.manifest.iloc[idx]
        base_name = self.image_files[idx]
        
        rgb_path = str(row['rgb_path']) if 'rgb_path' in row and pd.notna(row['rgb_path']) else os.path.join(self.data_dir, 'RGB', f"{base_name}.png")
        depth_path = str(row['depth_path']) if 'depth_path' in row and pd.notna(row['depth_path']) else os.path.join(self.data_dir, 'Depth', f"{base_name}.png")
        therm_path = str(row['thermal_path']) if 'thermal_path' in row and pd.notna(row['thermal_path']) else os.path.join(self.data_dir, 'Thermal', f"{base_name}.png")
        
        rgb = cv2.cvtColor(self._safe_imread(rgb_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        depth = self._safe_imread(depth_path, cv2.IMREAD_ANYDEPTH)
        therm = self._safe_imread(therm_path, cv2.IMREAD_ANYDEPTH)
        
        if hflip:
            rgb = cv2.flip(rgb, 1)
            depth = cv2.flip(depth, 1)
            therm = cv2.flip(therm, 1)
            
        rgb = cv2.resize(rgb, (self.image_size[1], self.image_size[0]))
        depth = cv2.resize(depth, (self.image_size[1], self.image_size[0]), interpolation=cv2.INTER_NEAREST)
        therm = cv2.resize(therm, (self.image_size[1], self.image_size[0]), interpolation=cv2.INTER_NEAREST)
        
        rgb_t = TF.to_tensor(rgb)
        depth_t = torch.from_numpy(depth.astype(np.float32)).unsqueeze(0) / self.depth_scale
        therm_t = torch.from_numpy(therm.astype(np.float32)).unsqueeze(0) / self.therm_scale
        
        return torch.cat([rgb_t, depth_t, therm_t], dim=0)

    def _get_or_compute_stats(self) -> Tuple[List[float], List[float]]:
        cache_path = os.path.join(self.data_dir, 'dataset_stats.pt')
        
        if os.path.exists(cache_path):
            stats = torch.load(cache_path, weights_only=True)
            return stats['mean'].tolist(), stats['std'].tolist()
            
        if self.split != 'train':
            raise RuntimeError(f"[!] CRITICAL: Missing '{cache_path}'. Initialize 'train' split first.")
            
        print("\n[*] Initializing Cache. Computing precise dataset statistics...")
        channels_sum = torch.zeros(5, dtype=torch.float64)
        channels_sq_sum = torch.zeros(5, dtype=torch.float64)
        num_pixels = 0

        for idx in range(len(self.manifest)):
            x_5ch = self._load_multimodal_tensors(idx, hflip=False).to(torch.float64)
            channels_sum += x_5ch.sum(dim=(1, 2))
            channels_sq_sum += (x_5ch ** 2).sum(dim=(1, 2))
            num_pixels += (self.image_size[0] * self.image_size[1])
            
            del x_5ch 

        mean = channels_sum / num_pixels
        variance = torch.clamp((channels_sq_sum / num_pixels) - (mean ** 2), min=1e-8)
        std = torch.sqrt(variance)
        
        torch.save({'mean': mean, 'std': std}, cache_path)
            
        return mean.tolist(), std.tolist()


class JEPAPretrainDataset(BaseRGBDTDataset):
    def __init__(self, data_dir: str = None, dataset_name: str = None, split: str = 'train', splits_root: str = "data/splits", image_size: Tuple[int, int] = (480, 640), mask_strategy: str = "multi_block", enable_augmentation: bool = True, modality_dropout_prob: float = 0.0):
        super().__init__(data_dir=data_dir, dataset_name=dataset_name, split=split, splits_root=splits_root, image_size=image_size)
        self.mask_strategy = mask_strategy
        self.enable_augmentation = enable_augmentation
        self.modality_dropout_prob = modality_dropout_prob

    # [ALGORITHM COMMENT: Multi-Block Masking Engine]
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

    # [ALGORITHM COMMENT: High-Sparsity Random Patch Masking]
    def _generate_random_patch_mask(self, h: int, w: int, patch_size: int = 16, mask_ratio: float = 0.60) -> torch.Tensor:
        mask = torch.zeros((1, h, w), dtype=torch.float32)
        num_patches_h, num_patches_w = h // patch_size, w // patch_size
        total_patches = num_patches_h * num_patches_w
        num_mask_patches = int(total_patches * mask_ratio)
        
        mask_indices = np.random.choice(total_patches, num_mask_patches, replace=False)
        
        for idx in mask_indices:
            row = (idx // num_patches_w) * patch_size
            col = (idx % num_patches_w) * patch_size
            mask[:, row:row+patch_size, col:col+patch_size] = 1.0
            
        return mask

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        hflip = False
        rotation_angle = 0.0
        
        if self.enable_augmentation and self.split == 'train':
            hflip = random.random() > 0.5
            
            if random.random() > 0.5:
                rotation_angle = random.uniform(-15.0, 15.0)
            
        x_full = self._load_multimodal_tensors(idx, hflip=hflip)
        
        if rotation_angle != 0.0:
            rgb_rot = TF.rotate(x_full[:3], rotation_angle, interpolation=InterpolationMode.BILINEAR)
            dt_rot = TF.rotate(x_full[3:], rotation_angle, interpolation=InterpolationMode.NEAREST)
            x_full = torch.cat([rgb_rot, dt_rot], dim=0)
        
        if self.enable_augmentation and self.split == 'train':
            if random.random() > 0.5:
                rgb = x_full[:3, :, :]
                rgb = TF.adjust_brightness(rgb, brightness_factor=1.0 + (random.random() - 0.5) * 0.4)
                rgb = TF.adjust_contrast(rgb, contrast_factor=1.0 + (random.random() - 0.5) * 0.4)
                rgb = TF.adjust_saturation(rgb, saturation_factor=1.0 + (random.random() - 0.5) * 0.4)
                rgb = TF.adjust_hue(rgb, hue_factor=(random.random() - 0.5) * 0.2)
                x_full[:3, :, :] = rgb
        
        x_normalized = TF.normalize(x_full, mean=self.mean, std=self.std)

        # INJECTED: Modality Dropout Regularizer
        # Evaluated strictly post-normalization to prevent distribution shift corruption.
        if self.split == 'train' and self.modality_dropout_prob > 0.0:
            if random.random() < self.modality_dropout_prob:
                x_normalized[3, :, :] = 0.0  # Drop Depth Modal Tensor
            if random.random() < self.modality_dropout_prob:
                x_normalized[4, :, :] = 0.0  # Drop Thermal Modal Tensor
        
        if self.mask_strategy == "random":
            mask = self._generate_random_patch_mask(self.image_size[0], self.image_size[1])
        else:
            mask = self._generate_multiblock_mask(self.image_size[0], self.image_size[1])
            
        x_visible = x_normalized * (1.0 - mask)
        
        return {'x_full': x_normalized, 'x_visible': x_visible, 'mask': mask}


class DownstreamSegmentationDataset(BaseRGBDTDataset):
    def __init__(self, data_dir: str = None, dataset_name: str = None, split: str = 'train', splits_root: str = "data/splits", image_size: Tuple[int, int] = (480, 640), enable_augmentation: bool = True, modality_dropout_prob: float = 0.0):
        super().__init__(data_dir=data_dir, dataset_name=dataset_name, split=split, splits_root=splits_root, image_size=image_size)
        self.enable_augmentation = enable_augmentation
        self.modality_dropout_prob = modality_dropout_prob

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        hflip = False
        rotation_angle = 0.0
        
        if self.enable_augmentation and self.split == 'train':
            hflip = random.random() > 0.5
            
            if random.random() > 0.5:
                rotation_angle = random.uniform(-15.0, 15.0)

        x_full = self._load_multimodal_tensors(idx, hflip=hflip)
        
        row = self.manifest.iloc[idx]
        base_name = self.image_files[idx]
        gt_path = str(row['mask_path']) if 'mask_path' in row and pd.notna(row['mask_path']) else os.path.join(self.data_dir, 'Class_Annotations', f"{base_name}.png")
        gt_mask = self._safe_imread(gt_path, cv2.IMREAD_GRAYSCALE)
        
        if hflip:
            gt_mask = cv2.flip(gt_mask, 1)
            
        gt_mask = cv2.resize(gt_mask, (self.image_size[1], self.image_size[0]), interpolation=cv2.INTER_NEAREST)
        gt_t = torch.as_tensor(gt_mask, dtype=torch.long)
        
        if rotation_angle != 0.0:
            rgb_rot = TF.rotate(x_full[:3], rotation_angle, interpolation=InterpolationMode.BILINEAR)
            dt_rot = TF.rotate(x_full[3:], rotation_angle, interpolation=InterpolationMode.NEAREST)
            x_full = torch.cat([rgb_rot, dt_rot], dim=0)
            
            gt_t = TF.rotate(gt_t.unsqueeze(0).float(), rotation_angle, interpolation=InterpolationMode.NEAREST, fill=255.0).squeeze(0).long()
        
        if self.enable_augmentation and self.split == 'train':
            if random.random() > 0.5:
                rgb = x_full[:3, :, :]
                rgb = TF.adjust_brightness(rgb, brightness_factor=1.0 + (random.random() - 0.5) * 0.4)
                rgb = TF.adjust_contrast(rgb, contrast_factor=1.0 + (random.random() - 0.5) * 0.4)
                rgb = TF.adjust_saturation(rgb, saturation_factor=1.0 + (random.random() - 0.5) * 0.4)
                rgb = TF.adjust_hue(rgb, hue_factor=(random.random() - 0.5) * 0.2)
                x_full[:3, :, :] = rgb
        
        x_normalized = TF.normalize(x_full, mean=self.mean, std=self.std)

        # INJECTED: Modality Dropout Regularizer
        # Evaluated strictly post-normalization to prevent distribution shift corruption.
        if self.split == 'train' and self.modality_dropout_prob > 0.0:
            if random.random() < self.modality_dropout_prob:
                x_normalized[3, :, :] = 0.0  # Drop Depth Modal Tensor
            if random.random() < self.modality_dropout_prob:
                x_normalized[4, :, :] = 0.0  # Drop Thermal Modal Tensor
                
        gt_t[gt_t == 255] = 255 

        return {'x_full': x_normalized, 'seg_mask': gt_t}


class TTA_DownstreamSegmentationDataset(BaseRGBDTDataset):
    def __init__(self, data_dir: str = None, dataset_name: str = None, split: str = 'test', splits_root: str = "data/splits", image_size: Tuple[int, int] = (480, 640)):
        super().__init__(data_dir=data_dir, dataset_name=dataset_name, split=split, splits_root=splits_root, image_size=image_size)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.manifest.iloc[idx]
        base_name = self.image_files[idx]
        gt_path = str(row['mask_path']) if 'mask_path' in row and pd.notna(row['mask_path']) else os.path.join(self.data_dir, 'Class_Annotations', f"{base_name}.png")
        gt_mask = self._safe_imread(gt_path, cv2.IMREAD_GRAYSCALE)
        gt_mask = cv2.resize(gt_mask, (self.image_size[1], self.image_size[0]), interpolation=cv2.INTER_NEAREST)
        gt_t = torch.as_tensor(gt_mask, dtype=torch.long)
        gt_t[gt_t == 255] = 255

        x_base = self._load_multimodal_tensors(idx, hflip=False)
        x_base_norm = TF.normalize(x_base, mean=self.mean, std=self.std)

        x_hflip = self._load_multimodal_tensors(idx, hflip=True)
        x_hflip_norm = TF.normalize(x_hflip, mean=self.mean, std=self.std)
        
        x_tta_batch = torch.stack([x_base_norm, x_hflip_norm], dim=0)

        return {'x_tta_batch': x_tta_batch, 'seg_mask': gt_t}