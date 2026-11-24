import os
import json
import math
import hashlib
import random
import cv2
import torch
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Any
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode


def seed_worker(worker_id: int) -> None:
    """
    Seeds Python random and NumPy random generators deterministically inside
    each DataLoader worker process using the PyTorch worker initial seed.
    """
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class BaseRGBDTDataset(Dataset):
    def __init__(self, data_dir: str = None, dataset_name: str = None, split: str = 'train',
                 splits_root: str = "data/splits", image_size: Tuple[int, int] = (480, 640)):
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
        self._warned_depth_8bit = False
        self._warned_therm_8bit = False

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
            if 'rgb_path' in row and pd.notna(row['rgb_path']):
                rgb_filename = os.path.basename(str(row['rgb_path']))
                base_name = os.path.splitext(rgb_filename)[0].replace('_rgb', '')
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
        print("[!] WARNING: metadata.json not found. Falling back to default 16-bit sensor scales.")
        return {}

    def _safe_imread(self, path: str, flags: int) -> np.ndarray:
        img = cv2.imread(path, flags)
        if img is None:
            raise ValueError(f"[!] I/O Error: Unable to read image or file corrupted at {path}")
        return img

    def _load_multimodal_tensors(self, idx: int, hflip: bool = False) -> torch.Tensor:
        row = self.manifest.iloc[idx]
        base_name = self.image_files[idx]

        rgb_path = (
            str(row['rgb_path']) if 'rgb_path' in row and pd.notna(row['rgb_path'])
            else os.path.join(self.data_dir, 'RGB', f"{base_name}.png")
        )
        depth_path = (
            str(row['depth_path']) if 'depth_path' in row and pd.notna(row['depth_path'])
            else os.path.join(self.data_dir, 'Depth', f"{base_name}.png")
        )
        therm_path = (
            str(row['thermal_path']) if 'thermal_path' in row and pd.notna(row['thermal_path'])
            else os.path.join(self.data_dir, 'Thermal', f"{base_name}.png")
        )

        rgb = cv2.cvtColor(self._safe_imread(rgb_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        depth = self._safe_imread(depth_path, cv2.IMREAD_ANYDEPTH)
        therm = self._safe_imread(therm_path, cv2.IMREAD_ANYDEPTH)

        if depth.dtype == np.uint8 and self.depth_scale > 255.0 and not self._warned_depth_8bit:
            print(f"[!] WARNING: 8-bit Depth image detected ({depth_path}) with depth_scale={self.depth_scale}. Verify metadata.json.")
            self._warned_depth_8bit = True
        if therm.dtype == np.uint8 and self.therm_scale > 255.0 and not self._warned_therm_8bit:
            print(f"[!] WARNING: 8-bit Thermal image detected ({therm_path}) with therm_scale={self.therm_scale}. Verify metadata.json.")
            self._warned_therm_8bit = True

        if hflip:
            rgb = cv2.flip(rgb, 1)
            depth = cv2.flip(depth, 1)
            therm = cv2.flip(therm, 1)

        rgb = cv2.resize(rgb, (self.image_size[1], self.image_size[0]), interpolation=cv2.INTER_LINEAR)
        depth = cv2.resize(depth, (self.image_size[1], self.image_size[0]), interpolation=cv2.INTER_NEAREST)
        therm = cv2.resize(therm, (self.image_size[1], self.image_size[0]), interpolation=cv2.INTER_NEAREST)

        rgb_t = TF.to_tensor(rgb)
        depth_t = torch.from_numpy(depth.astype(np.float32)).unsqueeze(0) / self.depth_scale
        therm_t = torch.from_numpy(therm.astype(np.float32)).unsqueeze(0) / self.therm_scale

        return torch.cat([rgb_t, depth_t, therm_t], dim=0)

    def _normalize_rgb_only(self, x_full: torch.Tensor) -> torch.Tensor:
        """
        Normalizes only the photometric RGB channels (0, 1, 2) using global dataset statistics,
        preserving Depth (channel 3) and Thermal (channel 4) in their native [0, 1] physical domain.
        """
        x_norm = x_full.clone()
        x_norm[:3, :, :] = TF.normalize(x_full[:3, :, :], mean=self.mean[:3], std=self.std[:3])
        return x_norm

    def _get_stats_cache_path(self) -> str:
        train_csv = os.path.join(self.splits_root, self.dataset_name, 'train.csv')
        if os.path.exists(train_csv):
            with open(train_csv, 'rb') as f:
                manifest_hash = hashlib.sha256(f.read()).hexdigest()[:8]
        else:
            manifest_hash = "default"
        filename = f"dataset_stats_{self.image_size[0]}x{self.image_size[1]}_{manifest_hash}.pt"
        return os.path.join(self.splits_root, self.dataset_name, filename)

    def _get_or_compute_stats(self) -> Tuple[List[float], List[float]]:
        cache_path = self._get_stats_cache_path()

        if os.path.exists(cache_path):
            stats = torch.load(cache_path, weights_only=True)
            return stats['mean'].tolist(), stats['std'].tolist()

        if self.split != 'train':
            raise RuntimeError(
                f"[!] CRITICAL: Missing '{cache_path}'. Initialize the 'train' split first "
                f"to compute deterministic normalization statistics."
            )

        print(f"\n[*] Initializing Stats Cache ({os.path.basename(cache_path)}). Computing dataset statistics...")
        channels_sum = torch.zeros(5, dtype=torch.float64)
        channels_sq_sum = torch.zeros(5, dtype=torch.float64)
        num_pixels = 0

        for idx in range(len(self.manifest)):
            x_5ch = self._load_multimodal_tensors(idx, hflip=False).to(torch.float64)
            channels_sum += x_5ch.sum(dim=(1, 2))
            channels_sq_sum += (x_5ch ** 2).sum(dim=(1, 2))
            num_pixels += (self.image_size[0] * self.image_size[1])
            del x_5ch

        mean = channels_sum / max(1, num_pixels)
        variance = torch.clamp((channels_sq_sum / max(1, num_pixels)) - (mean ** 2), min=1e-8)
        std = torch.sqrt(variance)

        tmp_path = f"{cache_path}.tmp"
        torch.save({'mean': mean, 'std': std}, tmp_path)
        os.replace(tmp_path, cache_path)

        return mean.tolist(), std.tolist()


class JEPAPretrainDataset(BaseRGBDTDataset):
    def __init__(self, data_dir: str = None, dataset_name: str = None, split: str = 'train',
                 splits_root: str = "data/splits", image_size: Tuple[int, int] = (480, 640),
                 mask_strategy: str = "random", enable_augmentation: bool = True,
                 modality_dropout_prob: float = 0.0):
        super().__init__(
            data_dir=data_dir,
            dataset_name=dataset_name,
            split=split,
            splits_root=splits_root,
            image_size=image_size
        )
        self.mask_strategy = mask_strategy
        self.enable_augmentation = enable_augmentation
        self.modality_dropout_prob = modality_dropout_prob

    def _generate_multiblock_mask(self, h: int, w: int, num_blocks: int = 4) -> torch.Tensor:
        mask = torch.zeros((1, h, w), dtype=torch.float32)
        for _ in range(num_blocks):
            scale = np.random.uniform(0.15, 0.20)
            aspect_ratio = np.random.uniform(0.75, 1.5)
            target_area = scale * h * w

            exact_w = math.sqrt(target_area * aspect_ratio)
            exact_h = math.sqrt(target_area / aspect_ratio)
            w_f, h_f = max(1, int(math.floor(exact_w))), max(1, int(math.floor(exact_h)))
            w_c, h_c = max(1, int(math.ceil(exact_w))), max(1, int(math.ceil(exact_h)))

            if abs((w_f * h_f) - target_area) <= abs((w_c * h_c) - target_area):
                block_w, block_h = w_f, h_f
            else:
                block_w, block_h = w_c, h_c

            block_w, block_h = min(block_w, w - 1), min(block_h, h - 1)
            top = np.random.randint(0, h - block_h + 1)
            left = np.random.randint(0, w - block_w + 1)
            mask[:, top:top + block_h, left:left + block_w] = 1.0

        return mask

    def _generate_random_patch_mask(self, h: int, w: int, patch_size: int = 32,
                                    mask_ratio: float = 0.60) -> torch.Tensor:
        """
        Generates random non-overlapping patches aligned with the MiT Stage-4 (stride 32)
        latent grid so every masked pixel patch maps 1-to-1 to a target latent token.
        """
        mask = torch.zeros((1, h, w), dtype=torch.float32)
        num_patches_h, num_patches_w = h // patch_size, w // patch_size
        total_patches = num_patches_h * num_patches_w
        num_mask_patches = max(1, int(total_patches * mask_ratio))

        mask_indices = np.random.choice(total_patches, num_mask_patches, replace=False)

        for idx in mask_indices:
            row = (idx // num_patches_w) * patch_size
            col = (idx % num_patches_w) * patch_size
            mask[:, row:row + patch_size, col:col + patch_size] = 1.0

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

        x_normalized = self._normalize_rgb_only(x_full)

        if self.mask_strategy == "random":
            mask = self._generate_random_patch_mask(self.image_size[0], self.image_size[1], patch_size=32)
        else:
            mask = self._generate_multiblock_mask(self.image_size[0], self.image_size[1])

        x_visible = x_normalized * (1.0 - mask)

        # Apply Modality Dropout strictly to x_visible so the EMA Target Encoder receives clean x_full
        if self.split == 'train' and self.modality_dropout_prob > 0.0:
            if random.random() < self.modality_dropout_prob:
                x_visible[3, :, :] = 0.0
            if random.random() < self.modality_dropout_prob:
                x_visible[4, :, :] = 0.0

        return {'x_full': x_normalized, 'x_visible': x_visible, 'mask': mask}


class DownstreamSegmentationDataset(BaseRGBDTDataset):
    def __init__(self, data_dir: str = None, dataset_name: str = None, split: str = 'train',
                 splits_root: str = "data/splits", image_size: Tuple[int, int] = (480, 640),
                 enable_augmentation: bool = True, modality_dropout_prob: float = 0.0):
        super().__init__(
            data_dir=data_dir,
            dataset_name=dataset_name,
            split=split,
            splits_root=splits_root,
            image_size=image_size
        )
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
        gt_path = (
            str(row['mask_path']) if 'mask_path' in row and pd.notna(row['mask_path'])
            else os.path.join(self.data_dir, 'Class_Annotations', f"{base_name}.png")
        )
        gt_mask = self._safe_imread(gt_path, cv2.IMREAD_GRAYSCALE)

        if hflip:
            gt_mask = cv2.flip(gt_mask, 1)

        gt_mask = cv2.resize(gt_mask, (self.image_size[1], self.image_size[0]), interpolation=cv2.INTER_NEAREST)
        gt_t = torch.as_tensor(gt_mask, dtype=torch.long)

        if rotation_angle != 0.0:
            rgb_rot = TF.rotate(x_full[:3], rotation_angle, interpolation=InterpolationMode.BILINEAR)
            dt_rot = TF.rotate(x_full[3:], rotation_angle, interpolation=InterpolationMode.NEAREST)
            x_full = torch.cat([rgb_rot, dt_rot], dim=0)
            gt_t = TF.rotate(
                gt_t.unsqueeze(0).float(),
                rotation_angle,
                interpolation=InterpolationMode.NEAREST,
                fill=255.0
            ).squeeze(0).long()

        if self.enable_augmentation and self.split == 'train':
            if random.random() > 0.5:
                rgb = x_full[:3, :, :]
                rgb = TF.adjust_brightness(rgb, brightness_factor=1.0 + (random.random() - 0.5) * 0.4)
                rgb = TF.adjust_contrast(rgb, contrast_factor=1.0 + (random.random() - 0.5) * 0.4)
                rgb = TF.adjust_saturation(rgb, saturation_factor=1.0 + (random.random() - 0.5) * 0.4)
                rgb = TF.adjust_hue(rgb, hue_factor=(random.random() - 0.5) * 0.2)
                x_full[:3, :, :] = rgb

        x_normalized = self._normalize_rgb_only(x_full)

        if self.split == 'train' and self.modality_dropout_prob > 0.0:
            if random.random() < self.modality_dropout_prob:
                x_normalized[3, :, :] = 0.0
            if random.random() < self.modality_dropout_prob:
                x_normalized[4, :, :] = 0.0

        return {'x_full': x_normalized, 'seg_mask': gt_t}


class TTA_DownstreamSegmentationDataset(BaseRGBDTDataset):
    def __init__(self, data_dir: str = None, dataset_name: str = None, split: str = 'test',
                 splits_root: str = "data/splits", image_size: Tuple[int, int] = (480, 640)):
        super().__init__(
            data_dir=data_dir,
            dataset_name=dataset_name,
            split=split,
            splits_root=splits_root,
            image_size=image_size
        )

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.manifest.iloc[idx]
        base_name = self.image_files[idx]
        gt_path = (
            str(row['mask_path']) if 'mask_path' in row and pd.notna(row['mask_path'])
            else os.path.join(self.data_dir, 'Class_Annotations', f"{base_name}.png")
        )
        gt_mask = self._safe_imread(gt_path, cv2.IMREAD_GRAYSCALE)
        gt_mask = cv2.resize(gt_mask, (self.image_size[1], self.image_size[0]), interpolation=cv2.INTER_NEAREST)
        gt_t = torch.as_tensor(gt_mask, dtype=torch.long)

        x_base_norm = self._normalize_rgb_only(self._load_multimodal_tensors(idx, hflip=False))
        x_hflip_norm = self._normalize_rgb_only(self._load_multimodal_tensors(idx, hflip=True))
        x_tta_batch = torch.stack([x_base_norm, x_hflip_norm], dim=0)

        return {'x_tta_batch': x_tta_batch, 'seg_mask': gt_t}