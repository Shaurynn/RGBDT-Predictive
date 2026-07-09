import os
import cv2
import csv
import torch
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

class JEPAPretrainDataset(Dataset):
    def __init__(self, data_dir: str, image_size: tuple = (480, 640)):
        self.data_dir = data_dir
        self.image_size = image_size
        csv_path = os.path.join(data_dir, 'train_dataset.csv')
        self.image_files = []
        with open(csv_path, 'r', encoding='utf-8-sig') as f:
            reader = csv.reader(f)
            next(reader) 
            for row in reader:
                if row: self.image_files.append(row[0].replace('_rgb.png', '').replace('.png', '').replace('.jpg', ''))
                
        # Dynamically compute or cache the 5-channel mean and std
        self.mean, self.std = self._compute_dataset_stats()

    def __len__(self): return len(self.image_files)
    
    def _compute_dataset_stats(self):
        """Computes true empirical mean and std across the training split channels."""
        channels_sum = torch.zeros(5, dtype=torch.float64)
        channels_sq_sum = torch.zeros(5, dtype=torch.float64)
        num_pixels = 0

        # Quick single-pass calculation over file list
        for base_name in self.image_files:
            rgb = cv2.cvtColor(cv2.imread(os.path.join(self.data_dir, 'RGB', f"{base_name}.png")), cv2.COLOR_BGR2RGB)
            depth = cv2.imread(os.path.join(self.data_dir, 'Depth', f"{base_name}.png"), cv2.IMREAD_ANYDEPTH)
            therm = cv2.imread(os.path.join(self.data_dir, 'Thermal', f"{base_name}.png"), cv2.IMREAD_ANYDEPTH)
            
            rgb = cv2.resize(rgb, (self.image_size[1], self.image_size[0]))
            depth = cv2.resize(depth, (self.image_size[1], self.image_size[0]), interpolation=cv2.INTER_NEAREST)
            therm = cv2.resize(therm, (self.image_size[1], self.image_size[0]), interpolation=cv2.INTER_NEAREST)
            
            rgb_t = TF.to_tensor(rgb)
            depth_t = torch.from_numpy(depth.astype(np.float32)).unsqueeze(0) / 10000.0
            therm_t = torch.from_numpy(therm.astype(np.float32)).unsqueeze(0) / 65535.0
            
            x_5ch = torch.cat([rgb_t, depth_t, therm_t], dim=0) # [5, H, W]
            channels_sum += x_5ch.sum(dim=(1, 2)).double()
            channels_sq_sum += (x_5ch ** 2).sum(dim=(1, 2)).double()
            num_pixels += (self.image_size[0] * self.image_size[1])

        mean = channels_sum / num_pixels
        std = torch.sqrt((channels_sq_sum / num_pixels) - (mean ** 2))
        return mean.tolist(), std.tolist()

    def _generate_multiblock_mask(self, h, w, num_blocks=4):
        """
        I-JEPA Compliant Masking: Generates 4 overlapping blocks with varying scales (0.15-0.2) 
        and aspect ratios (0.75-1.5) to force multi-scale semantic reasoning.
        """
        mask = torch.zeros((1, h, w), dtype=torch.float32)
        for _ in range(num_blocks):
            scale = np.random.uniform(0.15, 0.20)
            aspect_ratio = np.random.uniform(0.75, 1.5)
            block_area = scale * h * w
            
            block_w = max(1, int(np.round(np.sqrt(block_area * aspect_ratio))))
            block_h = max(1, int(np.round(np.sqrt(block_area / aspect_ratio))))
            block_w = min(block_w, w - 1)
            block_h = min(block_h, h - 1)
            
            top = np.random.randint(0, h - block_h + 1)
            left = np.random.randint(0, w - block_w + 1)
            mask[:, top:top + block_h, left:left + block_w] = 1.0
        return mask

    def __getitem__(self, idx):
        base_name = self.image_files[idx]
        rgb = cv2.cvtColor(cv2.imread(os.path.join(self.data_dir, 'RGB', f"{base_name}.png")), cv2.COLOR_BGR2RGB)
        depth = cv2.imread(os.path.join(self.data_dir, 'Depth', f"{base_name}.png"), cv2.IMREAD_ANYDEPTH)
        therm = cv2.imread(os.path.join(self.data_dir, 'Thermal', f"{base_name}.png"), cv2.IMREAD_ANYDEPTH)
        
        rgb = cv2.resize(rgb, (self.image_size[1], self.image_size[0]))
        depth = cv2.resize(depth, (self.image_size[1], self.image_size[0]), interpolation=cv2.INTER_NEAREST)
        therm = cv2.resize(therm, (self.image_size[1], self.image_size[0]), interpolation=cv2.INTER_NEAREST)
        
        # Global Physical Scaling (Fixing the normalization variance collapse)
        rgb_t = TF.to_tensor(rgb)
        depth_t = torch.from_numpy(depth.astype(np.float32)).unsqueeze(0) / 10000.0 
        therm_t = torch.from_numpy(therm.astype(np.float32)).unsqueeze(0) / 65535.0 
        
        # Unified 5-Channel Tensor [5, H, W]
        x_full = torch.cat([rgb_t, depth_t, therm_t], dim=0)
        
        # Apply the dynamically computed 5-channel mean and std in one step
        x_normalized = TF.normalize(x_full, mean=self.mean, std=self.std)
        
        # Apply mask generation to the normalized visible view
        mask = self._generate_multiblock_mask(self.image_size[0], self.image_size[1])
        x_visible = x_normalized * (1.0 - mask)
        
        return {'x_full': x_normalized, 'x_visible': x_visible, 'mask': mask}

class DownstreamSegmentationDataset(Dataset):
    """Missing Dataset Class Restored for Phase 2 Downstream Fine-tuning."""
    def __init__(self, data_dir: str, split: str = 'train', image_size: tuple = (480, 640)):
        self.data_dir = data_dir
        self.split = split
        self.image_size = image_size
        csv_path = os.path.join(data_dir, f'{split}_dataset.csv')
        self.image_files = []
        with open(csv_path, 'r', encoding='utf-8-sig') as f:
            reader = csv.reader(f)
            next(reader) 
            for row in reader:
                if row: self.image_files.append(row[0].replace('_rgb.png', '').replace('.png', '').replace('.jpg', ''))
                
        # Dynamically compute or cache the 5-channel mean and std
        self.mean, self.std = self._compute_dataset_stats()

    def __len__(self): return len(self.image_files)

    def _compute_dataset_stats(self):
        """Computes true empirical mean and std across the training split channels."""
        channels_sum = torch.zeros(5, dtype=torch.float64)
        channels_sq_sum = torch.zeros(5, dtype=torch.float64)
        num_pixels = 0

        # Quick single-pass calculation over file list
        for base_name in self.image_files:
            rgb = cv2.cvtColor(cv2.imread(os.path.join(self.data_dir, 'RGB', f"{base_name}.png")), cv2.COLOR_BGR2RGB)
            depth = cv2.imread(os.path.join(self.data_dir, 'Depth', f"{base_name}.png"), cv2.IMREAD_ANYDEPTH)
            therm = cv2.imread(os.path.join(self.data_dir, 'Thermal', f"{base_name}.png"), cv2.IMREAD_ANYDEPTH)
            
            rgb = cv2.resize(rgb, (self.image_size[1], self.image_size[0]))
            depth = cv2.resize(depth, (self.image_size[1], self.image_size[0]), interpolation=cv2.INTER_NEAREST)
            therm = cv2.resize(therm, (self.image_size[1], self.image_size[0]), interpolation=cv2.INTER_NEAREST)
            
            rgb_t = TF.to_tensor(rgb)
            depth_t = torch.from_numpy(depth.astype(np.float32)).unsqueeze(0) / 10000.0
            therm_t = torch.from_numpy(therm.astype(np.float32)).unsqueeze(0) / 65535.0
            
            x_5ch = torch.cat([rgb_t, depth_t, therm_t], dim=0) # [5, H, W]
            channels_sum += x_5ch.sum(dim=(1, 2)).double()
            channels_sq_sum += (x_5ch ** 2).sum(dim=(1, 2)).double()
            num_pixels += (self.image_size[0] * self.image_size[1])

        mean = channels_sum / num_pixels
        std = torch.sqrt((channels_sq_sum / num_pixels) - (mean ** 2))
        return mean.tolist(), std.tolist()

    def __getitem__(self, idx):
        base_name = self.image_files[idx]
        rgb = cv2.cvtColor(cv2.imread(os.path.join(self.data_dir, 'RGB', f"{base_name}.png")), cv2.COLOR_BGR2RGB)
        depth = cv2.imread(os.path.join(self.data_dir, 'Depth', f"{base_name}.png"), cv2.IMREAD_ANYDEPTH)
        therm = cv2.imread(os.path.join(self.data_dir, 'Thermal', f"{base_name}.png"), cv2.IMREAD_ANYDEPTH)
        gt_mask = cv2.imread(os.path.join(self.data_dir, 'Class_Annotations', f"{base_name}.png"), cv2.IMREAD_GRAYSCALE)
        
        rgb = cv2.resize(rgb, (self.image_size[1], self.image_size[0]))
        depth = cv2.resize(depth, (self.image_size[1], self.image_size[0]), interpolation=cv2.INTER_NEAREST)
        therm = cv2.resize(therm, (self.image_size[1], self.image_size[0]), interpolation=cv2.INTER_NEAREST)
        gt_mask = cv2.resize(gt_mask, (self.image_size[1], self.image_size[0]), interpolation=cv2.INTER_NEAREST)
        
        # rgb_t = TF.normalize(TF.to_tensor(rgb), mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        # depth_t = TF.normalize(torch.from_numpy(depth.astype(np.float32)).unsqueeze(0) / 10000.0, mean=[0.5], std=[0.15])
        # therm_t = TF.normalize(torch.from_numpy(therm.astype(np.float32)).unsqueeze(0) / 65535.0, mean=[0.5], std=[0.25])
        rgb_t = TF.to_tensor(rgb)
        depth_t = torch.from_numpy(depth.astype(np.float32)).unsqueeze(0) / 10000.0 
        therm_t = torch.from_numpy(therm.astype(np.float32)).unsqueeze(0) / 65535.0 
        
        # Unified 5-Channel Tensor [5, H, W]
        x_full = torch.cat([rgb_t, depth_t, therm_t], dim=0)
        
        # Apply the dynamically computed 5-channel mean and std
        x_normalized = TF.normalize(x_full, mean=self.mean, std=self.std)
        
        gt_t = torch.as_tensor(gt_mask, dtype=torch.long)
        gt_t[gt_t == 255] = 255 

        # return {'x_full': torch.cat([rgb_t, depth_t, therm_t], dim=0), 'seg_mask': gt_t}        
        return {'x_full': x_normalized, 'seg_mask': gt_t}