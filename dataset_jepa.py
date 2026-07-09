import os
import cv2
import csv
import torch
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

class JEPAPretrainDataset(Dataset):
    def __init__(self, data_dir: str, image_size: tuple = (480, 640), mask_ratio: float = 0.60):
        self.data_dir = data_dir
        self.image_size = image_size
        self.mask_ratio = mask_ratio
        
        csv_path = os.path.join(data_dir, 'train_dataset.csv')
        self.image_files = []
        with open(csv_path, 'r', encoding='utf-8-sig') as f:
            reader = csv.reader(f)
            next(reader) 
            for row in reader:
                if row:
                    self.image_files.append(row[0].replace('_rgb.png', '').replace('.png', '').replace('.jpg', ''))

    def __len__(self):
        return len(self.image_files)

    def _generate_patch_mask(self, h, w):
        """Generates random block masking applied uniformly across all 5 channels."""
        mask = torch.zeros((1, h, w), dtype=torch.float32)
        mask_area = int(h * w * self.mask_ratio)
        block_h = max(1, int(np.sqrt(mask_area * (h / w))))
        block_w = max(1, int(mask_area / block_h))
        
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
        
        rgb_t = TF.to_tensor(rgb)
        
        depth_np = depth.astype(np.float32)
        depth_t = torch.from_numpy(depth_np).unsqueeze(0) / (max(1000.0, depth_np.max()) if depth_np.max() > 0 else 1.0)
        
        therm_np = therm.astype(np.float32)
        therm_t = torch.from_numpy(therm_np).unsqueeze(0) / (max(65535.0, therm_np.max()) if therm_np.max() > 0 else 1.0)
        
        rgb_t = TF.normalize(rgb_t, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        depth_t = TF.normalize(depth_t, mean=[0.5], std=[0.15]) 
        therm_t = TF.normalize(therm_t, mean=[0.5], std=[0.25])
        
        # Unified 5-Channel Block
        x_full = torch.cat([rgb_t, depth_t, therm_t], dim=0)
        
        # Apply unified mask
        mask = self._generate_patch_mask(self.image_size[0], self.image_size[1])
        x_visible = x_full * (1.0 - mask)
        
        return {'x_full': x_full, 'x_visible': x_visible, 'mask': mask}