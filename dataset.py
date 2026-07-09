import os
import cv2
import csv # Added to parse your V1 dataset splits
import torch
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from torchvision import tv_tensors
from torchvision.transforms import v2

class TriModalPredictiveDataset(Dataset):
    def __init__(self, data_dir: str, split: str = 'train', image_size: tuple = (480, 640), mask_ratio: float = 0.50):
        """
        Dataloader for JEPA-Inspired TriModal Segmentation.
        Args:
            data_dir: Path to the MM5 dataset directory.
            split: 'train' or 'eval'.
            image_size: Target tensor resolution (H, W).
            mask_ratio: Percentage of the thermal image to block mask (0.0 to 1.0).
        """
        self.data_dir = data_dir
        self.split = split
        self.image_size = image_size
        self.mask_ratio = mask_ratio
        
        # --- DYNAMIC CLASS COUNT ---
        class_file = os.path.join(data_dir, "classes.txt")
        if os.path.exists(class_file):
            with open(class_file, "r") as f:
                # Count non-empty lines, excluding potential headers
                self.num_classes = len([line for line in f if line.strip()])
            print(f"[*] Detected {self.num_classes} classes from {class_file}")
        else:
            self.num_classes = 14 # Fallback
            print(f"[!] Warning: classes.txt not found in {data_dir}. Defaulting to {self.num_classes}")
            
        # Read from your custom V1 CSV files instead of the academic .txt files
        csv_path = os.path.join(data_dir, f'{split}_dataset.csv')
        self.image_files = []
        
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"[!] Could not find the split file at {csv_path}. Please ensure your DVC pull completed successfully.")
            
        with open(csv_path, 'r', encoding='utf-8-sig') as f:
            reader = csv.reader(f)
            header = next(reader) # Skip the header row
            for row in reader:
                if row:
                    # Grab the first column (assuming it holds the filename) 
                    # and strip any extensions so we have a clean base_name
                    base_name = row[0].replace('_rgb.png', '').replace('.png', '').replace('.jpg', '')
                    self.image_files.append(base_name)
                    
        # Define Deterministic Augmentations (Only for Training)
        if self.split == 'train':
            self.augmentations = v2.Compose([
                v2.RandomHorizontalFlip(p=0.5),
                v2.RandomVerticalFlip(p=0.2),
                v2.RandomRotation(degrees=15),
                v2.RandomResizedCrop(size=self.image_size, scale=(0.8, 1.0), ratio=(0.75, 1.33))
            ])
        else:
            self.augmentations = None

    def __len__(self):
        return len(self.image_files)
    
    def _apply_block_mask(self, thermal_tensor: torch.Tensor, seg_mask: torch.Tensor, max_retries: int = 25) -> tuple:
        """
        Applies an Object-Aware Block Mask. 
        Added guard clauses to handle 0-mask ratios and prevent division by zero.
        """
        # --- GUARD CLAUSE: If no masking is requested, return original ---
        if self.mask_ratio <= 0:
            return thermal_tensor, torch.zeros((1, thermal_tensor.shape[1], thermal_tensor.shape[2]), device=thermal_tensor.device)

        C, H, W = thermal_tensor.shape
        mask = torch.zeros((1, H, W), dtype=torch.float32, device=thermal_tensor.device)
        
        # Calculate block dimensions
        mask_area = int(H * W * self.mask_ratio)
        
        # Safety: Ensure dimensions are at least 1 pixel to prevent ZeroDivisionError
        block_h = max(1, int(np.sqrt(mask_area * (H / W))))
        block_w = max(1, int(mask_area / block_h))
        
        # Ensure block doesn't exceed image dimensions
        block_h = min(block_h, H - 1)
        block_w = min(block_w, W - 1)
        
        best_top, best_left = 0, 0
        max_overlap = -1
        
        # --- REJECTION SAMPLING LOOP ---
        for _ in range(max_retries):
            top = np.random.randint(0, H - block_h)
            left = np.random.randint(0, W - block_w)
            
            target_region = seg_mask[top:top + block_h, left:left + block_w]
            valid_pixels = ((target_region > 0) & (target_region != 255)).sum().item()
            
            if valid_pixels > max_overlap:
                max_overlap = valid_pixels
                best_top = top
                best_left = left
                
            if valid_pixels > (block_h * block_w * 0.15):
                break
                
        # Apply the mask
        mask[:, best_top:best_top + block_h, best_left:best_left + block_w] = 1.0
        
        masked_thermal = thermal_tensor.clone()
        masked_thermal = masked_thermal * (1.0 - mask)
        
        return masked_thermal, mask

    def _simulate_vertical_parallax(self, thermal_image: np.ndarray, max_shift: int = 40) -> np.ndarray:
        """
        Simulates the mechanical vertical parallax inherent to the 25mm Y-axis sensor offset.
        Dynamically shifts the thermal image vertically during training.
        """
        if self.split != 'train':
            return thermal_image # Keep static during validation
            
        # Random vertical shift prioritizing Y-axis displacement
        y_shift = np.random.randint(-max_shift, max_shift)
        x_shift = np.random.randint(-5, 5) # Minimal X-axis drift
        
        M = np.float32([[1, 0, x_shift], [0, 1, y_shift]])
        shifted_thermal = cv2.warpAffine(
            thermal_image, M, (thermal_image.shape[1], thermal_image.shape[0]), 
            borderMode=cv2.BORDER_REFLECT_101
        )
        return shifted_thermal
    
    def _sanitize_labels(self, mask):
        # Force all class indices to be 0 to (num_classes - 1) or 255
        clean_mask = torch.zeros_like(mask)
        clean_mask[mask == 255] = 255
        
        # Now dynamically safe against any class count
        valid_indices = (mask >= 0) & (mask < self.num_classes)
        clean_mask[valid_indices] = mask[valid_indices]
        
        return clean_mask

    def __getitem__(self, idx):
        base_name = self.image_files[idx]
        
        # 1. Load Modalities (Corrected MM5 Folder Structure)
        # Processed files are sequentially numbered (e.g., '236.png') across all directories
        rgb_path = os.path.join(self.data_dir, 'RGB', f"{base_name}.png")
        depth_path = os.path.join(self.data_dir, 'Depth', f"{base_name}.png") # Corrected from DF980N
        therm_path = os.path.join(self.data_dir, 'Thermal', f"{base_name}.png")
        mask_path = os.path.join(self.data_dir, 'Class_Annotations', f"{base_name}.png")
        
        rgb = cv2.cvtColor(cv2.imread(rgb_path), cv2.COLOR_BGR2RGB)
        
        # CRITICAL: cv2.IMREAD_ANYDEPTH forces OpenCV to retain the 16-bit structure
        depth = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)
        therm = cv2.imread(therm_path, cv2.IMREAD_ANYDEPTH)
        gt_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # 2. Resize to target tensor resolution
        rgb = cv2.resize(rgb, (self.image_size[1], self.image_size[0]))
        # Use INTER_NEAREST for raw data to prevent blending distinct spatial/thermal values
        depth = cv2.resize(depth, (self.image_size[1], self.image_size[0]), interpolation=cv2.INTER_NEAREST)
        therm = cv2.resize(therm, (self.image_size[1], self.image_size[0]), interpolation=cv2.INTER_NEAREST)
        gt_mask = cv2.resize(gt_mask, (self.image_size[1], self.image_size[0]), interpolation=cv2.INTER_NEAREST)

        # 3. Handle Mechanical Parallax
        therm = self._simulate_vertical_parallax(therm)
        
        # 4. Convert to Tensors for V2 Augmentations
        rgb_t = TF.to_tensor(rgb)       
        
        # Ensure depth bounds are validated at runtime rather than relying on strict 1000mm assumptions
        depth_np = depth.astype(np.float32)
        depth_max = depth_np.max()
        depth_scale = max(1000.0, depth_max) if depth_max > 0 else 1.0
        depth_t = torch.from_numpy(depth_np).unsqueeze(0) / depth_scale
        
        # Ensure thermal bounds are dynamic rather than assuming strict 16-bit (65535) cameras
        therm_np = therm.astype(np.float32)
        therm_max = therm_np.max()
        therm_scale = max(65535.0, therm_max) if therm_max > 0 else 1.0
        therm_t = torch.from_numpy(therm_np).unsqueeze(0) / therm_scale
        
        gt_t = torch.as_tensor(gt_mask, dtype=torch.long)
        
        # Wrap the mask to force NEAREST interpolation and prevent class blending
        gt_t = tv_tensors.Mask(torch.as_tensor(gt_mask, dtype=torch.long))
        
        
        # --- COORDINATED AUGMENTATION ---
        # Only apply if we are in the 'train' split
        if self.augmentations is not None:
            rgb_t, depth_t, therm_t, gt_t = self.augmentations(rgb_t, depth_t, therm_t, gt_t)
        # # --- COORDINATED AUGMENTATION ---
        # # v2 perfectly synchronizes the random parameters across all passed inputs
        # rgb_t, depth_t, therm_t, gt_t = self.augmentations(rgb_t, depth_t, therm_t, gt_t)
        
        # Unwrap the mask back to a standard PyTorch tensor
        gt_t = torch.as_tensor(gt_t, dtype=torch.long)
        # --- THE FIREWALL ---
        # This forces the labels to be valid and stops the CUDA crash
        gt_t = self._sanitize_labels(gt_t)
        
        # --- 5. MASK FIRST ---
        # Apply Object-Aware JEPA Predictive Block Masking
        # This creates masked_therm_t BEFORE we try to normalize it
        masked_therm_t, block_mask = self._apply_block_mask(therm_t, gt_t)
        
        # --- 6. NORMALIZE SECOND ---
        rgb_t = TF.normalize(rgb_t, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        depth_t = TF.normalize(depth_t, mean=[0.5], std=[0.15]) 
        
        # Now it is safe to normalize both thermal tensors
        masked_therm_t = TF.normalize(masked_therm_t, mean=[0.5], std=[0.25])
        therm_t = TF.normalize(therm_t, mean=[0.5], std=[0.25])
        
        # 7. Stack the Primary Stream (RGB + Depth = 4 Channels)
        rgbd_t = torch.cat([rgb_t, depth_t], dim=0) 

        return {
            'rgbd': rgbd_t,                  
            'therm_masked': masked_therm_t,  
            'therm_target': therm_t,         
            'block_mask': block_mask,        
            'seg_mask': gt_t                 
        }