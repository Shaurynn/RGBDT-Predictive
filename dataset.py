import os
import cv2
import csv # Added to parse your V1 dataset splits
import torch
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

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

    def __len__(self):
        return len(self.image_files)
    
    def _apply_block_mask(self, thermal_tensor: torch.Tensor) -> tuple:
        """
        Applies a contiguous Block Mask to the thermal tensor to force predictive learning.
        Returns the masked tensor and the binary mask (1 = masked, 0 = visible).
        """
        C, H, W = thermal_tensor.shape
        mask = torch.zeros((1, H, W), dtype=torch.float32)
        
        # Calculate block dimensions based on the mask_ratio
        mask_area = int(H * W * self.mask_ratio)
        block_h = int(np.sqrt(mask_area * (H / W)))
        block_w = int(mask_area / block_h)
        
        # Ensure block doesn't exceed image dimensions
        block_h = min(block_h, H - 1)
        block_w = min(block_w, W - 1)
        
        # Randomly select the top-left corner for the mask
        top = np.random.randint(0, H - block_h)
        left = np.random.randint(0, W - block_w)
        
        # Apply the mask (1 indicates the area the network must predict)
        mask[:, top:top + block_h, left:left + block_w] = 1.0
        
        # Create the input tensor by zeroing out the masked region
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
        
        # 4. Convert to Tensors & Apply Physical Scaling
        rgb_t = TF.to_tensor(rgb) # Automatically scales uint8 to [0.0, 1.0]
        
        # Convert 16-bit Depth (millimeters) to Float32 Meters for neural stability
        depth_t = torch.from_numpy(depth.astype(np.float32)).unsqueeze(0) / 1000.0 
        
        # Convert 16-bit Thermal to true Float32 Celsius
        therm_t = torch.from_numpy(therm.astype(np.float32)).unsqueeze(0)
        therm_t = (therm_t / 64.0) - 273.15
        
        gt_t = torch.as_tensor(gt_mask, dtype=torch.long)
        
        # 5. Modality-Wise Normalization
        rgb_t = TF.normalize(rgb_t, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        # Note: You will need to calculate the exact dataset-wide mean/std for these physical values later. 
        # Using placeholder constraints for an indoor agricultural environment (e.g., ~22°C ambient).
        depth_t = TF.normalize(depth_t, mean=[0.5], std=[0.15]) 
        therm_t = TF.normalize(therm_t, mean=[22.0], std=[5.0])
        
        # 6. Stack the Primary Stream (RGB + Depth = 4 Channels)
        rgbd_t = torch.cat([rgb_t, depth_t], dim=0) 
        
        # 7. Apply JEPA Predictive Block Masking to Thermal Stream
        masked_therm_t, block_mask = self._apply_block_mask(therm_t)

        return {
            'rgbd': rgbd_t,                  
            'therm_masked': masked_therm_t,  
            'therm_target': therm_t,         
            'block_mask': block_mask,        
            'seg_mask': gt_t                 
        }