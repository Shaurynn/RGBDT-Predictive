import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class GlobalContextModalityAttention(nn.Module):
    def __init__(self, embed_dim: int = 512, num_heads: int = 4):
        """
        Cross-Modal Attention block built from primitives.
        Optimized for ONNX export and TensorRT execution.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        # Lightweight enhancement before pooling
        self.enhance_rgbd = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True)
        )
        self.enhance_therm = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True)
        )
        
        # Standard Multi-Head Attention primitive
        # batch_first=True is critical for ONNX/TensorRT stability
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim, 
            num_heads=num_heads, 
            batch_first=True
        )
        
        # Layer Normalization for the residual connection
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, z_rgbd: torch.Tensor, z_therm: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_rgbd: Primary pre-logits [Batch, Channels, Height, Width]
            z_therm: Thermal pre-logits [Batch, Channels, Height, Width]
        """
        B, C, H, W = z_rgbd.shape
        
        # 1. Enhance and Globally Pool (Collapse spatial dimensions)
        # This removes sensitivity to parallax and mechanical misalignment
        c_rgbd = F.adaptive_avg_pool2d(self.enhance_rgbd(z_rgbd), 1).flatten(1)   # [B, C]
        c_therm = F.adaptive_avg_pool2d(self.enhance_therm(z_therm), 1).flatten(1) # [B, C]
        
        # 2. Prepare Tokens for Attention
        # RGB-D acts as the Query (What physical structure are we looking at?)
        query = c_rgbd.unsqueeze(1) # [B, 1, C]
        
        # Both modalities act as Keys/Values (What thermodynamic context exists?)
        key_value = torch.stack([c_rgbd, c_therm], dim=1) # [B, 2, C]
        
        # 3. Cross-Modal Attention
        attn_out, _ = self.cross_attn(
            query=query,
            key=key_value,
            value=key_value,
            need_weights=False # Disable weight return for faster TensorRT execution
            ) # [B, 1, C]
        
        # 4. Residual Connection & Normalization
        # Squeeze out the sequence dimension and add the original RGB-D context
        context_fused = self.norm(attn_out.squeeze(1) + c_rgbd) # [B, C]
        
        # 5. Broadcast back to spatial resolution
        # [B, C] -> [B, C, 1, 1] -> [B, C, H, W]
        fused_spatial = context_fused.view(B, C, 1, 1).expand(-1, -1, H, W)
        
        return fused_spatial
    
class TriModalPredictiveNetwork(nn.Module):
    def __init__(self, num_classes: int = 14): # Defaulting to MM5's top-level class count
        super().__init__()
        
        # Backbone A: RGB-D Geometry Encoder (4 Input Channels)
        # Using pretrained=False because ImageNet weights are useless for 4-channel spatial tensors
        self.rgbd_encoder = timm.create_model(
            'segformer_b0', 
            in_chans=4, 
            pretrained=False, 
            features_only=True
        )
        
        # Backbone B: Thermodynamic Encoder (1 Input Channel)
        self.therm_encoder = timm.create_model(
            'segformer_b0', 
            in_chans=1, 
            pretrained=False, 
            features_only=True
        )
        
    # Add a lightweight reconstruction head to output the predicted thermal tensor
        self.therm_decoder = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1) # Outputs 1 channel (Thermal)
        )
        
        # The GCMA Fusion Head
        # MiT-B0 typically outputs 256 channels at its final stage
        self.fusion_head = GlobalContextModalityAttention(embed_dim=256, num_heads=4)
        
        # The Final Classifier (Predicting the Segmentation Mask)
        self.classifier = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, kernel_size=1)
        )

    def forward(self, x_rgbd: torch.Tensor, x_therm: torch.Tensor) -> torch.Tensor:
        """
        x_rgbd: [Batch, 4, H, W]
        x_therm: [Batch, 1, H, W] (Masked during pre-training, unmasked during inference)
        """
        # 1. Independent Feature Extraction
        # features_only=True returns a list of feature maps from each stage. 
        # We grab the final layer [-1] for late fusion.
        feat_rgbd = self.rgbd_encoder(x_rgbd)[-1]   # [Batch, 256, H/32, W/32]
        feat_therm = self.therm_encoder(x_therm)[-1] # [Batch, 256, H/32, W/32]
        
        # 2. Cross-Modal Fusion
        fused_features = self.fusion_head(feat_rgbd, feat_therm)
        
        # 3. Classification
        # Branch 1: Predict the segmentation mask (Anatomy)
        logits = self.classifier(fused_features)
        
        # Branch 2: Predict the missing thermal physics (Physics)
        therm_preds = self.therm_decoder(fused_features)
        
        # Upsample both to original resolution
        out_seg = F.interpolate(logits, size=(x_rgbd.shape[2], x_rgbd.shape[3]), mode='bilinear', align_corners=False)
        out_therm = F.interpolate(therm_preds, size=(x_rgbd.shape[2], x_rgbd.shape[3]), mode='bilinear', align_corners=False)
        
        return out_seg, out_therm