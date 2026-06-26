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
        
        # --- Modality-Specific Supervisory Head (Thermal Expert) ---
        # This forces the thermal encoder to independently recognize rot/decay 
        # without relying on the RGB-D visual texture.
        self.aux_therm_classifier = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(128, num_classes, kernel_size=1)
        )

    def forward(self, x_rgbd: torch.Tensor, x_therm: torch.Tensor):
        """
        x_rgbd: [Batch, 4, H, W]
        x_therm: [Batch, 1, H, W]
        """
        # 1. Independent Feature Extraction
        feat_rgbd = self.rgbd_encoder(x_rgbd)[-1]   
        feat_therm = self.therm_encoder(x_therm)[-1] 
        
        # 2. Cross-Modal Fusion & Primary Classification
        fused_features = self.fusion_head(feat_rgbd, feat_therm)
        logits_seg = self.classifier(fused_features)
        
        # 3. JEPA Physics Prediction
        therm_preds = self.therm_decoder(fused_features)
        
        # Upsample Primary Outputs
        out_seg = F.interpolate(logits_seg, size=(x_rgbd.shape[2], x_rgbd.shape[3]), mode='bilinear', align_corners=False)
        out_therm = F.interpolate(therm_preds, size=(x_rgbd.shape[2], x_rgbd.shape[3]), mode='bilinear', align_corners=False)
        
        # 4. Training-Only Expert Supervision
        if self.training:
            # Generate a segmentation mask using ONLY the thermal features
            aux_logits = self.aux_therm_classifier(feat_therm)
            out_aux_seg = F.interpolate(aux_logits, size=(x_rgbd.shape[2], x_rgbd.shape[3]), mode='bilinear', align_corners=False)
            return out_seg, out_therm, out_aux_seg
            
        # During eval/deployment, the aux head is entirely ignored
        return out_seg, out_therm
        
