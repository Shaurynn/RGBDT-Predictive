import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp

# ====================================================================================
# --- SHARED COMPONENTS ---
# ====================================================================================

class GlobalContextModalityAttention(nn.Module):
    def __init__(self, embed_dim: int = 512, num_heads: int = 8):
        """
        Cross-Modal Attention block.
        Uses spatial RGB-D features to query Global Thermodynamic Context.
        Shared by both the standard and latent predictive networks.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
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
        
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim, 
            num_heads=num_heads, 
            batch_first=True
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, z_rgbd: torch.Tensor, z_therm: torch.Tensor) -> torch.Tensor:
        B, C, H, W = z_rgbd.shape
        
        feat_rgbd = self.enhance_rgbd(z_rgbd)
        feat_therm = self.enhance_therm(z_therm)
        
        # Extract Global Context (Keys & Values)
        c_rgbd = F.adaptive_avg_pool2d(feat_rgbd, 1).flatten(1)
        c_therm = F.adaptive_avg_pool2d(feat_therm, 1).flatten(1)
        key_value = torch.stack([c_rgbd, c_therm], dim=1)
        
        # Spatial Geometry (Queries)
        query = feat_rgbd.view(B, C, H * W).permute(0, 2, 1)
        
        attn_out, _ = self.cross_attn(query=query, key=key_value, value=key_value, need_weights=False)
        context_fused = self.norm(attn_out + query)
        fused_spatial = context_fused.permute(0, 2, 1).contiguous().view(B, C, H, W)
        
        return fused_spatial


# ====================================================================================
# --- ARCHITECTURE 1: The Legacy Tri-Objective Model ---
# ====================================================================================

class TriModalPredictiveNetwork(nn.Module):
    def __init__(self, num_classes: int): 
        super().__init__()
        
        # 1. ImageNet Stem Patch for 4-Channels (Upgraded to mit_b1)
        self.rgbd_encoder = smp.encoders.get_encoder("mit_b1", in_channels=3, depth=5, weights="imagenet")
        first_conv = self.rgbd_encoder.patch_embed1.proj
        
        new_conv = nn.Conv2d(4, first_conv.out_channels, kernel_size=first_conv.kernel_size, stride=first_conv.stride, padding=first_conv.padding, bias=(first_conv.bias is not None))
        with torch.no_grad():
            new_conv.weight[:, :3, :, :] = first_conv.weight
            new_conv.weight[:, 3:4, :, :] = first_conv.weight.mean(dim=1, keepdim=True)
            if first_conv.bias is not None: new_conv.bias = first_conv.bias
        self.rgbd_encoder.patch_embed1.proj = new_conv
        
        # Thermal Encoder (Upgraded to mit_b1)
        self.therm_encoder = smp.encoders.get_encoder("mit_b1", in_channels=1, depth=5, weights=None)
        
        # GCMA Fusion Head (Scaled for 512 channels, increased to 8 attention heads)
        self.fusion_head = GlobalContextModalityAttention(embed_dim=512, num_heads=8)
        
        # Classifiers updated to accept 512-channel embeddings
        self.classifier = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, kernel_size=1)
        )
        self.therm_decoder = nn.Sequential(
            nn.Conv2d(512, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True), 
            nn.Conv2d(128, 1, kernel_size=1) 
        )
        self.aux_therm_classifier = nn.Sequential(
            nn.Conv2d(512, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.Dropout2d(0.1),
            nn.Conv2d(128, num_classes, kernel_size=1)
        )

    def forward(self, x_rgbd: torch.Tensor, x_therm: torch.Tensor):
        feat_rgbd = self.rgbd_encoder(x_rgbd)[-1]   
        feat_therm = self.therm_encoder(x_therm)[-1] 
        fused_features = self.fusion_head(feat_rgbd, feat_therm)
        
        logits_seg = self.classifier(fused_features)
        therm_preds = self.therm_decoder(fused_features)
        
        out_seg = F.interpolate(logits_seg, size=(x_rgbd.shape[2], x_rgbd.shape[3]), mode='bilinear', align_corners=False)
        out_therm = F.interpolate(therm_preds, size=(x_rgbd.shape[2], x_rgbd.shape[3]), mode='bilinear', align_corners=False)
        
        if self.training:
            aux_logits = self.aux_therm_classifier(feat_therm)
            out_aux_seg = F.interpolate(aux_logits, size=(x_rgbd.shape[2], x_rgbd.shape[3]), mode='bilinear', align_corners=False)
            return out_seg, out_therm, out_aux_seg
            
        return out_seg, out_therm


# ====================================================================================
# --- ARCHITECTURE 2: The LeWM-Inspired Latent Model ---
# ====================================================================================

class TriModalLatentPredictiveNetwork(nn.Module):
    def __init__(self, num_classes: int): 
        super().__init__()
        
        # 1. ImageNet Stem Patch for 4-Channels (Upgraded to mit_b1)
        self.rgbd_encoder = smp.encoders.get_encoder("mit_b1", in_channels=3, depth=5, weights="imagenet")
        first_conv = self.rgbd_encoder.patch_embed1.proj
        
        new_conv = nn.Conv2d(4, first_conv.out_channels, kernel_size=first_conv.kernel_size, stride=first_conv.stride, padding=first_conv.padding, bias=(first_conv.bias is not None))
        with torch.no_grad():
            new_conv.weight[:, :3, :, :] = first_conv.weight
            new_conv.weight[:, 3:4, :, :] = first_conv.weight.mean(dim=1, keepdim=True)
            if first_conv.bias is not None: new_conv.bias = first_conv.bias
        self.rgbd_encoder.patch_embed1.proj = new_conv
        
        # Thermal Encoder (Upgraded to mit_b1)
        self.therm_encoder = smp.encoders.get_encoder("mit_b1", in_channels=1, depth=5, weights=None)
        
        # GCMA Fusion Head (Scaled for 512 channels, increased to 8 attention heads)
        self.fusion_head = GlobalContextModalityAttention(embed_dim=512, num_heads=8)
        
        # Anatomy Classifier
        self.classifier = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, kernel_size=1)
        )
        
        # The Expanded Latent World Model Engine
        # Deepened from a single bottleneck to a 3-layer mapping to better predict physical state transitions
        self.latent_predictor = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=1, bias=False),
            nn.BatchNorm2d(1024), nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, kernel_size=1, bias=False),
            nn.BatchNorm2d(1024), nn.ReLU(inplace=True),
            nn.Conv2d(1024, 512, kernel_size=1)
        )

    def forward(self, x_rgbd: torch.Tensor, x_therm_masked: torch.Tensor, x_therm_target: torch.Tensor = None):
        feat_rgbd = self.rgbd_encoder(x_rgbd)[-1]   
        feat_therm_masked = self.therm_encoder(x_therm_masked)[-1] 
        fused_features = self.fusion_head(feat_rgbd, feat_therm_masked)
        
        logits_seg = self.classifier(fused_features)
        out_seg = F.interpolate(logits_seg, size=(x_rgbd.shape[2], x_rgbd.shape[3]), mode='bilinear', align_corners=False)
        
        # Inference / Deployment Mode 
        if not self.training or x_therm_target is None:
            return out_seg
            
        # Target Prediction Mode
        with torch.no_grad():
            z_target = self.therm_encoder(x_therm_target)[-1].detach()
            
        z_pred = self.latent_predictor(fused_features)
        return out_seg, z_pred, z_target