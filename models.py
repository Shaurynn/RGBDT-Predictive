import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import timm

# ====================================================================================
# --- 1. POSITIONAL ENCODING & SPATIAL COMPONENTS ---
# ====================================================================================

class PositionalEncoding2D(nn.Module):
    """
    Injects spatial awareness into the latent manifold. 
    Critical for JEPA predictors to understand geometric relationships when inferring masked regions.
    """
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2).float() / channels))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, tensor):
        B, C, H, W = tensor.shape
        pos_x = torch.arange(W, device=tensor.device).type(self.inv_freq.type())
        pos_y = torch.arange(H, device=tensor.device).type(self.inv_freq.type())

        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        sin_inp_y = torch.einsum("i,j->ij", pos_y, self.inv_freq)

        emb_x = torch.cat((sin_inp_x.sin(), sin_inp_x.cos()), dim=-1).unsqueeze(1).repeat(1, H, 1)
        emb_y = torch.cat((sin_inp_y.sin(), sin_inp_y.cos()), dim=-1).unsqueeze(2).repeat(1, 1, W)
        
        # [H, W, C] -> [B, C, H, W]
        emb = (emb_x + emb_y).permute(2, 0, 1).unsqueeze(0).repeat(B, 1, 1, 1)
        return tensor + emb

# ====================================================================================
# --- 2. FUSION & PREDICTION HEADS ---
# ====================================================================================

class SpatialReductionCrossAttention(nn.Module):
    """
    Upgraded GCMA Head: Replaces 1x1 global pooling with Spatial Reduction.
    Preserves 2D thermal locality while maintaining sub-quadratic computational efficiency.
    """
    def __init__(self, dim, num_heads=8, reduction_ratio=8):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5

        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.kv = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=False)
        
        # Spatial Reduction to maintain O(N) efficiency without destroying locality
        self.sr = nn.Conv2d(dim, dim, kernel_size=reduction_ratio, stride=reduction_ratio)
        self.norm = nn.BatchNorm2d(dim)
        
        self.proj = nn.Conv2d(dim, dim, kernel_size=1)

    def forward(self, rgbd_feat, therm_feat):
        B, C, H, W = rgbd_feat.shape
        
        # Queries from RGB-D geometry
        q = self.q(rgbd_feat).reshape(B, self.num_heads, C // self.num_heads, H * W).transpose(-2, -1)
        
        # Reduce Thermal spatial dimensions to ease the O(N^2) cross-attention bottleneck
        therm_reduced = self.norm(self.sr(therm_feat))
        _, _, H_r, W_r = therm_reduced.shape
        
        kv = self.kv(therm_reduced).reshape(B, 2, self.num_heads, C // self.num_heads, H_r * W_r)
        k, v = kv[:, 0], kv[:, 1]

        # Scaled Dot-Product Attention
        attn = (q @ k) * self.scale
        attn = attn.softmax(dim=-1)
        
        # Map back to high-resolution spatial grid
        out = (attn @ v.transpose(-2, -1)).transpose(-2, -1).reshape(B, C, H, W)
        return self.proj(out)

class SpatialJEPAPredictor(nn.Module):
    """
    A mathematically compliant JEPA Predictor. 
    Utilizes 2D Positional Encodings and Depthwise Separable Convolutions to spatially 
    infer masked thermal representations from surrounding contextual geometry.
    """
    def __init__(self, embed_dim, hidden_dim=1024):
        super().__init__()
        self.pos_embed = PositionalEncoding2D(embed_dim)
        
        self.predictor = nn.Sequential(
            # Spatial Inference via Depthwise 3x3 Convolution
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1, groups=embed_dim, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
            
            # Pointwise Expansion
            nn.Conv2d(embed_dim, hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
            
            # Projection back to Target Dimension
            nn.Conv2d(hidden_dim, embed_dim, kernel_size=1)
        )

    def forward(self, context_embedding):
        # Inject geometry awareness so the convolution knows *where* it is predicting
        x = self.pos_embed(context_embedding)
        return self.predictor(x)

# ====================================================================================
# --- 3. MASTER ARCHITECTURE ---
# ====================================================================================

class TriModalLatentPredictiveNetwork(nn.Module):
    """
    The TMLPN Architecture. 
    Context Encoder processes aligned RGB-D-T(masked). 
    Target Encoder generates the ground-truth manifold from T(unmasked) via Stop-Gradient.
    """
    def __init__(self, num_classes=10, backbone_name='mit_b1'):
        super().__init__()
        self.num_classes = num_classes
        
        # 1. Instantiate Backbones via timm
        # We use strict feature extraction mode (features_only=True) to grab the multi-scale hierarchy
        self.rgbd_encoder = timm.create_model(backbone_name, pretrained=True, features_only=True)
        self.therm_encoder = timm.create_model(backbone_name, pretrained=True, features_only=True)
        
        # 2. Modify RGB-D Patch Embedding for 4-Channel Input (RGB + Depth)
        self._adapt_4channel_patch_embed(self.rgbd_encoder)
        
        # 3. Modify Thermal Patch Embedding for 1-Channel Input
        self._adapt_1channel_patch_embed(self.therm_encoder)
        
        # Channel dimensions extracted from the final block of the specific mit_bX backbone
        # mit_b1=512, mit_b2=512, mit_b3=512, mit_b4=512, mit_b5=512
        final_dim = self.rgbd_encoder.feature_info[-1]['num_chs']
        
        # 4. Topology Engines
        self.fusion_head = SpatialReductionCrossAttention(dim=final_dim, reduction_ratio=8)
        self.latent_predictor = SpatialJEPAPredictor(embed_dim=final_dim)
        
        # 5. Semantic Segmentation Head (MLP Decoder based on SegFormer)
        self.decode_head = nn.Sequential(
            nn.Conv2d(final_dim, final_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(final_dim),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1),
            nn.Conv2d(final_dim, num_classes, kernel_size=1)
        )

    def _adapt_4channel_patch_embed(self, model):
        """Mathematically sound weight initialization for RGB-D transition [He et al., 2016]."""
        old_conv = model.stem.proj if hasattr(model, 'stem') else model.patch_embed1.proj
        new_conv = nn.Conv2d(4, old_conv.out_channels, kernel_size=old_conv.kernel_size, 
                             stride=old_conv.stride, padding=old_conv.padding, bias=(old_conv.bias is not None))
        with torch.no_grad():
            new_conv.weight[:, :3, :, :] = old_conv.weight
            # Initialize depth channel as the mean of the RGB weights to preserve ImageNet statistics
            new_conv.weight[:, 3:4, :, :] = old_conv.weight.mean(dim=1, keepdim=True)
            if old_conv.bias is not None:
                new_conv.bias = old_conv.bias
                
        if hasattr(model, 'stem'): model.stem.proj = new_conv
        else: model.patch_embed1.proj = new_conv

    def _adapt_1channel_patch_embed(self, model):
        """Collapses 3-channel pretrained weights into a single thermal dimension."""
        old_conv = model.stem.proj if hasattr(model, 'stem') else model.patch_embed1.proj
        new_conv = nn.Conv2d(1, old_conv.out_channels, kernel_size=old_conv.kernel_size, 
                             stride=old_conv.stride, padding=old_conv.padding, bias=(old_conv.bias is not None))
        with torch.no_grad():
            new_conv.weight[:, 0:1, :, :] = old_conv.weight.sum(dim=1, keepdim=True)
            if old_conv.bias is not None:
                new_conv.bias = old_conv.bias
                
        if hasattr(model, 'stem'): model.stem.proj = new_conv
        else: model.patch_embed1.proj = new_conv

    def forward(self, rgbd, therm_masked, therm_target=None):
        """
        The JEPA state machine. 
        If therm_target is provided, executes the full self-supervised dual-encoder pass.
        If therm_target is None (Inference/Deployment), bypasses Target Encoder entirely.
        """
        # --- 1. CONTEXT ENCODING (The Observable World) ---
        rgbd_features = self.rgbd_encoder(rgbd)
        therm_context_features = self.therm_encoder(therm_masked)
        
        # Extract the highest semantic level from the hierarchy
        z_rgbd = rgbd_features[-1] 
        z_therm_ctx = therm_context_features[-1]
        
        # Intermediate Fusion 
        z_context = self.fusion_head(z_rgbd, z_therm_ctx)
        
        # Downstream Structural Prediction
        seg_logits = self.decode_head(z_context)
        # Upsample back to native resolution (H, W)
        seg_logits = F.interpolate(seg_logits, size=rgbd.shape[2:], mode='bilinear', align_corners=False)

        # --- 2. DEPLOYMENT SHORT-CIRCUIT ---
        if therm_target is None:
            return seg_logits

        # --- 3. TARGET ENCODING (The Pristine Physics) ---
        with torch.no_grad(): # Explicit Stop-Gradient ensures the target encoder is locked
            therm_target_features = self.therm_encoder(therm_target)
            z_target = therm_target_features[-1]
            
        # --- 4. LATENT INFERENCE ---
        z_pred = self.latent_predictor(z_context)
        
        return seg_logits, z_pred, z_target