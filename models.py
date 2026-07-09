import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import segmentation_models_pytorch as smp

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

        # Generate frequencies for each axis
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq) # [W, C/2]
        sin_inp_y = torch.einsum("i,j->ij", pos_y, self.inv_freq) # [H, C/2]

        # 1. Create the base X and Y embeddings: Shape [W, C] and [H, C]
        emb_x = torch.cat((sin_inp_x.sin(), sin_inp_x.cos()), dim=-1) 
        emb_y = torch.cat((sin_inp_y.sin(), sin_inp_y.cos()), dim=-1)
        
        # 2. Correctly align and broadcast to a shared [H, W, C] spatial grid
        emb_x = emb_x.unsqueeze(0).expand(H, W, C) # Broadcast X across the height
        emb_y = emb_y.unsqueeze(1).expand(H, W, C) # Broadcast Y across the width
        
        # 3. Sum the frequencies, permute to [C, H, W], and broadcast to the Batch size
        emb = (emb_x + emb_y).permute(2, 0, 1).unsqueeze(0).expand(B, C, H, W)
        
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
    Utilizes 2D Positional Encodings, a learnable MASK token, and Depthwise Separable Convolutions 
    to selectively infer masked thermal targets from the surrounding contextual geometry.
    """
    def __init__(self, embed_dim, hidden_dim=1024):
        super().__init__()
        self.pos_embed = PositionalEncoding2D(embed_dim)
        
        # The learnable representation of "missing" physical data
        self.mask_token = nn.Parameter(torch.zeros(1, embed_dim, 1, 1))
        
        self.predictor = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1, groups=embed_dim, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
            nn.Conv2d(embed_dim, hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, embed_dim, kernel_size=1)
        )

    def forward(self, context_embedding, block_mask):
        B, C, H, W = context_embedding.shape
        
        # FIX: Dimensional Safeguard. Ensure the mask is a 4D float tensor [B, C, H, W]
        # before pushing it through the interpolation engine.
        if block_mask.dim() == 3:
            block_mask = block_mask.unsqueeze(1)
        block_mask = block_mask.float()
        
        # Downsample the high-res binary block mask to match the latent feature map dimensions
        mask_resized = F.interpolate(block_mask, size=(H, W), mode='nearest')
        
        # Erase contextual bleed in the masked regions and inject the learnable mask token
        masked_context = context_embedding * (1.0 - mask_resized) + (self.mask_token * mask_resized)
        
        # Inject geometry awareness so the convolution knows *where* the mask tokens are located
        x = self.pos_embed(masked_context)
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
        
        # 1. Instantiate Backbones via Segmentation Models PyTorch (SMP)
        # SMP natively houses the 'mit_b1' series and automatically handles N-channel weight initialization.
        self.rgbd_encoder = smp.encoders.get_encoder(
            name=backbone_name,
            in_channels=4,
            weights='imagenet'
        )
        self.therm_encoder = smp.encoders.get_encoder(
            name=backbone_name,
            in_channels=1,
            weights='imagenet'
        )
        
        # Extract the final channel dimension directly from the SMP encoder (e.g., 512 for mit_b1)
        final_dim = self.rgbd_encoder.out_channels[-1]
        
        # 2. Topology Engines
        self.fusion_head = SpatialReductionCrossAttention(dim=final_dim, reduction_ratio=8)
        self.latent_predictor = SpatialJEPAPredictor(embed_dim=final_dim)
        
        # 3. Semantic Segmentation Head (MLP Decoder based on SegFormer)
        self.decode_head = nn.Sequential(
            nn.Conv2d(final_dim, final_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(final_dim),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1),
            nn.Conv2d(final_dim, num_classes, kernel_size=1)
        )

    def forward(self, rgbd, therm_masked, therm_target=None, block_mask=None):
        # --- 1. CONTEXT ENCODING (The Observable World) ---
        # SMP encoders naturally return a list of spatial hierarchies
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
        if therm_target is None or block_mask is None:
            return seg_logits

        # --- 3. TARGET ENCODING (The Pristine Physics) ---
        with torch.no_grad(): # Explicit Stop-Gradient ensures the target encoder is locked
            therm_target_features = self.therm_encoder(therm_target)
            z_target = therm_target_features[-1]
            
        # --- 4. LATENT INFERENCE (True JEPA Target Selectivity) ---
        z_pred = self.latent_predictor(z_context, block_mask)
        
        return seg_logits, z_pred, z_target