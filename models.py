import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp
import copy

# ====================================================================================
# --- 1. MODALITY-SPECIFIC TOKENIZATION ---
# ====================================================================================

class ModalityIsolatedPatchEmbed(nn.Module):
    # PATCH: Added enable_dirac flag to isolate initialization effects from dual-stem topology
    def __init__(self, original_proj, enable_dirac=True):
        super().__init__()
        self.rgb_proj = original_proj
        
        self.depth_therm_proj = nn.Conv2d(
            in_channels=2, 
            out_channels=original_proj.out_channels, 
            kernel_size=original_proj.kernel_size, 
            stride=original_proj.stride, 
            padding=original_proj.padding, 
            bias=original_proj.bias is not None
        )
        nn.init.kaiming_normal_(self.depth_therm_proj.weight, mode='fan_out', nonlinearity='relu')
        if self.depth_therm_proj.bias is not None:
            nn.init.zeros_(self.depth_therm_proj.bias)
            
        # Explicitly decoupled physical calibration priors for V3 Architecture
        self.depth_scale = nn.Parameter(torch.ones(1, 1, 1, 1))
        self.depth_bias = nn.Parameter(torch.zeros(1, 1, 1, 1))
        
        # Bounded radiometric scaling for thermal calibration
        self.therm_scale = nn.Parameter(torch.ones(1, 1, 1, 1))
        self.therm_bias = nn.Parameter(torch.zeros(1, 1, 1, 1))
        
        self.dt_alignment = nn.Conv2d(
            in_channels=original_proj.out_channels,
            out_channels=original_proj.out_channels,
            kernel_size=1,
            bias=False
        )
        
        # PATCH: Conditional Dirac Isolation
        if enable_dirac:
            nn.init.dirac_(self.dt_alignment.weight)
        else:
            nn.init.kaiming_normal_(self.dt_alignment.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        x_rgb = x[:, :3, :, :]
        x_depth = x[:, 3:4, :, :]
        x_therm = x[:, 4:5, :, :]
        
        # 1. Encode Depth Scale-Invariance: Normalize by spatial mean to ensure geometric scale invariance
        # INJECTED EPSILON (+ 1e-8) to prevent zero-division NaN generation during perfectly uniform edge regions
        depth_mean = x_depth.mean(dim=(2, 3), keepdim=True) + 1e-8
        x_depth_normalized = x_depth / depth_mean
        x_depth_calibrated = (x_depth_normalized * self.depth_scale) + self.depth_bias
        
        # 2. Encode Thermal Radiometric Calibration: Bounded scaling via sigmoid to prevent uncalibrated drift
        x_therm_calibrated = (x_therm * torch.sigmoid(self.therm_scale)) + self.therm_bias
        
        # Re-concatenate calibrated physical tensors
        x_dt_calibrated = torch.cat([x_depth_calibrated, x_therm_calibrated], dim=1)
        
        dt_features = self.depth_therm_proj(x_dt_calibrated)
        dt_aligned = self.dt_alignment(dt_features)
        
        return self.rgb_proj(x_rgb) + dt_aligned

class NaiveEarlyFusionPatchEmbed(nn.Module):
    def __init__(self, original_proj):
        super().__init__()
        
        self.unified_proj = nn.Conv2d(
            in_channels=5, 
            out_channels=original_proj.out_channels, 
            kernel_size=original_proj.kernel_size, 
            stride=original_proj.stride, 
            padding=original_proj.padding, 
            bias=original_proj.bias is not None
        )
        
        nn.init.kaiming_normal_(self.unified_proj.weight, mode='fan_out', nonlinearity='relu')
        if self.unified_proj.bias is not None:
            nn.init.zeros_(self.unified_proj.bias)

    def forward(self, x):
        return self.unified_proj(x)

# ====================================================================================
# --- 2. JEPA COMPONENTS ---
# ====================================================================================

class PositionalEncoding2D(nn.Module):
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

        emb_x = torch.cat((sin_inp_x.sin(), sin_inp_x.cos()), dim=-1).unsqueeze(0).expand(H, W, -1)
        emb_y = torch.cat((sin_inp_y.sin(), sin_inp_y.cos()), dim=-1).unsqueeze(1).expand(H, W, -1)
        
        emb = (emb_x + emb_y).permute(2, 0, 1).unsqueeze(0).expand(B, -1, H, W)
        return tensor + emb

class SpatialJEPAPredictor(nn.Module):
    def __init__(self, embed_dim, hidden_dim=1024):
        super().__init__()
        self.pos_embed = PositionalEncoding2D(embed_dim)
        
        self.mask_token = nn.Parameter(torch.zeros(1, embed_dim, 1, 1))
        nn.init.normal_(self.mask_token, std=0.02)
        
        self.predictor = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1, groups=embed_dim, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1, groups=embed_dim, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
            nn.Conv2d(embed_dim, hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, embed_dim, kernel_size=1)
        )

    def forward(self, z_context, latent_mask):
        grid = z_context * (1.0 - latent_mask) + (self.mask_token * latent_mask)
        grid_with_pos = self.pos_embed(grid)
        return z_context + self.predictor(grid_with_pos)

# ====================================================================================
# --- 3. LORA INJECTION MODULES ---
# ====================================================================================

class LoRALinear(nn.Module):
    """Wraps an existing nn.Linear layer to inject low-rank adaptation matrices A and B."""
    def __init__(self, original_layer, r=8, alpha=16):
        super().__init__()
        self.original_layer = original_layer
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        
        in_features = original_layer.in_features
        out_features = original_layer.out_features
        
        self.lora_A = nn.Parameter(torch.zeros(in_features, r))
        self.lora_B = nn.Parameter(torch.zeros(r, out_features))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        # Y = Wx + (ABx) * scaling
        return self.original_layer(x) + (x @ self.lora_A @ self.lora_B) * self.scaling

def apply_lora_to_mit(model, r=8, alpha=16):
    """
    Dynamically traverses the MiT backbone and injects LoRA into Query and Value 
    projection layers to allow safe task-bending without catastrophic forgetting.
    """
    for name, module in model.named_modules():
        # MiT SRA blocks name their Q and KV projections `.q` and `.kv`
        if name.endswith('.q') or name.endswith('.kv'):
            parent_name = name.rsplit('.', 1)[0]
            child_name = name.rsplit('.', 1)[1]
            parent = model.get_submodule(parent_name)
            original = getattr(parent, child_name)
            
            if isinstance(original, nn.Linear):
                setattr(parent, child_name, LoRALinear(original, r, alpha))

# ====================================================================================
# --- 4. THE MASTER ARCHITECTURES ---
# ====================================================================================
    
class MultimodalJEPA(nn.Module):
    def __init__(self, backbone_name='mit_b1', isolated_stem=True, enable_dirac=True):
        super().__init__()
        self.isolated_stem = isolated_stem
        self.context_encoder = smp.encoders.get_encoder(backbone_name, in_channels=3, weights='imagenet')
        original_proj = self.context_encoder.patch_embed1.proj
        
        if self.isolated_stem:
            self.context_encoder.patch_embed1.proj = ModalityIsolatedPatchEmbed(original_proj, enable_dirac=enable_dirac)
        else:
            self.context_encoder.patch_embed1.proj = NaiveEarlyFusionPatchEmbed(original_proj)
        
        self.target_encoder = copy.deepcopy(self.context_encoder)
        for p in self.target_encoder.parameters(): p.requires_grad = False
            
        self.predictor = SpatialJEPAPredictor(embed_dim=self.context_encoder.out_channels[-1])
        self.encoder_mask_token = nn.Parameter(torch.zeros(1, 5, 1, 1))
        nn.init.normal_(self.encoder_mask_token, std=0.02)

    def train(self, mode=True):
        super().train(mode)
        self.target_encoder.eval()
        return self

    @torch.no_grad()
    def update_target_network(self, tau=0.996):
        for ctx_p, tgt_p in zip(self.context_encoder.parameters(), self.target_encoder.parameters()):
            tgt_p.data = tau * tgt_p.data + (1.0 - tau) * ctx_p.data
        for ctx_b, tgt_b in zip(self.context_encoder.buffers(), self.target_encoder.buffers()):
            tgt_b.data = tau * tgt_b.data + (1.0 - tau) * ctx_b.data

    def forward(self, x_visible, x_full, high_res_mask):
        B, C, H, W = x_full.shape
        mask_expanded = high_res_mask.expand(-1, C, -1, -1)
        x_context_input = x_visible + (self.encoder_mask_token * mask_expanded)
        
        z_context = self.context_encoder(x_context_input)[-1]
        with torch.no_grad():
            z_target = self.target_encoder(x_full)[-1].detach()
            
        latent_mask = F.interpolate(high_res_mask, size=z_context.shape[2:], mode='nearest')
        z_pred = self.predictor(z_context, latent_mask)
        return z_pred, z_target, latent_mask

class SegFormerAllMLPDecoder(nn.Module):
    def __init__(self, in_channels_list, embedding_dim=256, num_classes=10):
        super().__init__()
        self.linear_c4 = nn.Conv2d(in_channels_list[3], embedding_dim, kernel_size=1)
        self.linear_c3 = nn.Conv2d(in_channels_list[2], embedding_dim, kernel_size=1)
        self.linear_c2 = nn.Conv2d(in_channels_list[1], embedding_dim, kernel_size=1)
        self.linear_c1 = nn.Conv2d(in_channels_list[0], embedding_dim, kernel_size=1)

        self.linear_fuse = nn.Sequential(
            nn.Conv2d(embedding_dim * 4, embedding_dim, kernel_size=1, bias=False),
            # Substituted standard BN with GroupNorm(1, C) to act natively as LayerNorm 
            # across channels, preserving stability under bounded hardware batch constraints.
            nn.GroupNorm(1, embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1)
        )
        self.linear_pred = nn.Conv2d(embedding_dim, num_classes, kernel_size=1)

    def forward(self, features):
        c1, c2, c3, c4 = features[-4:] 
        _c4 = self.linear_c4(c4)
        _c3 = self.linear_c3(c3)
        _c2 = self.linear_c2(c2)
        _c1 = self.linear_c1(c1)

        _c4 = F.interpolate(_c4, size=c1.shape[2:], mode='bilinear', align_corners=False)
        _c3 = F.interpolate(_c3, size=c1.shape[2:], mode='bilinear', align_corners=False)
        _c2 = F.interpolate(_c2, size=c1.shape[2:], mode='bilinear', align_corners=False)

        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))
        return self.linear_pred(_c)

class TMLPN_Downstream_v1(nn.Module):
    """Legacy V1 Architecture (Preserved for compatibility and fallback)"""
    def __init__(self, num_classes=10, backbone_name='mit_b1', isolated_stem=True):
        super().__init__()
        self.isolated_stem = isolated_stem
        self.context_encoder = smp.encoders.get_encoder(backbone_name, in_channels=3, weights=None)
        original_proj = self.context_encoder.patch_embed1.proj
        if self.isolated_stem:
            self.context_encoder.patch_embed1.proj = ModalityIsolatedPatchEmbed(original_proj)
        else:
            self.context_encoder.patch_embed1.proj = NaiveEarlyFusionPatchEmbed(original_proj)
        
        encoder_channels = self.context_encoder.out_channels[-4:]
        self.decode_head = SegFormerAllMLPDecoder(in_channels_list=encoder_channels, embedding_dim=256, num_classes=num_classes)

    def forward(self, x_full):
        features = self.context_encoder(x_full)
        logits = self.decode_head(features)
        return F.interpolate(logits, size=x_full.shape[2:], mode='bilinear', align_corners=False)

class TMLPN_Downstream_v2(nn.Module):
    """Legacy V2 Architecture (Preserved for compatibility and fallback)"""
    def __init__(self, num_classes=10, backbone_name='mit_b1', isolated_stem=True, use_lora=False, lora_r=8, lora_alpha=16):
        super().__init__()
        self.isolated_stem = isolated_stem
        self.context_encoder = smp.encoders.get_encoder(backbone_name, in_channels=3, weights=None)
        
        original_proj = self.context_encoder.patch_embed1.proj
        if self.isolated_stem:
            self.context_encoder.patch_embed1.proj = ModalityIsolatedPatchEmbed(original_proj)
        else:
            self.context_encoder.patch_embed1.proj = NaiveEarlyFusionPatchEmbed(original_proj)
            
        if use_lora:
            apply_lora_to_mit(self.context_encoder, r=lora_r, alpha=lora_alpha)
        
        encoder_channels = self.context_encoder.out_channels[-4:]
        self.decode_head = SegFormerAllMLPDecoder(in_channels_list=encoder_channels, embedding_dim=256, num_classes=num_classes)

    def forward(self, x_full, return_features=False):
        features = self.context_encoder(x_full)
        logits = self.decode_head(features)
        logits_upsampled = F.interpolate(logits, size=x_full.shape[2:], mode='bilinear', align_corners=False)
        
        if return_features:
            return logits_upsampled, features[-4:]
        return logits_upsampled

class TMLPN_Downstream_v3(nn.Module):
    """
    PHASE 2: V3 Supervised Semantic Segmentation Architecture.
    Integrates Modality-Decoupled Physical Priors, LoRA, and Feature-Level KD.
    """
    def __init__(self, num_classes=10, backbone_name='mit_b1', isolated_stem=True, enable_dirac=True, use_lora=False, lora_r=8, lora_alpha=16):
        super().__init__()
        self.isolated_stem = isolated_stem
        self.context_encoder = smp.encoders.get_encoder(backbone_name, in_channels=3, weights=None)
        
        # Stem Extraction with explicit V3 Physical Priors
        original_proj = self.context_encoder.patch_embed1.proj
        if self.isolated_stem:
            self.context_encoder.patch_embed1.proj = ModalityIsolatedPatchEmbed(original_proj, enable_dirac=enable_dirac)
        else:
            self.context_encoder.patch_embed1.proj = NaiveEarlyFusionPatchEmbed(original_proj)
            
        # LoRA Injection
        if use_lora:
            apply_lora_to_mit(self.context_encoder, r=lora_r, alpha=lora_alpha)
        
        encoder_channels = self.context_encoder.out_channels[-4:]
        self.decode_head = SegFormerAllMLPDecoder(in_channels_list=encoder_channels, embedding_dim=256, num_classes=num_classes)

    def forward(self, x_full, return_features=False):
        features = self.context_encoder(x_full)
        logits = self.decode_head(features)
        logits_upsampled = F.interpolate(logits, size=x_full.shape[2:], mode='bilinear', align_corners=False)
        
        if return_features:
            # Drop the stem and return the 4 main transformer stages for feature distillation
            return logits_upsampled, features[-4:]
        return logits_upsampled