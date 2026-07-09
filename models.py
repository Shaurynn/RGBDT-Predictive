import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp
import copy

# ====================================================================================
# --- 1. MODALITY-SPECIFIC TOKENIZATION ---
# ====================================================================================

class MultimodalJEPA(nn.Module):
    def __init__(self, backbone_name='mit_b1'):
        super().__init__()
        # Load strict 3-channel ImageNet weights
        self.context_encoder = smp.encoders.get_encoder(backbone_name, in_channels=3, weights='imagenet')
        
        # Surgically replace the first overlap patch embedding projection with the isolated stems
        original_proj = self.context_encoder.patch_embed1.proj
        self.context_encoder.patch_embed1.proj = ModalityIsolatedPatchEmbed(original_proj)
        
        self.target_encoder = copy.deepcopy(self.context_encoder)
        
        # 1. Lock the affine weights
        for p in self.target_encoder.parameters(): 
            p.requires_grad = False
            
        # 2. Lock the state machine to prevent Dropout/BN leakage
        self.target_encoder.eval()
            
        self.predictor = SpatialJEPAPredictor(embed_dim=self.context_encoder.out_channels[-1])

    def train(self, mode=True):
        """
        Overrides the default PyTorch train() method.
        Ensures that calling model.train() in the training loop does NOT 
        accidentally push the Target Encoder back into stochastic training mode.
        """
        super().train(mode)
        self.target_encoder.eval()
        return self

    @torch.no_grad()
    def update_target_network(self, tau=0.996):
        """
        True EMA Momentum Update.
        Synchronizes both the affine parameters AND the internal buffers (e.g., BN stats).
        """
        # Update Affine Parameters
        for ctx_p, tgt_p in zip(self.context_encoder.parameters(), self.target_encoder.parameters()):
            tgt_p.data = tau * tgt_p.data + (1.0 - tau) * ctx_p.data
            
        # Update Internal Buffers
        for ctx_b, tgt_b in zip(self.context_encoder.buffers(), self.target_encoder.buffers()):
            tgt_b.data = tau * tgt_b.data + (1.0 - tau) * ctx_b.data

    def forward(self, x_visible, x_full, high_res_mask):
        z_context = self.context_encoder(x_visible)[-1]
        
        # Target execution remains pristine and deterministic
        with torch.no_grad():
            z_target = self.target_encoder(x_full)[-1]
            
        B, C, H, W = z_context.shape
        latent_mask = F.interpolate(high_res_mask, size=(H, W), mode='nearest')
        
        z_pred = self.predictor(z_context, latent_mask)
        return z_pred, z_target, latent_mask

# ====================================================================================
# --- 2. JEPA COMPONENTS ---
# ====================================================================================

class PositionalEncoding2D(nn.Module):
    """
    Restored to pure additive frequencies. 
    Strictly aligns with JEPA/MAE literature for positional conditioning.
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

        emb_x = torch.cat((sin_inp_x.sin(), sin_inp_x.cos()), dim=-1).unsqueeze(0).expand(H, W, -1)
        emb_y = torch.cat((sin_inp_y.sin(), sin_inp_y.cos()), dim=-1).unsqueeze(1).expand(H, W, -1)
        
        emb = (emb_x + emb_y).permute(2, 0, 1).unsqueeze(0).expand(B, -1, H, W)
        return tensor + emb

class SpatialJEPAPredictor(nn.Module):
    """
    I-JEPA Compliant 2D Predictor.
    Executes true target-selective prediction via Token Replacement and 
    additive Positional Conditioning on a hierarchical 2D feature map.
    """
    def __init__(self, embed_dim, hidden_dim=1024):
        super().__init__()
        self.pos_embed = PositionalEncoding2D(embed_dim)
        
        # The explicit learnable target indicator
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
        # 1. Token Replacement: Erase context at target locations, inject [MASK]
        grid = z_context * (1.0 - latent_mask) + (self.mask_token * latent_mask)
        
        # 2. Positional Conditioning: Add spatial awareness 
        grid_with_pos = self.pos_embed(grid)
        
        # 3. Residual Inference: Anchor the prediction to the original context.
        # This mathematically forces the CNN to learn the spatial delta, 
        # naturally preserving the context representation.
        return z_context + self.predictor(grid_with_pos)

# ====================================================================================
# --- 3. THE MASTER ARCHITECTURES ---
# ====================================================================================

class ModalityIsolatedPatchEmbed(nn.Module):
    """
    Physically isolates modality ingestion at the stem.
    Integrates Learnable Physical Calibration Priors to recover absolute metric 
    and radiometric scale lost during numerical dataset standardization.
    """
    def __init__(self, original_proj):
        super().__init__()
        # Inherit unmodified weights for RGB (channels 0-2)
        self.rgb_proj = original_proj
        
        # Kaiming-initialized independent filters for Depth and Thermal (channels 3-4)
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
            
        # --- Learnable Physical Calibration Priors ---
        # Allows the network to dynamically recover physical scale (gamma) and shift (beta)
        self.dt_scale = nn.Parameter(torch.ones(1, 2, 1, 1))
        self.dt_bias = nn.Parameter(torch.zeros(1, 2, 1, 1))

    def forward(self, x):
        # Split 5-channel input into RGB (3ch) and Depth/Thermal (2ch)
        x_rgb = x[:, :3, :, :]
        x_dt = x[:, 3:, :, :]
        
        # Dynamically recalibrate the physical streams prior to convolution
        x_dt_calibrated = (x_dt * self.dt_scale) + self.dt_bias
        
        # Process separately and sum features within the latent embedding dimension
        return self.rgb_proj(x_rgb) + self.depth_therm_proj(x_dt_calibrated)
    
class MultimodalJEPA(nn.Module):
    def __init__(self, backbone_name='mit_b1'):
        super().__init__()
        # Load strict 3-channel ImageNet weights
        self.context_encoder = smp.encoders.get_encoder(backbone_name, in_channels=3, weights='imagenet')
        
        # Surgically replace the first overlap patch embedding projection with the isolated stems
        original_proj = self.context_encoder.patch_embed1.proj
        self.context_encoder.patch_embed1.proj = ModalityIsolatedPatchEmbed(original_proj)
        
        self.target_encoder = copy.deepcopy(self.context_encoder)
        for p in self.target_encoder.parameters(): p.requires_grad = False
            
        self.predictor = SpatialJEPAPredictor(embed_dim=self.context_encoder.out_channels[-1])

    @torch.no_grad()
    def update_target_network(self, tau=0.996):
        for ctx_p, tgt_p in zip(self.context_encoder.parameters(), self.target_encoder.parameters()):
            tgt_p.data = tau * tgt_p.data + (1.0 - tau) * ctx_p.data

    def forward(self, x_visible, x_full, high_res_mask):
        z_context = self.context_encoder(x_visible)[-1]
        
        # Target execution remains pristine and deterministic
        with torch.no_grad():
            # Explicit .detach() severs the computational graph entirely, providing 
            # an absolute mathematical guarantee against gradient leakage.
            z_target = self.target_encoder(x_full)[-1].detach()
            
        B, C, H, W = z_context.shape
        latent_mask = F.interpolate(high_res_mask, size=(H, W), mode='nearest')
        
        z_pred = self.predictor(z_context, latent_mask)
        return z_pred, z_target, latent_mask

class SegFormerAllMLPDecoder(nn.Module):
    """
    Lightweight multi-scale decoder designed for hierarchical Vision Transformers.
    Unifies spatial details from early stages with deep semantics from late stages
    without the computational overhead of heavy transposed convolutions.
    """
    def __init__(self, in_channels_list, embedding_dim=256, num_classes=10):
        super().__init__()
        # Project all hierarchical scales to a unified embedding dimension
        self.linear_c4 = nn.Conv2d(in_channels_list[3], embedding_dim, kernel_size=1)
        self.linear_c3 = nn.Conv2d(in_channels_list[2], embedding_dim, kernel_size=1)
        self.linear_c2 = nn.Conv2d(in_channels_list[1], embedding_dim, kernel_size=1)
        self.linear_c1 = nn.Conv2d(in_channels_list[0], embedding_dim, kernel_size=1)

        # Fuse the concatenated features
        self.linear_fuse = nn.Sequential(
            nn.Conv2d(embedding_dim * 4, embedding_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1)
        )
        self.linear_pred = nn.Conv2d(embedding_dim, num_classes, kernel_size=1)

    def forward(self, features):
        # SMP extracts multiple scales. We drop the initial stem and take the 4 main transformer stages.
        c1, c2, c3, c4 = features[-4:] 

        # 1. Unified Channel Projection
        _c4 = self.linear_c4(c4)
        _c3 = self.linear_c3(c3)
        _c2 = self.linear_c2(c2)
        _c1 = self.linear_c1(c1)

        # 2. Upsample deep semantics to the high-resolution 1/4 grid
        _c4 = F.interpolate(_c4, size=c1.shape[2:], mode='bilinear', align_corners=False)
        _c3 = F.interpolate(_c3, size=c1.shape[2:], mode='bilinear', align_corners=False)
        _c2 = F.interpolate(_c2, size=c1.shape[2:], mode='bilinear', align_corners=False)

        # 3. Concatenate and Fuse
        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))

        # 4. Generate Class Logits at 1/4 resolution
        return self.linear_pred(_c)

class TMLPN_Downstream(nn.Module):
    """
    PHASE 2: Supervised Semantic Segmentation Architecture.
    Utilizes a Multi-Scale All-MLP Decoder to preserve fine-grained spatial boundaries.
    """
    def __init__(self, num_classes=10, backbone_name='mit_b1'):
        super().__init__()
        
        self.context_encoder = smp.encoders.get_encoder(backbone_name, in_channels=3, weights=None)
        
        # Surgically replace the stem to match Phase 1 multimodal isolation
        original_proj = self.context_encoder.patch_embed1.proj
        self.context_encoder.patch_embed1.proj = ModalityIsolatedPatchEmbed(original_proj)
        
        # Extract the channel dimensions for the 4 hierarchical transformer stages
        encoder_channels = self.context_encoder.out_channels[-4:]
        
        self.decode_head = SegFormerAllMLPDecoder(
            in_channels_list=encoder_channels, 
            embedding_dim=256, 
            num_classes=num_classes
        )

    def forward(self, x_full):
        # Extract the list of multi-scale feature maps from the backbone
        features = self.context_encoder(x_full)
        
        # The decoder outputs predictions at 1/4 resolution
        logits = self.decode_head(features)
        
        # Final gentle 4x upsampling to the native sensor geometry
        return F.interpolate(logits, size=x_full.shape[2:], mode='bilinear', align_corners=False)