import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp


# ====================================================================================
# --- 1. MODALITY-SPECIFIC TOKENIZATION & GATING ---
# ====================================================================================

class ConcurrentSpatialChannelSE(nn.Module):
    """
    Concurrent Spatial and Channel Squeeze & Excitation (scSE)
    Reference: Roy et al., MICCAI 2018
    """
    def __init__(self, channels: int, reduction_ratio: int = 16):
        super().__init__()
        hidden = max(1, channels // reduction_ratio)
        self.cSE = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        self.sSE = nn.Sequential(
            nn.Conv2d(channels, 1, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x * self.cSE(x)) + (x * self.sSE(x))


class ModalityIsolatedPatchEmbed(nn.Module):
    """
    Isolates ImageNet-pretrained RGB patch projection from physical Depth+Thermal
    projections using valid-pixel scale-invariant depth normalization, identity-centered
    bounded thermal calibration, Dirac 1x1 alignment, and optional scSE gating.
    """
    def __init__(self, original_proj, enable_dirac: bool = True, enable_scse: bool = True):
        super().__init__()
        self.rgb_proj = original_proj
        self.enable_scse = enable_scse

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

        self.depth_scale = nn.Parameter(torch.ones(1, 1, 1, 1))
        self.depth_bias = nn.Parameter(torch.zeros(1, 1, 1, 1))

        # Centered at 0.0 so 2.0 * sigmoid(0.0) == 1.0 (exact physical identity), bounded in (0, 2)
        self.therm_scale = nn.Parameter(torch.zeros(1, 1, 1, 1))
        self.therm_bias = nn.Parameter(torch.zeros(1, 1, 1, 1))

        self.dt_alignment = nn.Conv2d(
            in_channels=original_proj.out_channels,
            out_channels=original_proj.out_channels,
            kernel_size=1,
            bias=False
        )
        if enable_dirac:
            nn.init.dirac_(self.dt_alignment.weight)
        else:
            nn.init.kaiming_normal_(self.dt_alignment.weight, mode='fan_out', nonlinearity='relu')

        if self.enable_scse:
            self.scse_gate = ConcurrentSpatialChannelSE(
                channels=original_proj.out_channels,
                reduction_ratio=16
            )

    def forward_split(self, x: torch.Tensor):
        """Returns (rgb_aligned, dt_aligned) prior to additive fusion for analysis & forward reuse."""
        x_rgb = x[:, :3, :, :]
        x_depth = x[:, 3:4, :, :]
        x_therm = x[:, 4:5, :, :]

        valid_depth_mask = (x_depth > 1e-6).to(x_depth.dtype)
        valid_count = valid_depth_mask.sum(dim=(2, 3), keepdim=True)
        depth_mean = (x_depth * valid_depth_mask).sum(dim=(2, 3), keepdim=True) / valid_count.clamp(min=1.0)
        x_depth_normalized = torch.where(
            valid_count > 0,
            (x_depth / depth_mean.clamp(min=1e-4)) * valid_depth_mask,
            torch.zeros_like(x_depth)
        )
        x_depth_calibrated = ((x_depth_normalized * self.depth_scale) + self.depth_bias) * valid_depth_mask

        valid_therm_mask = (x_therm > 1e-6).to(x_therm.dtype)
        therm_gain = 2.0 * torch.sigmoid(self.therm_scale)
        x_therm_calibrated = ((x_therm * therm_gain) + self.therm_bias) * valid_therm_mask

        x_dt_calibrated = torch.cat([x_depth_calibrated, x_therm_calibrated], dim=1)
        dt_features = self.depth_therm_proj(x_dt_calibrated)
        dt_aligned = self.dt_alignment(dt_features)
        rgb_aligned = self.rgb_proj(x_rgb)
        return rgb_aligned, dt_aligned

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rgb_aligned, dt_aligned = self.forward_split(x)
        fused_features = rgb_aligned + dt_aligned
        if self.enable_scse:
            fused_features = self.scse_gate(fused_features)
        return fused_features


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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.unified_proj(x)


# ====================================================================================
# --- 2. JEPA COMPONENTS ---
# ====================================================================================

class PositionalEncoding2D(nn.Module):
    """
    Asymmetric 2D sinusoidal positional encoding that allocates disjoint channel
    halves to X and Y coordinates and concatenates them along the channel axis.
    """
    def __init__(self, channels: int):
        super().__init__()
        if channels % 4 != 0:
            raise ValueError(f"PositionalEncoding2D requires channels divisible by 4, got {channels}")
        self.channels = channels
        half_ch = channels // 2
        inv_freq = 1.0 / (10000.0 ** (torch.arange(0, half_ch, 2).float() / half_ch))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        _, _, H, W = tensor.shape
        pos_x = torch.arange(W, device=tensor.device, dtype=self.inv_freq.dtype)
        pos_y = torch.arange(H, device=tensor.device, dtype=self.inv_freq.dtype)

        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        sin_inp_y = torch.einsum("i,j->ij", pos_y, self.inv_freq)

        emb_x = torch.cat((sin_inp_x.sin(), sin_inp_x.cos()), dim=-1).unsqueeze(0).expand(H, W, -1)
        emb_y = torch.cat((sin_inp_y.sin(), sin_inp_y.cos()), dim=-1).unsqueeze(1).expand(H, W, -1)

        emb = torch.cat((emb_x, emb_y), dim=-1).permute(2, 0, 1).unsqueeze(0).to(dtype=tensor.dtype)
        return tensor + emb


class SpatialJEPAPredictor(nn.Module):
    def __init__(self, embed_dim: int, hidden_dim: int = 1024):
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

    def forward(self, z_context: torch.Tensor, latent_mask: torch.Tensor) -> torch.Tensor:
        grid = z_context * (1.0 - latent_mask) + (self.mask_token * latent_mask)
        grid_with_pos = self.pos_embed(grid)
        return grid + self.predictor(grid_with_pos)


# ====================================================================================
# --- 3. LORA INJECTION MODULES ---
# ====================================================================================

class LoRALinear(nn.Module):
    def __init__(self, original_layer: nn.Linear, r: int = 8, alpha: int = 16):
        super().__init__()
        if r <= 0:
            raise ValueError(f"[-] CRITICAL: LoRA rank 'r' must be strictly positive. Received: {r}")

        self.original_layer = original_layer
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r

        self.lora_A = nn.Parameter(torch.zeros(original_layer.in_features, r))
        self.lora_B = nn.Parameter(torch.zeros(r, original_layer.out_features))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.original_layer(x) + (x @ self.lora_A @ self.lora_B) * self.scaling


def apply_lora_to_mit(model: nn.Module, r: int = 8, alpha: int = 16) -> None:
    for name, _ in list(model.named_modules()):
        if name.endswith('.q') or name.endswith('.kv'):
            parent_name, child_name = name.rsplit('.', 1)
            parent = model.get_submodule(parent_name)
            original = getattr(parent, child_name)
            if isinstance(original, nn.Linear):
                setattr(parent, child_name, LoRALinear(original, r, alpha))


# ====================================================================================
# --- 4. THE MASTER ARCHITECTURES ---
# ====================================================================================

class MultimodalJEPA(nn.Module):
    def __init__(self, backbone_name: str = 'mit_b1', isolated_stem: bool = True,
                 enable_dirac: bool = True, enable_scse: bool = True):
        super().__init__()
        self.isolated_stem = isolated_stem
        self.context_encoder = smp.encoders.get_encoder(backbone_name, in_channels=3, weights='imagenet')
        original_proj = self.context_encoder.patch_embed1.proj

        if self.isolated_stem:
            self.context_encoder.patch_embed1.proj = ModalityIsolatedPatchEmbed(
                original_proj,
                enable_dirac=enable_dirac,
                enable_scse=enable_scse
            )
        else:
            self.context_encoder.patch_embed1.proj = NaiveEarlyFusionPatchEmbed(original_proj)

        self.target_encoder = copy.deepcopy(self.context_encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad = False

        self.predictor = SpatialJEPAPredictor(embed_dim=self.context_encoder.out_channels[-1])
        self.encoder_mask_token = nn.Parameter(torch.zeros(1, 3, 1, 1))
        nn.init.normal_(self.encoder_mask_token, std=0.02)

    def train(self, mode: bool = True):
        super().train(mode)
        self.target_encoder.eval()
        return self

    @torch.no_grad()
    def update_target_network(self, tau: float = 0.996) -> None:
        for ctx_p, tgt_p in zip(self.context_encoder.parameters(), self.target_encoder.parameters()):
            tgt_p.data.mul_(tau).add_(ctx_p.data, alpha=1.0 - tau)
        for ctx_b, tgt_b in zip(self.context_encoder.buffers(), self.target_encoder.buffers()):
            if ctx_b.is_floating_point():
                tgt_b.data.mul_(tau).add_(ctx_b.data, alpha=1.0 - tau)
            else:
                tgt_b.data.copy_(ctx_b.data)

    def forward(self, x_visible: torch.Tensor, x_full: torch.Tensor, high_res_mask: torch.Tensor):
        rgb_masked = x_visible[:, :3, :, :] + (self.encoder_mask_token * high_res_mask)
        x_context_input = torch.cat([rgb_masked, x_visible[:, 3:, :, :]], dim=1)

        z_context = self.context_encoder(x_context_input)[-1]
        with torch.no_grad():
            z_target = self.target_encoder(x_full)[-1].detach()

        latent_mask = (
            F.adaptive_avg_pool2d(high_res_mask.float(), output_size=z_context.shape[2:]) >= 0.5
        ).to(z_context.dtype)

        z_pred = self.predictor(z_context, latent_mask)
        return z_pred, z_target, latent_mask, z_context


class SegFormerAllMLPDecoder(nn.Module):
    def __init__(self, in_channels_list, embedding_dim: int = 256, num_classes: int = 10):
        super().__init__()
        self.linear_c4 = nn.Conv2d(in_channels_list[3], embedding_dim, kernel_size=1)
        self.linear_c3 = nn.Conv2d(in_channels_list[2], embedding_dim, kernel_size=1)
        self.linear_c2 = nn.Conv2d(in_channels_list[1], embedding_dim, kernel_size=1)
        self.linear_c1 = nn.Conv2d(in_channels_list[0], embedding_dim, kernel_size=1)

        self.linear_fuse = nn.Sequential(
            nn.Conv2d(embedding_dim * 4, embedding_dim, kernel_size=1, bias=False),
            nn.GroupNorm(1, embedding_dim),
            nn.ReLU(inplace=False),  # Out-of-place for clean Grad-CAM feature hooking
            nn.Dropout2d(p=0.1)
        )
        self.linear_pred = nn.Conv2d(embedding_dim, num_classes, kernel_size=1)

    def forward(self, features):
        c1, c2, c3, c4 = features[-4:]
        _c4 = F.interpolate(self.linear_c4(c4), size=c1.shape[2:], mode='bilinear', align_corners=False)
        _c3 = F.interpolate(self.linear_c3(c3), size=c1.shape[2:], mode='bilinear', align_corners=False)
        _c2 = F.interpolate(self.linear_c2(c2), size=c1.shape[2:], mode='bilinear', align_corners=False)
        _c1 = self.linear_c1(c1)

        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))
        return self.linear_pred(_c)


class TMLPN_Downstream_v1(nn.Module):
    def __init__(self, num_classes: int = 10, backbone_name: str = 'mit_b1', isolated_stem: bool = True):
        super().__init__()
        self.isolated_stem = isolated_stem
        self.context_encoder = smp.encoders.get_encoder(backbone_name, in_channels=3, weights=None)
        original_proj = self.context_encoder.patch_embed1.proj
        if self.isolated_stem:
            self.context_encoder.patch_embed1.proj = ModalityIsolatedPatchEmbed(
                original_proj, enable_dirac=False, enable_scse=False
            )
        else:
            self.context_encoder.patch_embed1.proj = NaiveEarlyFusionPatchEmbed(original_proj)

        encoder_channels = self.context_encoder.out_channels[-4:]
        self.decode_head = SegFormerAllMLPDecoder(
            in_channels_list=encoder_channels, embedding_dim=256, num_classes=num_classes
        )

    def forward(self, x_full: torch.Tensor, return_features: bool = False):
        features = self.context_encoder(x_full)
        logits = self.decode_head(features)
        logits_upsampled = F.interpolate(logits, size=x_full.shape[2:], mode='bilinear', align_corners=False)
        if return_features:
            return logits_upsampled, features[-4:]
        return logits_upsampled


class TMLPN_Downstream_v2(nn.Module):
    def __init__(self, num_classes: int = 10, backbone_name: str = 'mit_b1', isolated_stem: bool = True,
                 use_lora: bool = False, lora_r: int = 8, lora_alpha: int = 16):
        super().__init__()
        self.isolated_stem = isolated_stem
        self.context_encoder = smp.encoders.get_encoder(backbone_name, in_channels=3, weights=None)
        original_proj = self.context_encoder.patch_embed1.proj
        if self.isolated_stem:
            self.context_encoder.patch_embed1.proj = ModalityIsolatedPatchEmbed(
                original_proj, enable_dirac=False, enable_scse=False
            )
        else:
            self.context_encoder.patch_embed1.proj = NaiveEarlyFusionPatchEmbed(original_proj)

        if use_lora:
            apply_lora_to_mit(self.context_encoder, r=lora_r, alpha=lora_alpha)

        encoder_channels = self.context_encoder.out_channels[-4:]
        self.decode_head = SegFormerAllMLPDecoder(
            in_channels_list=encoder_channels, embedding_dim=256, num_classes=num_classes
        )

    def forward(self, x_full: torch.Tensor, return_features: bool = False):
        features = self.context_encoder(x_full)
        logits = self.decode_head(features)
        logits_upsampled = F.interpolate(logits, size=x_full.shape[2:], mode='bilinear', align_corners=False)
        if return_features:
            return logits_upsampled, features[-4:]
        return logits_upsampled


class TMLPN_Downstream_v3(nn.Module):
    def __init__(self, num_classes: int = 10, backbone_name: str = 'mit_b1', isolated_stem: bool = True,
                 enable_dirac: bool = True, enable_scse: bool = True,
                 use_lora: bool = False, lora_r: int = 8, lora_alpha: int = 16):
        super().__init__()
        self.isolated_stem = isolated_stem
        self.context_encoder = smp.encoders.get_encoder(backbone_name, in_channels=3, weights=None)

        original_proj = self.context_encoder.patch_embed1.proj
        if self.isolated_stem:
            self.context_encoder.patch_embed1.proj = ModalityIsolatedPatchEmbed(
                original_proj, enable_dirac=enable_dirac, enable_scse=enable_scse
            )
        else:
            self.context_encoder.patch_embed1.proj = NaiveEarlyFusionPatchEmbed(original_proj)

        if use_lora:
            apply_lora_to_mit(self.context_encoder, r=lora_r, alpha=lora_alpha)

        encoder_channels = self.context_encoder.out_channels[-4:]
        self.decode_head = SegFormerAllMLPDecoder(
            in_channels_list=encoder_channels, embedding_dim=256, num_classes=num_classes
        )

    def forward(self, x_full: torch.Tensor, return_features: bool = False):
        features = self.context_encoder(x_full)
        logits = self.decode_head(features)
        logits_upsampled = F.interpolate(logits, size=x_full.shape[2:], mode='bilinear', align_corners=False)
        if return_features:
            return logits_upsampled, features[-4:]
        return logits_upsampled


# ====================================================================================
# --- 5. CHECKPOINT-AWARE INSTANTIATION UTILITY ---
# ====================================================================================

def instantiate_model_from_checkpoint(ModelClass, num_classes: int, backbone: str,
                                      state_dict: dict, device: torch.device,
                                      strict: bool = True) -> nn.Module:
    is_isolated = any('rgb_proj' in k for k in state_dict.keys())
    has_scse = any('scse_gate' in k for k in state_dict.keys())
    has_lora = any('lora_A' in k for k in state_dict.keys())

    lora_r = 8
    if has_lora:
        for k, v in state_dict.items():
            if k.endswith('lora_A') and v.ndim == 2:
                lora_r = int(v.shape[1])
                break

    model_kwargs = {
        "num_classes": num_classes,
        "backbone_name": backbone,
        "isolated_stem": is_isolated,
    }
    if ModelClass.__name__ == "TMLPN_Downstream_v3":
        model_kwargs.update({
            "enable_dirac": True,
            "enable_scse": has_scse,
            "use_lora": has_lora,
            "lora_r": lora_r,
            "lora_alpha": lora_r * 2,
        })
    elif ModelClass.__name__ == "TMLPN_Downstream_v2":
        model_kwargs.update({
            "use_lora": has_lora,
            "lora_r": lora_r,
            "lora_alpha": lora_r * 2,
        })

    model = ModelClass(**model_kwargs).to(device)
    model.load_state_dict(state_dict, strict=strict)
    return model