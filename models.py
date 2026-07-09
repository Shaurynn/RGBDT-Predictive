import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp
import copy

class PositionalEncoding2D(nn.Module):
    """Returns pure 2D sinusoidal frequencies for concatenation conditioning."""
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2).float() / channels))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, tensor):
        B, _, H, W = tensor.shape
        pos_x = torch.arange(W, device=tensor.device).type(self.inv_freq.type())
        pos_y = torch.arange(H, device=tensor.device).type(self.inv_freq.type())

        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        sin_inp_y = torch.einsum("i,j->ij", pos_y, self.inv_freq)

        emb_x = torch.cat((sin_inp_x.sin(), sin_inp_x.cos()), dim=-1).unsqueeze(0).expand(H, W, -1)
        emb_y = torch.cat((sin_inp_y.sin(), sin_inp_y.cos()), dim=-1).unsqueeze(1).expand(H, W, -1)
        
        emb = (emb_x + emb_y).permute(2, 0, 1).unsqueeze(0).expand(B, -1, H, W)
        return emb

class SpatialJEPAPredictor(nn.Module):
    """
    Explicitly conditions the CNN inference by concatenating the context embedding, 
    the target mask, and the positional coordinates.
    """
    def __init__(self, embed_dim, hidden_dim=1024):
        super().__init__()
        self.pos_embed = PositionalEncoding2D(embed_dim)
        # Input channels: Context (embed_dim) + Mask (1) + Positional (embed_dim)
        in_channels = embed_dim * 2 + 1
        
        self.predictor = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim, kernel_size=3, padding=1, bias=False),
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

    def forward(self, context_embedding, latent_mask):
        pos_encoding = self.pos_embed(context_embedding)
        # Concatenation formally ensures conditional target prediction
        x = torch.cat([context_embedding, latent_mask, pos_encoding], dim=1)
        return self.predictor(x)

class MultimodalJEPA(nn.Module):
    def __init__(self, backbone_name='mit_b1'):
        super().__init__()
        self.context_encoder = smp.encoders.get_encoder(backbone_name, in_channels=5, weights='imagenet')
        self.target_encoder = copy.deepcopy(self.context_encoder)
        
        for p in self.target_encoder.parameters(): p.requires_grad = False
        self.predictor = SpatialJEPAPredictor(embed_dim=self.context_encoder.out_channels[-1])

    @torch.no_grad()
    def update_target_network(self, tau=0.996):
        for ctx_p, tgt_p in zip(self.context_encoder.parameters(), self.target_encoder.parameters()):
            tgt_p.data = tau * tgt_p.data + (1.0 - tau) * ctx_p.data

    def forward(self, x_visible, x_full, high_res_mask):
        z_context = self.context_encoder(x_visible)[-1]
        
        with torch.no_grad():
            z_target = self.target_encoder(x_full)[-1]
            
        # Downsample the mask to provide positional context to the predictor
        B, C, H, W = z_context.shape
        latent_mask = F.interpolate(high_res_mask, size=(H, W), mode='nearest')
        
        z_pred = self.predictor(z_context, latent_mask)
        return z_pred, z_target, latent_mask

class TMLPN_Downstream(nn.Module):
    def __init__(self, num_classes=10, backbone_name='mit_b1'):
        super().__init__()
        self.context_encoder = smp.encoders.get_encoder(backbone_name, in_channels=5, weights=None)
        final_dim = self.context_encoder.out_channels[-1]
        
        self.decode_head = nn.Sequential(
            nn.Conv2d(final_dim, final_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(final_dim),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1),
            nn.Conv2d(final_dim, num_classes, kernel_size=1)
        )

    def forward(self, x_full):
        features = self.context_encoder(x_full)[-1]
        logits = self.decode_head(features)
        return F.interpolate(logits, size=x_full.shape[2:], mode='bilinear', align_corners=False)