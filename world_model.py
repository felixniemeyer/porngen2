import torch
import torch.nn as nn
import torchvision.ops as ops
import math

class ConvGRUCell(nn.Module):
    """
    Spatially-aware recurrent memory core.
    """
    def __init__(self, input_dim, hidden_dim, kernel_size, bias=True):
        super().__init__()
        self.hidden_dim = hidden_dim
        padding = kernel_size // 2
        
        self.conv_gates = nn.Conv2d(input_dim + hidden_dim, 2 * hidden_dim, kernel_size, padding=padding, bias=bias)
        self.conv_cand = nn.Conv2d(input_dim + hidden_dim, hidden_dim, kernel_size, padding=padding, bias=bias)
        
    def forward(self, x, h_prev):
        combined = torch.cat([x, h_prev], dim=1)
        gates = self.conv_gates(combined)
        
        # Split into reset and update gates
        reset_gate, update_gate = torch.chunk(gates, 2, dim=1)
        reset_gate = torch.sigmoid(reset_gate)
        update_gate = torch.sigmoid(update_gate)
        
        combined_cand = torch.cat([x, reset_gate * h_prev], dim=1)
        cand = torch.tanh(self.conv_cand(combined_cand))
        
        h_new = (1 - update_gate) * h_prev + update_gate * cand
        return h_new


class Encoder(nn.Module):
    """
    Phase 1: Compresses the 32x64 spatial resolution down to 8x16.
    Scalable via base_channels.
    """
    def __init__(self, in_channels=3, base_channels=64, latent_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels * 2, latent_dim, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        return self.net(x)


class Decoder(nn.Module):
    """
    Phase 3: Plans the next frame by upsampling back to 32x64.
    Scalable via base_channels.
    """
    def __init__(self, latent_dim=256, base_channels=64, out_channels=3, kernel_size=3):
        super().__init__()
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, base_channels * 2, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.to_residual = nn.Conv2d(base_channels, out_channels, kernel_size=3, padding=1)
        
        # Offset field needs 2 * kernel_size^2 channels
        self.to_offset = nn.Conv2d(base_channels, 2 * kernel_size * kernel_size, kernel_size=3, padding=1)
        
        # Initialize offset to 0
        nn.init.constant_(self.to_offset.weight, 0)
        nn.init.constant_(self.to_offset.bias, 0)
        
        # Zero-Initialization for residual
        nn.init.constant_(self.to_residual.weight, 0.0)
        nn.init.constant_(self.to_residual.bias, 0.0)

    def forward(self, h):
        feat = self.upsample(h)
        residual = self.to_residual(feat)
        offset = self.to_offset(feat)
        return residual, offset


class LatentDCNWorldModel(nn.Module):
    """
    Predictive Latent World Model with Deformable Skip-Connections.
    Fully parameterizable capacity.
    """
    def __init__(self, e_dim, cond_channels, base_channels=64, latent_dim=256, dcn_kernel_size=3):
        super().__init__()
        self.latent_dim = latent_dim
        self.cond_channels = cond_channels
        self.dcn_kernel_size = dcn_kernel_size
        
        self.encoder = Encoder(in_channels=3, base_channels=base_channels, latent_dim=latent_dim)
        
        # Projects Scene_Embedding E_t into spatial dimensions
        self.cond_proj = nn.Linear(e_dim, cond_channels)
        
        # Phase 2: Memory Core
        self.memory = ConvGRUCell(
            input_dim=latent_dim + cond_channels, 
            hidden_dim=latent_dim, 
            kernel_size=3
        )
        
        self.decoder = Decoder(latent_dim=latent_dim, base_channels=base_channels, out_channels=3, kernel_size=dcn_kernel_size)
        
        # DCN Weight: acts on the raw frame
        self.dcn_weight = nn.Parameter(torch.zeros(3, 3, dcn_kernel_size, dcn_kernel_size))
        self.dcn_bias = nn.Parameter(torch.zeros(3))
        
        # Identity initialization
        center = dcn_kernel_size // 2
        with torch.no_grad():
            for i in range(3):
                self.dcn_weight[i, i, center, center] = 1.0

    def forward(self, x_t, h_prev, e_t):
        B, C, H, W = x_t.shape
        
        # Phase 1: Encode raw frame
        z_t = self.encoder(x_t)
        _, _, H_z, W_z = z_t.shape
        
        # Phase 2: Conditioned Memory
        e_cond = self.cond_proj(e_t)
        e_spatial = e_cond.view(B, self.cond_channels, 1, 1).expand(-1, -1, H_z, W_z)
        
        z_cond = torch.cat([z_t, e_spatial], dim=1)
        h_t = self.memory(z_cond, h_prev)
        
        # Phase 3: Decode
        residual_map, offset_field = self.decoder(h_t)
        
        # Phase 4: DCN
        x_warped = ops.deform_conv2d(
            input=x_t, 
            offset=offset_field, 
            weight=self.dcn_weight, 
            bias=self.dcn_bias, 
            padding=self.dcn_kernel_size // 2
        )
        
        x_t_plus_1 = x_warped + residual_map
        return x_t_plus_1, h_t
