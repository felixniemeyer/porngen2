import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiTimescaleState(nn.Module):
    def __init__(self, channels):
        super().__init__()
        # Input: z_t, temp_diff (2x channels), h_fast
        in_dim = channels * 4
        self.fast_gate = nn.Conv2d(in_dim, channels, 3, padding=1)
        self.slow_gate = nn.Conv2d(in_dim, channels, 3, padding=1)
        self.update = nn.Conv2d(in_dim, channels * 2, 3, padding=1)
        
    def forward(self, z, temp_diff, h_fast, h_slow, uncertainty):
        cat_in = torch.cat([z, temp_diff, h_fast], dim=1)
        
        # Gates
        gf = torch.sigmoid(self.fast_gate(cat_in))
        
        # SLOW MEMORY PROTECTION:
        # Heavily damp the slow gate (0.1), AND explicitly multiply by (1 - uncertainty).
        # If the model is highly uncertain (e.g. fast motion blur, occlusion), 
        # it cannot overwrite the slow identity/background memory.
        gs = torch.sigmoid(self.slow_gate(cat_in)) * 0.1 * (1.0 - uncertainty)
        
        updates = torch.tanh(self.update(cat_in))
        u_fast, u_slow = torch.chunk(updates, 2, dim=1)
        
        h_fast_new = (1 - gf) * h_fast + gf * u_fast
        h_slow_new = (1 - gs) * h_slow + gs * u_slow
        
        return h_fast_new, h_slow_new

class DynamicsCore(nn.Module):
    def __init__(self, in_channels, state_channels):
        super().__init__()
        # Input is z_t, h_fast, h_slow
        dim = in_channels + state_channels * 2
        self.net = nn.Sequential(
            nn.Conv2d(dim, 256, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(256, 3, 3, padding=1) # 2 for flow (dx, dy), 1 for uncertainty
        )
        # Initialize flow to zero
        nn.init.constant_(self.net[-1].weight, 0)
        nn.init.constant_(self.net[-1].bias, 0)
        
    def forward(self, z, h_fast, h_slow):
        x = torch.cat([z, h_fast, h_slow], dim=1)
        out = self.net(x)
        # Scale flow slightly to prevent wild early jumps
        flow = out[:, :2, :, :] * 0.1
        uncertainty = torch.sigmoid(out[:, 2:3, :, :])
        return flow, uncertainty

class SparseRefiner(nn.Module):
    def __init__(self, channels):
        super().__init__()
        # Takes warped draft, current observation (z_t) as skip, and uncertainty mask
        self.enc1 = nn.Conv2d(channels * 2 + 1, 128, 3, padding=1)
        self.enc2 = nn.Conv2d(128, 256, 3, stride=2, padding=1)
        self.dec1 = nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1)
        self.out = nn.Conv2d(128, channels, 3, padding=1)
        self.act = nn.GELU()
        
        # Zero-init the residual output so it relies on warp initially
        nn.init.constant_(self.out.weight, 0)
        nn.init.constant_(self.out.bias, 0)
        
    def forward(self, z_warped, z_t, uncertainty):
        x = torch.cat([z_warped, z_t, uncertainty], dim=1)
        e1 = self.act(self.enc1(x))
        e2 = self.act(self.enc2(e1))
        d1 = self.act(self.dec1(e2))
        res = self.out(d1 + e1) # Skip connection
        
        # Modulate residual strictly by predicted uncertainty
        return z_warped + (res * uncertainty)

class LatentEncoder(nn.Module):
    def __init__(self, out_channels=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, 4, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(64, out_channels, 4, stride=2, padding=1),
            nn.GELU()
        )
    def forward(self, x):
        return self.net(x)

class LatentDecoder(nn.Module):
    def __init__(self, in_channels=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(in_channels, 64, 4, stride=2, padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(64, 3, 4, stride=2, padding=1),
            nn.Tanh()
        )
    def forward(self, z):
        return self.net(z)

def warp_latent(z, flow):
    B, C, H, W = z.shape
    # Create normalized meshgrid in [-1, 1]
    xx = torch.linspace(-1.0, 1.0, W, device=z.device)
    yy = torch.linspace(-1.0, 1.0, H, device=z.device)
    grid_y, grid_x = torch.meshgrid(yy, xx, indexing='ij')
    grid = torch.stack((grid_x, grid_y), dim=0).unsqueeze(0).repeat(B, 1, 1, 1) # B, 2, H, W
    
    warp_grid = grid + flow
    warp_grid = warp_grid.permute(0, 2, 3, 1) # B, H, W, 2
    
    z_warped = F.grid_sample(z, warp_grid, mode='bilinear', padding_mode='reflection', align_corners=True)
    return z_warped

class LaRMS(nn.Module):
    """
    Latent Autoregressive Refinement & Memory Simulator.
    Strictly Markovian: X_t+1 = f(X_t, H_t)
    """
    def __init__(self, e_dim=128, cond_channels=32, latent_dim=128):
        super().__init__()
        self.latent_dim = latent_dim
        
        self.encoder = LatentEncoder(out_channels=latent_dim)
        self.decoder = LatentDecoder(in_channels=latent_dim)
        
        self.state_updater = MultiTimescaleState(channels=latent_dim)
        self.dynamics = DynamicsCore(in_channels=latent_dim, state_channels=latent_dim)
        self.refiner = SparseRefiner(channels=latent_dim)
        
        params = sum(p.numel() for p in self.parameters())
        print(f"Initialized LaRMS model with {params/1e6:.2f}M parameters.")
        
    def init_hidden(self, batch_size, height, width, device):
        # Assumes 4x spatial downsampling from the LatentEncoder
        h, w = height // 4, width // 4
        h_fast = torch.zeros(batch_size, self.latent_dim, h, w, device=device)
        h_slow = torch.zeros(batch_size, self.latent_dim, h, w, device=device)
        return (h_fast, h_slow)
        
    def forward(self, x_t, h_prev, e_t=None):
        h_fast, h_slow = h_prev
        
        # 1. Encode Observation
        z_t = self.encoder(x_t)
        
        # 2. Predict Dynamics from OLD State & Current Observation
        flow, uncertainty = self.dynamics(z_t, h_fast, h_slow)
        
        # 3. Latent Transport (Reprojection Draft for t+1)
        z_warped = warp_latent(z_t, flow)
        
        # 4. Sparse Refinement (with Uncertainty-Gated Stochasticity)
        if self.training:
            # Inject noise where uncertain to force the refiner to hallucinate structure
            noise = torch.randn_like(z_warped) * 0.05
            z_warped_noisy = z_warped + (noise * uncertainty)
        else:
            z_warped_noisy = z_warped
            
        z_t_next = self.refiner(z_warped_noisy, z_t, uncertainty)
        
        # 5. Temporal Difference / Surprise Signal
        # A much stronger error signal: how much we warped vs actual, and how much we refined vs warped
        temp_diff = torch.cat([z_t - z_warped, z_t_next - z_warped], dim=1)
        
        # 6. Update Multi-Timescale Memory
        h_fast_new, h_slow_new = self.state_updater(z_t, temp_diff, h_fast, h_slow, uncertainty)
        
        # 7. Decode
        x_t_next = self.decoder(z_t_next)
        
        return x_t_next, (h_fast_new, h_slow_new)
