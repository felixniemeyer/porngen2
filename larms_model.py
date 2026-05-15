import torch
import torch.nn as nn
import torch.nn.functional as F

def get_norm(channels):
    """Utility to create a GroupNorm layer for stabilization."""
    return nn.GroupNorm(num_groups=min(32, channels), num_channels=channels)

class MultiTimescaleState(nn.Module):
    def __init__(self, channels):
        super().__init__()
        # Input: z_t, temp_diff (2x channels), h_fast
        in_dim = channels * 4
        self.norm = get_norm(in_dim)
        self.fast_gate = nn.Conv2d(in_dim, channels, 3, padding=1)
        self.slow_gate = nn.Conv2d(in_dim, channels, 3, padding=1)
        self.update = nn.Conv2d(in_dim, channels * 2, 3, padding=1)
        
    def forward(self, z, temp_diff, h_fast, h_slow, uncertainty):
        cat_in = torch.cat([z, temp_diff, h_fast], dim=1)
        cat_in = self.norm(cat_in) # Keep inputs centered
        
        gf = torch.sigmoid(self.fast_gate(cat_in))
        # Protect slow memory from uncertain pixels
        gs = torch.sigmoid(self.slow_gate(cat_in)) * 0.1 * (1.0 - uncertainty)
        
        updates = torch.tanh(self.update(cat_in))
        u_fast, u_slow = torch.chunk(updates, 2, dim=1)
        
        h_fast_new = (1 - gf) * h_fast + gf * u_fast
        h_slow_new = (1 - gs) * h_slow + gs * u_slow
        
        # --- ACTIVE DRIFT PREVENTION ---
        # State decay prevents numerical 'ghosts' from accumulating into snakes
        return h_fast_new * 0.999, h_slow_new * 0.999

class DynamicsCore(nn.Module):
    def __init__(self, in_channels, state_channels):
        super().__init__()
        dim = in_channels + state_channels * 2
        self.net = nn.Sequential(
            nn.Conv2d(dim, 256, 3, padding=1),
            get_norm(256),
            nn.GELU(),
            nn.Conv2d(256, 256, 3, padding=1),
            get_norm(256),
            nn.GELU(),
            nn.Conv2d(256, 3, 3, padding=1)
        )
        nn.init.constant_(self.net[-1].weight, 0)
        nn.init.constant_(self.net[-1].bias, 0)
        
    def forward(self, z, h_fast, h_slow):
        x = torch.cat([z, h_fast, h_slow], dim=1)
        out = self.net(x)
        # 4x reduction in flow strength for more subtle, realistic motion
        flow = out[:, :2, :, :] * 0.025
        # Sharpen the uncertainty mask to force more 'binary' choices (sharp vs generated)
        uncertainty = torch.sigmoid(out[:, 2:3, :, :] * 2.0)
        return flow, uncertainty

class SparseRefiner(nn.Module):
    def __init__(self, channels):
        super().__init__()
        # Input: warped, skip(z_t), uncertainty
        self.enc1 = nn.Conv2d(channels * 2 + 1, 128, 3, padding=1)
        self.norm1 = get_norm(128)
        self.enc2 = nn.Conv2d(128, 256, 3, stride=2, padding=1)
        self.norm2 = get_norm(256)
        
        # Bilinear Upsampling + Conv instead of Transposed Conv (Anti-Snake)
        self.dec1 = nn.Conv2d(256, 128, 3, padding=1)
        self.norm3 = get_norm(128)
        
        self.out = nn.Conv2d(128, channels, 3, padding=1)
        self.act = nn.GELU()
        nn.init.constant_(self.out.weight, 0)
        nn.init.constant_(self.out.bias, 0)
        
    def forward(self, z_warped, z_t, uncertainty):
        x = torch.cat([z_warped, z_t, uncertainty], dim=1)
        e1 = self.act(self.norm1(self.enc1(x)))
        e2 = self.act(self.norm2(self.enc2(e1)))
        
        d1 = F.interpolate(e2, scale_factor=2, mode='bilinear', align_corners=True)
        d1 = self.act(self.norm3(self.dec1(d1)))
        
        res = self.out(d1 + e1)
        return z_warped + (res * uncertainty)

class LatentEncoder(nn.Module):
    def __init__(self, out_channels=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, 4, stride=2, padding=1),
            get_norm(64),
            nn.GELU(),
            nn.Conv2d(64, out_channels, 4, stride=2, padding=1),
            get_norm(out_channels),
            nn.GELU()
        )
    def forward(self, x):
        return self.net(x)

class LatentDecoder(nn.Module):
    def __init__(self, in_channels=128):
        super().__init__()
        # 8x16 -> 16x32 -> 32x64
        self.conv1 = nn.Conv2d(in_channels, 64, 3, padding=1)
        self.norm1 = get_norm(64)
        self.conv2 = nn.Conv2d(64, 3, 3, padding=1)
        
    def forward(self, z):
        x = F.interpolate(z, scale_factor=2, mode='bilinear', align_corners=True)
        x = F.gelu(self.norm1(self.conv1(x)))
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        return torch.tanh(self.conv2(x))

def warp_latent(z, flow):
    B, C, H, W = z.shape
    xx = torch.linspace(-1.0, 1.0, W, device=z.device)
    yy = torch.linspace(-1.0, 1.0, H, device=z.device)
    grid_y, grid_x = torch.meshgrid(yy, xx, indexing='ij')
    grid = torch.stack((grid_x, grid_y), dim=0).unsqueeze(0).repeat(B, 1, 1, 1)
    warp_grid = (grid + flow).permute(0, 2, 3, 1)
    return F.grid_sample(z, warp_grid, mode='bilinear', padding_mode='reflection', align_corners=True)

class LaRMS(nn.Module):
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
        h, w = height // 4, width // 4
        h_fast = torch.zeros(batch_size, self.latent_dim, h, w, device=device)
        h_slow = torch.zeros(batch_size, self.latent_dim, h, w, device=device)
        return (h_fast, h_slow)
        
    def forward(self, x_t, h_prev, e_t=None):
        h_fast, h_slow = h_prev
        z_t = self.encoder(x_t)
        
        flow, uncertainty = self.dynamics(z_t, h_fast, h_slow)
        # Center the flow spatially to prevent cumulative zoom/drift
        flow = flow - flow.mean(dim=(2, 3), keepdim=True)
        
        z_warped = warp_latent(z_t, flow)
        
        if self.training:
            # Inject noise where uncertain to force the refiner to hallucinate structure
            # 4x reduction in noise strength to match the scaled-back motion
            noise = torch.randn_like(z_warped) * 0.0125
            z_warped_noisy = z_warped + (noise * uncertainty)
        else:
            z_warped_noisy = z_warped
            
        z_t_next = self.refiner(z_warped_noisy, z_t, uncertainty)
        temp_diff = torch.cat([z_t - z_warped, z_t_next - z_warped], dim=1)
        h_fast_new, h_slow_new = self.state_updater(z_t, temp_diff, h_fast, h_slow, uncertainty)
        x_t_next = self.decoder(z_t_next)
        
        # For debugging/visualization, return internal states
        debug_info = {
            'uncertainty': uncertainty,
            'flow': flow,
            'h_fast_mag': h_fast_new.abs().mean(dim=1, keepdim=True),
            'h_slow_mag': h_slow_new.abs().mean(dim=1, keepdim=True)
        }
        
        return x_t_next, (h_fast_new, h_slow_new), debug_info

    def load_migrated(self, path, device):
        """Attempts to load weights even if architecture changed by slicing tensors."""
        ckpt_sd = torch.load(path, map_location=device, weights_only=True)
        model_sd = self.state_dict()
        
        migrated_sd = {}
        salvage_count = 0
        for k, v_ckpt in ckpt_sd.items():
            if k in model_sd:
                v_model = model_sd[k]
                if v_ckpt.shape == v_model.shape:
                    migrated_sd[k] = v_ckpt
                    salvage_count += 1
                elif v_ckpt.ndim == v_model.ndim:
                    # OPTIMISTIC SLICING: Copy the overlapping region for same-rank tensors
                    new_v = v_model.clone()
                    slices = tuple(slice(0, min(v_ckpt.shape[i], v_model.shape[i])) for i in range(v_model.ndim))
                    new_v[slices] = v_ckpt[slices]
                    migrated_sd[k] = new_v
                    salvage_count += 1
        
        missing, unexpected = self.load_state_dict(migrated_sd, strict=False)
        print(f"LaRMS Optimistic Migration: Salvaged {salvage_count} layers (loaded or sliced).")
        if missing:
            print(f" -> {len(missing)} keys still missing (brand new layers).")
