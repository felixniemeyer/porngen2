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
    Uses 2 layers of stride-2 convolutions.
    """
    def __init__(self, in_channels=3, latent_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, latent_dim, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        return self.net(x)


class Decoder(nn.Module):
    """
    Phase 3: Plans the next frame by upsampling back to 32x64.
    Outputs both a coarse residual map and an offset field for the DCN.
    """
    def __init__(self, hidden_dim=256, out_channels=3, kernel_size=3):
        super().__init__()
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(hidden_dim, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.to_residual = nn.Conv2d(64, out_channels, kernel_size=3, padding=1)
        
        # Offset field needs 2 * kernel_size^2 channels (dx, dy for each spatial weight)
        self.to_offset = nn.Conv2d(64, 2 * kernel_size * kernel_size, kernel_size=3, padding=1)
        
        # Initialize offset to 0 so the DCN starts as a standard convolution initially
        nn.init.constant_(self.to_offset.weight, 0)
        nn.init.constant_(self.to_offset.bias, 0)
        
        # Zero-Initialization for residual map to enforce reliance on DCN early in training
        nn.init.constant_(self.to_residual.weight, 0.0)
        nn.init.constant_(self.to_residual.bias, 0.0)

    def forward(self, h):
        feat = self.upsample(h)
        residual = self.to_residual(feat)
        offset = self.to_offset(feat)
        return residual, offset


class WorldModelEngine(nn.Module):
    """
    The full Predictive Latent World Model.
    Orchestrates the Encoder, Memory Core, Decoder, and DCN.
    """
    def __init__(self, e_dim, cond_channels, latent_dim=256, dcn_kernel_size=3):
        super().__init__()
        self.latent_dim = latent_dim
        self.cond_channels = cond_channels
        self.dcn_kernel_size = dcn_kernel_size
        
        self.encoder = Encoder(in_channels=3, latent_dim=latent_dim)
        
        # Projects Scene_Embedding E_t into spatial dimensions
        self.cond_proj = nn.Linear(e_dim, cond_channels)
        
        # Phase 2: Memory Core, now conditioned on user control signal
        self.memory = ConvGRUCell(
            input_dim=latent_dim + cond_channels, 
            hidden_dim=latent_dim, 
            kernel_size=3
        )
        
        self.decoder = Decoder(hidden_dim=latent_dim, out_channels=3, kernel_size=dcn_kernel_size)
        
        # DCN Weight: acts on the raw frame (3 channels in -> 3 channels out)
        self.dcn_weight = nn.Parameter(torch.zeros(3, 3, dcn_kernel_size, dcn_kernel_size))
        self.dcn_bias = nn.Parameter(torch.zeros(3))
        
        # Identity initialization: center pixel = 1 for corresponding channels
        # Ensures a "pure" nearest-neighbor pixel shift initially.
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
        # Map E_t to channels, then broadcast spatially to match z_t
        e_cond = self.cond_proj(e_t)  # (B, cond_channels)
        e_spatial = e_cond.view(B, self.cond_channels, 1, 1).expand(-1, -1, H_z, W_z)
        
        # Concatenate visuals and control signal
        z_cond = torch.cat([z_t, e_spatial], dim=1)
        
        # Update hidden state physics
        h_t = self.memory(z_cond, h_prev)
        
        # Phase 3: Decode to get plans
        residual_map, offset_field = self.decoder(h_t)
        
        # Phase 4: Deformable Skip-Connection (Detail Retriever)
        # Warps x_t using the offsets from the World Model
        x_warped = ops.deform_conv2d(
            input=x_t, 
            offset=offset_field, 
            weight=self.dcn_weight, 
            bias=self.dcn_bias, 
            padding=self.dcn_kernel_size // 2
        )
        
        # Final predicted frame
        x_t_plus_1 = x_warped + residual_map
        
        return x_t_plus_1, h_t


# =====================================================================
# ⚙️ Training Paradigm / Example Loop
# =====================================================================

def simulate_training_loop():
    try:
        import lpips
    except ImportError:
        print("Please install lpips: pip install lpips")
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Hyperparameters
    B = 2             # Batch size
    seq_len = 16      # BPTT chunk size
    C, H, W = 3, 32, 64 # Updated resolution to 64x32
    e_dim = 128       # Scene control vector dimension
    cond_channels = 32
    latent_dim = 256
    
    # Initialize components
    model = WorldModelEngine(
        e_dim=e_dim, 
        cond_channels=cond_channels, 
        latent_dim=latent_dim
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion_l1 = nn.L1Loss()
    
    # Perceptual Loss (requires inputs scaled between [-1, 1])
    criterion_perceptual = lpips.LPIPS(net='vgg').to(device)
    criterion_perceptual.eval() # freeze VGG
    
    # Mock Data: [B, seq_len + 1, C, H, W] for next-frame prediction
    video_chunk = torch.rand(B, seq_len + 1, C, H, W).to(device) * 2 - 1 # [-1, 1] range
    scene_embeddings = torch.randn(B, seq_len, e_dim).to(device)
    
    # Initial hidden state (all zeros) - downsampled size is 8x16
    h_prev = torch.zeros(B, latent_dim, 8, 16).to(device)
    
    model.train()
    optimizer.zero_grad()
    
    total_loss = 0.0
    
    print("Starting BPTT over 16 frames...")
    for t in range(seq_len):
        x_t = video_chunk[:, t]
        x_target = video_chunk[:, t+1]
        e_t = scene_embeddings[:, t]
        
        # Forward pass for step t
        x_pred, h_prev = model(x_t, h_prev, e_t)
        
        # Losses
        loss_l1 = criterion_l1(x_pred, x_target)
        loss_p = criterion_perceptual(x_pred, x_target).mean()
        
        step_loss = loss_l1 + 0.5 * loss_p # arbitrary weighting
        total_loss = total_loss + step_loss
        
    # Backpropagate through time
    total_loss.backward()
    optimizer.step()
    
    # 💥 CRUCIAL: Detach hidden state to prevent OOM errors in the next chunk!
    h_prev = h_prev.detach()
    
    print(f"Training chunk complete. Total Loss: {total_loss.item():.4f}")
    print(f"Hidden state successfully detached. Shape: {h_prev.shape}")

if __name__ == '__main__':
    # simulate_training_loop()
    pass
