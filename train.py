import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import warnings

# Suppress the torchvision pretrained vs weights warning originating from LPIPS
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision.models._utils")

# Import our custom modules
from world_model import WorldModelEngine
from data_pipeline import WorldModelDataset, run_labelling_process

def main():
    # 1. Ensure labels/embeddings exist
    print("Checking dataset...")
    run_labelling_process(video_dir='./train-videos', output_dir='./train-embeddings')

    # 2. Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    batch_size = 4
    seq_len = 256
    epochs = 1
    learning_rate = 1e-4
    
    # Model dimensions
    e_dim = 128
    cond_channels = 32
    latent_dim = 256
    
    # 3. Setup DataLoader
    dataset = WorldModelDataset(
        video_dir='./train-videos', 
        embed_dir='./train-embeddings', 
        seq_len=seq_len, 
        height=32, 
        width=64
    )
    
    if len(dataset.samples) == 0:
        print("No videos found! Please place some .mp4 files in ./train-videos and re-run.")
        return

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    
    # 4. Initialize Model
    model = WorldModelEngine(
        e_dim=e_dim, 
        cond_channels=cond_channels, 
        latent_dim=latent_dim
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion_l1 = nn.L1Loss()
    
    # Attempt to load LPIPS for perceptual loss
    try:
        import lpips
        criterion_perceptual = lpips.LPIPS(net='vgg').to(device)
        criterion_perceptual.eval() # Keep VGG weights frozen
        use_perceptual = True
        print("LPIPS loaded successfully.")
    except ImportError:
        print("LPIPS not installed. Falling back to L1 loss only. (pip install lpips)")
        use_perceptual = False

    os.makedirs("checkpoints", exist_ok=True)
    
    # 5. Training Loop
    print("Starting training...")
    model.train()
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        
        # We use tqdm for a nice progress bar
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch_idx, (video_chunk, embed_chunk) in enumerate(pbar):
            # video_chunk: (B, seq_len + 1, C, H, W)
            # embed_chunk: (B, seq_len + 1, e_dim)
            video_chunk = video_chunk.to(device)
            embed_chunk = embed_chunk.to(device)
            
            B = video_chunk.shape[0]
            
            # Initial hidden state for the BPTT chunk (zeros)
            # Shape matches the latent space output of the encoder: 8x16
            h_prev = torch.zeros(B, latent_dim, 8, 16).to(device)
            
            optimizer.zero_grad()
            total_bptt_loss = 0.0
            
            # Unroll the sequence
            for t in range(seq_len):
                x_t = video_chunk[:, t]
                x_target = video_chunk[:, t+1]
                e_t = embed_chunk[:, t]
                
                # Noise Injection / Denoising Objective
                # With 15% probability, we replace the input frame with pure noise.
                # This breaks the "pixel copier" shortcut and forces the model's Residual Map 
                # and Hidden State to learn to synthesize the scene from memory.
                if torch.rand(1).item() < 0.15:
                    # Generate noise in [-1, 1] range
                    x_t_input = (torch.rand_like(x_t) * 2.0) - 1.0
                else:
                    x_t_input = x_t
                
                # Forward pass
                x_pred, h_prev = model(x_t_input, h_prev, e_t)
                
                # Calculate Losses
                loss_l1 = criterion_l1(x_pred, x_target)
                step_loss = loss_l1
                
                if use_perceptual:
                    loss_p = criterion_perceptual(x_pred, x_target).mean()
                    step_loss = step_loss + 0.5 * loss_p
                
                total_bptt_loss += step_loss
                
            # Backpropagate through time
            total_bptt_loss.backward()
            optimizer.step()
            
            # 💥 CRUCIAL: Detach hidden state to prevent OOM errors in the next batch!
            h_prev = h_prev.detach()
            
            # Memory logging
            import resource
            mem_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
            
            # Logging
            avg_loss = total_bptt_loss.item() / seq_len
            epoch_loss += avg_loss
            pbar.set_postfix({'loss': f"{avg_loss:.4f}", 'mem_MB': f"{mem_mb:.1f}"})
            
        print(f"Epoch {epoch+1} Average Loss: {epoch_loss / len(dataloader):.4f}")
        
        # Save checkpoint periodically
        if (epoch + 1) % 1 == 0:
            ckpt_path = f"checkpoints/world_model_ep{epoch+1}.pt"
            torch.save(model.state_dict(), ckpt_path)
            print(f"Saved checkpoint to {ckpt_path}")

if __name__ == "__main__":
    main()
