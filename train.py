import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import warnings
import mlflow

# Suppress the torchvision pretrained vs weights warning originating from LPIPS
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision.models._utils")

# Import our custom modules
from world_model import WorldModelEngine
from data_pipeline import WorldModelDataset, run_labelling_process

def parse_args():
    parser = argparse.ArgumentParser(description="Train the Predictive Latent World Model")
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--seq_len', type=int, default=256, help='Sequence length for BPTT')
    parser.add_argument('--epochs', type=int, default=1, help='Number of epochs to train for in this run')
    parser.add_argument('--start_epoch', type=int, default=0, help='Starting epoch number for checkpoint naming (e.g., set to 1 if resuming from ep1)')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--e_dim', type=int, default=128, help='Embedding dimension')
    parser.add_argument('--cond_channels', type=int, default=32, help='Condition channels')
    parser.add_argument('--latent_dim', type=int, default=256, help='Latent dimension')
    parser.add_argument('--degrade_prob', type=float, default=0.05, help='Probability of applying a collapse mode at each step')
    parser.add_argument('--experiment_name', type=str, default='WorldModel_Training', help='MLflow experiment name')
    parser.add_argument('--resume_from', type=str, default=None, help='Path to checkpoint to resume training from')
    return parser.parse_args()

def apply_degradation(x_t):
    """Randomly applies one of several collapse modes to the input frame."""
    mode = torch.randint(0, 5, (1,)).item()
    B, C, H, W = x_t.shape
    device = x_t.device

    if mode == 0: # Pure Noise
        return (torch.rand_like(x_t) * 2.0) - 1.0
    
    elif mode == 1: # Solid Color
        color = (torch.rand(B, 3, 1, 1, device=device) * 2.0) - 1.0
        return color.expand(-1, -1, H, W)
    
    elif mode == 2: # Spatial Gradient
        color_a = (torch.rand(B, 3, 1, 1, device=device) * 2.0) - 1.0
        color_b = (torch.rand(B, 3, 1, 1, device=device) * 2.0) - 1.0
        # Create a simple vertical gradient
        lin = torch.linspace(0, 1, H, device=device).view(1, 1, H, 1)
        return color_a * lin + color_b * (1 - lin)
    
    elif mode == 3: # Extreme Contrast / Binary
        return (x_t * 10.0).clamp(-1.0, 1.0)
    
    else: # Heavy Blur
        from torchvision.transforms.functional import gaussian_blur
        return gaussian_blur(x_t, kernel_size=[15, 15], sigma=[5.0, 5.0])

def main():
    args = parse_args()

    # 1. Ensure labels/embeddings exist
    print("Checking dataset...")
    run_labelling_process(video_dir='./train-videos', output_dir='./train-embeddings')

    # 2. Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 3. Setup DataLoader
    dataset = WorldModelDataset(
        video_dir='./train-videos', 
        embed_dir='./train-embeddings', 
        seq_len=args.seq_len, 
        height=32, 
        width=64
    )
    
    if len(dataset.samples) == 0:
        print("No videos found! Please place some .mp4 files in ./train-videos and re-run.")
        return

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    
    # 4. Initialize Model
    model = WorldModelEngine(
        e_dim=args.e_dim, 
        cond_channels=args.cond_channels, 
        latent_dim=args.latent_dim
    ).to(device)
    
    if args.resume_from:
        if os.path.exists(args.resume_from):
            print(f"Resuming training from checkpoint: {args.resume_from}")
            model.load_state_dict(torch.load(args.resume_from, map_location=device, weights_only=True))
        else:
            print(f"Warning: Checkpoint {args.resume_from} not found. Starting from scratch.")
            
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
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
    
    # Setup MLflow
    mlflow.set_experiment(args.experiment_name)
    with mlflow.start_run():
        # Log parameters
        mlflow.log_params(vars(args))
        mlflow.log_param("device", str(device))
        mlflow.log_param("use_perceptual_loss", use_perceptual)

        # 5. Training Loop
        print("Starting training...")
        model.train()
        
        global_step = 0
        
        for epoch in range(args.epochs):
            current_epoch = args.start_epoch + epoch + 1
            epoch_loss = 0.0
            
            # We use tqdm for a nice progress bar
            pbar = tqdm(dataloader, desc=f"Epoch {current_epoch}/{args.start_epoch + args.epochs}")
            for batch_idx, (video_chunk, embed_chunk) in enumerate(pbar):
                video_chunk = video_chunk.to(device)
                embed_chunk = embed_chunk.to(device)
                
                B = video_chunk.shape[0]
                h_prev = torch.zeros(B, args.latent_dim, 8, 16).to(device)
                
                optimizer.zero_grad()
                total_bptt_loss = 0.0
                
                # Unroll the sequence
                for t in range(args.seq_len):
                    x_t = video_chunk[:, t]
                    x_target = video_chunk[:, t+1]
                    e_t = embed_chunk[:, t]
                    
                    # Diverse Collapse & Recovery Objective
                    if torch.rand(1).item() < args.degrade_prob:
                        x_t_input = apply_degradation(x_t)
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
                avg_loss = total_bptt_loss.item() / args.seq_len
                epoch_loss += avg_loss
                pbar.set_postfix({'loss': f"{avg_loss:.4f}", 'mem_MB': f"{mem_mb:.1f}"})
                
                # MLflow step logging
                mlflow.log_metric("step_loss", avg_loss, step=global_step)
                mlflow.log_metric("mem_mb", mem_mb, step=global_step)
                global_step += 1
                
            avg_epoch_loss = epoch_loss / len(dataloader)
            print(f"Epoch {current_epoch} Average Loss: {avg_epoch_loss:.4f}")
            mlflow.log_metric("epoch_loss", avg_epoch_loss, step=current_epoch)
            
            # Save checkpoint periodically
            if current_epoch % 1 == 0:
                ckpt_path = f"checkpoints/world_model_ep{current_epoch}.pt"
                torch.save(model.state_dict(), ckpt_path)
                print(f"Saved checkpoint to {ckpt_path}")

if __name__ == "__main__":
    main()
