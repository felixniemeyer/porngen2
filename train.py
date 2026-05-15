import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import warnings
import mlflow
import cv2
import numpy as np

# Suppress the torchvision pretrained vs weights warning originating from LPIPS
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision.models._utils")

# Import our custom modules
from world_model import LatentDCNWorldModel
from data_pipeline import WorldModelDataset, run_labelling_process

def parse_args():
    parser = argparse.ArgumentParser(description="Train the Predictive Latent World Model")
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--seq_len', type=int, default=256, help='Sequence length for BPTT')
    parser.add_argument('--epochs', type=int, default=1, help='Number of epochs to train for in this run')
    parser.add_argument('--start_epoch', type=int, default=0, help='Starting epoch number for checkpoint naming')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--lr_decay', type=float, default=0.96, help='Learning rate decay per epoch')
    parser.add_argument('--e_dim', type=int, default=128, help='Embedding dimension')
    parser.add_argument('--cond_channels', type=int, default=32, help='Condition channels')
    parser.add_argument('--base_channels', type=int, default=64, help='Base channels for Encoder/Decoder scaling')
    parser.add_argument('--latent_dim', type=int, default=256, help='Latent dimension')
    parser.add_argument('--samples_per_video', type=int, default=100, help='Number of random chunks to sample per video per epoch')
    parser.add_argument('--degrade_prob', type=float, default=0.05, help='Probability of applying a collapse mode at each step')
    parser.add_argument('--sample_interval', type=int, default=50, help='Steps between saving/showing visual samples')
    parser.add_argument('--show_preview', action='store_true', help='If set, shows a live preview window during training')
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
        lin = torch.linspace(0, 1, H, device=device).view(1, 1, H, 1)
        return (color_a * lin + color_b * (1 - lin)).expand(-1, -1, -1, W)
    elif mode == 3: # Extreme Contrast
        return (x_t * 10.0).clamp(-1.0, 1.0)
    else: # Heavy Blur
        from torchvision.transforms.functional import gaussian_blur
        return gaussian_blur(x_t, kernel_size=[15, 15], sigma=[5.0, 5.0])

def tensor_to_cv2(t):
    """Converts a [-1, 1] tensor to a [0, 255] numpy BGR image."""
    img = t.detach().cpu().squeeze(0).clamp(-1, 1)
    img = ((img + 1.0) * 127.5).byte().permute(1, 2, 0).numpy()
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

def save_and_show_sample(x_input, x_target, x_pred, step, epoch, show_preview=False):
    """Saves a comparison strip and optionally shows it in a window."""
    inp_img = tensor_to_cv2(x_input[0:1])
    tgt_img = tensor_to_cv2(x_target[0:1])
    prd_img = tensor_to_cv2(x_pred[0:1])
    
    # Concatenate horizontally
    strip = np.hstack([inp_img, tgt_img, prd_img])
    
    # Scale up for visibility
    h, w, _ = strip.shape
    preview = cv2.resize(strip, (w * 4, h * 4), interpolation=cv2.INTER_NEAREST)
    
    # Add labels
    cv2.putText(preview, f"Input | Target | Pred (Step {step})", (10, 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    # Save to disk
    os.makedirs("train_samples", exist_ok=True)
    cv2.imwrite(f"train_samples/sample_ep{epoch}_step{step}.png", preview)
    
    if show_preview:
        cv2.imshow("Training Preview", preview)
        cv2.waitKey(1) # Refresh window

def main():
    args = parse_args()

    # 1. Ensure labels/embeddings exist
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
        width=64,
        samples_per_video=args.samples_per_video
    )
    
    if len(dataset.samples) == 0:
        print("No videos found!")
        return

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    
    # 4. Initialize Model
    model = LatentDCNWorldModel(
        e_dim=args.e_dim, 
        cond_channels=args.cond_channels, 
        base_channels=args.base_channels,
        latent_dim=args.latent_dim
    ).to(device)
    
    if args.resume_from:
        if os.path.exists(args.resume_from):
            print(f"Resuming training from checkpoint: {args.resume_from}")
            model.load_state_dict(torch.load(args.resume_from, map_location=device, weights_only=True))
        else:
            print(f"Warning: Checkpoint {args.resume_from} not found.")
            
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_decay)
    criterion_l1 = nn.L1Loss()
    
    try:
        import lpips
        criterion_perceptual = lpips.LPIPS(net='vgg').to(device)
        criterion_perceptual.eval()
        use_perceptual = True
        print("LPIPS loaded.")
    except ImportError:
        use_perceptual = False

    os.makedirs("checkpoints", exist_ok=True)
    
    mlflow.set_experiment(args.experiment_name)
    with mlflow.start_run():
        mlflow.log_params(vars(args))
        mlflow.log_param("device", str(device))

        print("Starting training...")
        model.train()

        # Calculate global step offset so MLflow curves are continuous
        global_step = args.start_epoch * len(dataloader)

        for epoch in range(args.epochs):
            current_epoch = args.start_epoch + epoch + 1
            epoch_loss = 0.0
            pbar = tqdm(dataloader, desc=f"Epoch {current_epoch}/{args.start_epoch + args.epochs}")
            
            for batch_idx, (video_chunk, embed_chunk) in enumerate(pbar):
                video_chunk = video_chunk.to(device)
                embed_chunk = embed_chunk.to(device)
                B = video_chunk.shape[0]
                h_prev = torch.zeros(B, args.latent_dim, 8, 16).to(device)
                
                optimizer.zero_grad()
                total_bptt_loss = 0.0
                
                # For visualization, we'll pick one random step in the sequence to capture
                vis_step_t = torch.randint(0, args.seq_len, (1,)).item()
                vis_sample = None
                
                for t in range(args.seq_len):
                    x_t = video_chunk[:, t]
                    x_target = video_chunk[:, t+1]
                    e_t = embed_chunk[:, t]
                    
                    if torch.rand(1).item() < args.degrade_prob:
                        x_t_input = apply_degradation(x_t)
                    else:
                        x_t_input = x_t
                    
                    x_pred, h_prev = model(x_t_input, h_prev, e_t)
                    
                    # Capture visualization sample
                    if global_step % args.sample_interval == 0 and t == vis_step_t:
                        vis_sample = (x_t_input.detach(), x_target.detach(), x_pred.detach())
                    
                    loss_l1 = criterion_l1(x_pred, x_target)
                    step_loss = loss_l1
                    if use_perceptual:
                        loss_p = criterion_perceptual(x_pred, x_target).mean()
                        step_loss = step_loss + 0.5 * loss_p
                    total_bptt_loss += step_loss
                    
                total_bptt_loss.backward()
                optimizer.step()
                h_prev = h_prev.detach()
                
                # Handle Visualization
                if vis_sample is not None:
                    save_and_show_sample(*vis_sample, global_step, current_epoch, args.show_preview)
                    # Also log to MLflow
                    mlflow.log_artifact(f"train_samples/sample_ep{current_epoch}_step{global_step}.png")
                
                import resource
                mem_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
                avg_loss = total_bptt_loss.item() / args.seq_len
                epoch_loss += avg_loss
                pbar.set_postfix({'loss': f"{avg_loss:.4f}", 'mem_MB': f"{mem_mb:.1f}"})
                
                mlflow.log_metric("step_loss", avg_loss, step=global_step)
                mlflow.log_metric("mem_mb", mem_mb, step=global_step)
                global_step += 1
                
            avg_epoch_loss = epoch_loss / len(dataloader)
            print(f"Epoch {current_epoch} Average Loss: {avg_epoch_loss:.4f}")
            mlflow.log_metric("epoch_loss", avg_epoch_loss, step=current_epoch)
            
            ckpt_path = f"checkpoints/world_model_ep{current_epoch}.pt"
            torch.save(model.state_dict(), ckpt_path)

if __name__ == "__main__":
    main()
