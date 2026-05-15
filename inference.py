import os
# Suppress OpenCV Qt font warnings
os.environ["QT_LOGGING_RULES"] = "qt.qpa.fonts.warning=false;qt.qpa.fonts.critical=false"

import torch
import cv2
import time
import glob
import argparse
import numpy as np
import torchvision.transforms.functional as F
from world_model import WorldModelEngine

def get_latest_checkpoint(checkpoint_dir="checkpoints"):
    """Finds the latest checkpoint file based on modification time."""
    checkpoints = glob.glob(os.path.join(checkpoint_dir, "*.pt"))
    if not checkpoints:
        return None
    latest_checkpoint = max(checkpoints, key=os.path.getmtime)
    return latest_checkpoint

def load_seed_frame(video_path, height=32, width=64):
    """Loads the very first frame of a video to use as the seed for generation."""
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        raise ValueError(f"Could not read from {video_path}")
        
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Prepare tensor
    vframe = torch.tensor(frame, dtype=torch.uint8).permute(2, 0, 1).unsqueeze(0).float() # (1, 3, H, W)
    vframe = F.resize(vframe, [height, width], antialias=True)
    vframe = (vframe / 127.5) - 1.0 # [-1, 1]
    
    return vframe

def generate_noise_seed(height=32, width=64):
    """Generates a purely random noisy image in [-1, 1] as the seed."""
    return (torch.rand(1, 3, height, width) * 2.0) - 1.0

def parse_args():
    parser = argparse.ArgumentParser(description="World Model Realtime Inference")
    parser.add_argument('--checkpoint', type=str, default='latest', help='Path to model checkpoint. Default: automatically finds the newest .pt file in ./checkpoints/')
    parser.add_argument('--seed', type=str, default='noise', help="Seed type: 'noise' (default) or path to a specific .mp4 file")
    parser.add_argument('--fps', type=int, default=30, help='Target playback frames per second')
    parser.add_argument('--gain', type=float, default=1.0, help='Scaling factor for the predicted frame to prevent feedback saturation')
    parser.add_argument('--noise_level', type=float, default=0.02, help='Amount of innovation noise to add at each step')
    parser.add_argument('--stimulus', action='store_true', help='If set, adds a moving dot to stimulate the world model')
    parser.add_argument('--balance_channels', action='store_true', help='If set, balances RGB channels by 10%% each frame to prevent color collapse')
    parser.add_argument('--debug', action='store_true', help='If set, logs internal state statistics for diagnosis')
    parser.add_argument('--e_dim', type=int, default=128, help='Embedding dimension (must match training)')
    parser.add_argument('--cond_channels', type=int, default=32, help='Condition channels (must match training)')
    parser.add_argument('--latent_dim', type=int, default=256, help='Latent dimension (must match training)')
    return parser.parse_args()

def generate_video():
    args = parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Inference device: {device}")
    
    height, width = 32, 64
    
    # Resolve Checkpoint
    if args.checkpoint.lower() == 'latest':
        checkpoint_path = get_latest_checkpoint()
        if checkpoint_path is None:
            print("No checkpoints found in ./checkpoints/")
            return
        print(f"Auto-detected latest checkpoint: {checkpoint_path}")
    else:
        checkpoint_path = args.checkpoint
        
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found at {checkpoint_path}")
        return
        
    # 1. Load Model
    model = WorldModelEngine(
        e_dim=args.e_dim, 
        cond_channels=args.cond_channels, 
        latent_dim=args.latent_dim
    ).to(device)
    
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    model.eval()
    
    # 2. Load Seed Frame
    if args.seed.lower() == 'noise':
        print("Using pure random noise as the seed frame.")
        x_t = generate_noise_seed(height, width).to(device) # Shape: (1, 3, 32, 64)
    else:
        if not os.path.exists(args.seed):
            print(f"Seed video not found at {args.seed}")
            return
        print(f"Using seed frame from: {args.seed}")
        x_t = load_seed_frame(args.seed, height, width).to(device)
    
    # Initial hidden state (Zeros)
    h_prev = torch.zeros(1, args.latent_dim, 8, 16).to(device)
    
    # Interactive State
    should_reset = [False]
    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            should_reset[0] = True
            
    cv2.namedWindow("World Model Inference")
    cv2.setMouseCallback("World Model Inference", on_mouse)

    print(f"Starting infinite realtime inference (Target: {args.fps} FPS)...")
    print("  -> Press 'r' or CLICK the window to reset with fresh noise.")
    print("  -> Press 'q' to quit.")
    
    # Dummy embedding (all ones) for the 'video' concept
    e_t = torch.ones((1, args.e_dim)).to(device)
    
    frame_count = 0
    start_time = time.time()
    
    # Target frame delay in ms
    target_delay_ms = int(1000 / args.fps)
    
    with torch.no_grad():
        while True:
            # Handle Reset
            if should_reset[0]:
                print("Resetting world model state...")
                x_t = generate_noise_seed(height, width).to(device)
                h_prev = torch.zeros(1, args.latent_dim, 8, 16).to(device)
                should_reset[0] = False
                frame_count = 0

            frame_start = time.time()
            
            # Predict next frame
            x_t_plus_1, h_prev = model(x_t, h_prev, e_t)
            
            # --- STABILIZATION & STIMULATION ---
            
            # 0. Apply Gain (Cooldown)
            if args.gain != 1.0:
                x_t_plus_1 = x_t_plus_1 * args.gain
                
            # 1. Clamp output to [-1, 1] to prevent value explosion/collapse
            x_t = x_t_plus_1.clamp(-1.0, 1.0)
            
            # 2. Channel Balancing (Preventing Color Collapse)
            if args.balance_channels:
                # Calculate mean of each channel (B, 3, 1, 1)
                c_means = x_t.mean(dim=(2, 3), keepdim=True)
                # Calculate global average across all pixels/channels
                g_mean = c_means.mean(dim=1, keepdim=True)
                # Pull each channel 10% toward the global average
                x_t = x_t - 0.1 * (c_means - g_mean)
                x_t = x_t.clamp(-1.0, 1.0)
            
            # 3. Add Innovation Noise (Stimulation)
            if args.noise_level > 0:
                noise = (torch.rand_like(x_t) * 2.0 - 1.0) * args.noise_level
                x_t = (x_t + noise).clamp(-1.0, 1.0)
            
            # 4. Add Stimulus Dot (moving pertubation)
            if args.stimulus:
                # Calculate a more irregular, slower moving position
                t = frame_count * 0.03 
                cx = int(32 + 15 * np.cos(t) + 10 * np.sin(t * 0.7))
                cy = int(16 + 8 * np.sin(t * 1.3) + 5 * np.cos(t * 0.5))
                cx = max(0, min(width - 2, cx))
                cy = max(0, min(height - 2, cy))
                x_t[:, :, cy:cy+2, cx:cx+2] = 1.0 
            
            # Post-process for display
            out_frame = x_t.squeeze(0).cpu() # (3, H, W)
            out_frame = ((out_frame + 1.0) * 127.5).byte().permute(1, 2, 0).numpy() # (H, W, 3)
            out_frame_bgr = cv2.cvtColor(out_frame, cv2.COLOR_RGB2BGR)
            
            # Scale up for better visibility on screen (e.g., 8x scale)
            display_frame = cv2.resize(out_frame_bgr, (width * 8, height * 8), interpolation=cv2.INTER_NEAREST)
            
            # Show frame
            cv2.imshow("World Model Inference", display_frame)
            
            # Calculate how long the frame took to generate
            frame_gen_time = (time.time() - frame_start) * 1000
            
            # Wait for the remaining time to hit target FPS, minimum 1ms
            wait_time = max(1, target_delay_ms - int(frame_gen_time))
            
            key = cv2.waitKey(wait_time) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                should_reset[0] = True
            
            # Measure realtime performance periodically
            frame_count += 1
            if frame_count % args.fps == 0:
                elapsed = time.time() - start_time
                fps_val = args.fps / elapsed
                print(f"Generating at: {fps_val:.2f} fps")
                
                if args.debug:
                    # Diagnostics
                    c_m = x_t.mean(dim=(2, 3)).squeeze().cpu().numpy()
                    c_s = x_t.std(dim=(2, 3)).squeeze().cpu().numpy()
                    h_norm = h_prev.abs().mean().item()
                    print(f"  [DEBUG] R_mean: {c_m[0]:.3f} | G_mean: {c_m[1]:.3f} | B_mean: {c_m[2]:.3f}")
                    print(f"  [DEBUG] R_std:  {c_s[0]:.3f} | G_std:  {c_s[1]:.3f} | B_std:  {c_s[2]:.3f}")
                    print(f"  [DEBUG] Hidden State Abs Mean: {h_norm:.4f} | Range: [{x_t.min():.2f}, {x_t.max():.2f}]")
                
                start_time = time.time()
                
    cv2.destroyAllWindows()
    print("Inference stopped.")

if __name__ == "__main__":
    generate_video()