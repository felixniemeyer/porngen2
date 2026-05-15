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
from world_model import LatentDCNWorldModel
from larms_model import LaRMS

def get_latest_checkpoint(checkpoint_dir="checkpoints", model_type=None):
    """Finds the latest checkpoint file based on modification time."""
    search_path = os.path.join(checkpoint_dir, model_type, "**", "*.pt") if model_type else os.path.join(checkpoint_dir, "**", "*.pt")
    checkpoints = glob.glob(search_path, recursive=True)
    if not checkpoints:
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
    vframe = torch.tensor(frame, dtype=torch.uint8).permute(2, 0, 1).unsqueeze(0).float() # (1, 3, H, W)
    vframe = F.resize(vframe, [height, width], antialias=True)
    vframe = (vframe / 127.5) - 1.0 # [-1, 1]
    return vframe

def generate_noise_seed(height=32, width=64):
    """Generates a purely random noisy image in [-1, 1] as the seed."""
    return (torch.rand(1, 3, height, width) * 2.0) - 1.0

def parse_args():
    parser = argparse.ArgumentParser(description="World Model Realtime Inference")
    parser.add_argument('--checkpoint', type=str, default='latest', help='Path to checkpoint')
    parser.add_argument('--model_type', type=str, default='larms', choices=['dcn', 'larms'])
    parser.add_argument('--seed', type=str, default='noise')
    parser.add_argument('--fps', type=int, default=30)
    parser.add_argument('--gain', type=float, default=1.0)
    parser.add_argument('--noise_level', type=float, default=0.02)
    parser.add_argument('--stimulus', action='store_true')
    parser.add_argument('--balance_channels', action='store_true')
    parser.add_argument('--debug', action='store_true', help='Log stats to console')
    parser.add_argument('--debug_view', action='store_true', help='Show internal hidden states in UI')
    parser.add_argument('--e_dim', type=int, default=128)
    parser.add_argument('--cond_channels', type=int, default=32)
    parser.add_argument('--base_channels', type=int, default=64)
    parser.add_argument('--latent_dim', type=int, default=256)
    return parser.parse_args()

def visualize_tensor(t, title, scale=8):
    """Utility to convert a single-channel latent/mask to a visible BGR image."""
    img = t.detach().cpu().squeeze().clamp(0, 1).numpy()
    img = (img * 255).astype(np.uint8)
    img_color = cv2.applyColorMap(img, cv2.COLORMAP_VIRIDIS)
    h, w = img_color.shape[:2]
    return cv2.resize(img_color, (w * scale, h * scale), interpolation=cv2.INTER_NEAREST)

def generate_video():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Inference device: {device}")
    
    height, width = 32, 64
    
    if args.checkpoint.lower() == 'latest':
        checkpoint_path = get_latest_checkpoint(model_type=args.model_type)
        if checkpoint_path is None:
            print("No checkpoints found."); return
        print(f"Auto-detected: {checkpoint_path}")
    else:
        checkpoint_path = args.checkpoint
        
    if not os.path.exists(checkpoint_path):
        print(f"Not found: {checkpoint_path}"); return
        
    # 1. Load Model
    if args.model_type == 'dcn':
        model = LatentDCNWorldModel(e_dim=args.e_dim, cond_channels=args.cond_channels, 
                                   base_channels=args.base_channels, latent_dim=args.latent_dim).to(device)
        model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    elif args.model_type == 'larms':
        model = LaRMS(e_dim=args.e_dim, cond_channels=args.cond_channels, latent_dim=args.latent_dim).to(device)
        model.load_migrated(checkpoint_path, device)
    
    model.eval()
    
    # 2. Seed
    if args.seed.lower() == 'noise':
        x_t = generate_noise_seed(height, width).to(device)
    else:
        x_t = load_seed_frame(args.seed, height, width).to(device)
    
    # 3. State
    if args.model_type == 'dcn':
        h_prev = torch.zeros(1, args.latent_dim, 8, 16).to(device)
    elif args.model_type == 'larms':
        h_prev = model.init_hidden(1, height, width, device)
    
    should_reset = [False]
    should_flip = [False]
    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN: should_reset[0] = True
            
    cv2.namedWindow("World Model Inference")
    cv2.setMouseCallback("World Model Inference", on_mouse)

    e_t = torch.ones((1, args.e_dim)).to(device)
    frame_count = 0
    start_time = time.time()
    target_delay_ms = int(1000 / args.fps)
    
    with torch.no_grad():
        while True:
            if should_reset[0]:
                x_t = generate_noise_seed(height, width).to(device)
                h_prev = torch.zeros(1, args.latent_dim, 8, 16).to(device) if args.model_type == 'dcn' else model.init_hidden(1, height, width, device)
                should_reset[0] = False; frame_count = 0
            
            if should_flip[0]:
                x_t = torch.flip(x_t, [3])
                if isinstance(h_prev, tuple): h_prev = tuple(torch.flip(h, [3]) for h in h_prev)
                else: h_prev = torch.flip(h_prev, [3])
                should_flip[0] = False

            frame_start = time.time()
            
            # Predict
            res = model(x_t, h_prev, e_t)
            if len(res) == 3:
                x_t_plus_1, h_prev, debug_info = res
            else:
                x_t_plus_1, h_prev = res
                debug_info = None
            
            if args.gain != 1.0: x_t_plus_1 = x_t_plus_1 * args.gain
            x_t = x_t_plus_1.clamp(-1.0, 1.0)
            
            if args.balance_channels:
                c_means = x_t.mean(dim=(2, 3), keepdim=True)
                g_mean = c_means.mean(dim=1, keepdim=True)
                x_t = (x_t - 0.1 * (c_means - g_mean)).clamp(-1.0, 1.0)
            
            if args.noise_level > 0:
                noise = (torch.rand_like(x_t) * 2.0 - 1.0) * args.noise_level
                x_t = (x_t + noise).clamp(-1.0, 1.0)
            
            if args.stimulus:
                t = frame_count * 0.03 
                cx, cy = int(32 + 15 * np.cos(t) + 10 * np.sin(t * 0.7)), int(16 + 8 * np.sin(t * 1.3) + 5 * np.cos(t * 0.5))
                x_t[:, :, max(0, min(height-2, cy)):max(0, min(height-2, cy))+2, max(0, min(width-2, cx)):max(0, min(width-2, cx))+2] = 1.0 
            
            # Display
            out_frame = x_t.squeeze(0).cpu()
            out_frame = ((out_frame + 1.0) * 127.5).byte().permute(1, 2, 0).numpy()
            out_frame_bgr = cv2.cvtColor(out_frame, cv2.COLOR_RGB2BGR)
            display_frame = cv2.resize(out_frame_bgr, (width * 8, height * 8), interpolation=cv2.INTER_NEAREST)
            
            if args.debug_view and debug_info:
                # Build diagnostic strip
                u_vis = visualize_tensor(debug_info['uncertainty'], "Uncertainty")
                hf_vis = visualize_tensor(debug_info['h_fast_mag'], "Fast Memory")
                hs_vis = visualize_tensor(debug_info['h_slow_mag'], "Slow Memory")
                strip = np.hstack([display_frame, u_vis, hf_vis, hs_vis])
                cv2.imshow("World Model Inference", strip)
            else:
                cv2.imshow("World Model Inference", display_frame)
            
            wait_time = max(1, target_delay_ms - int((time.time() - frame_start) * 1000))
            key = cv2.waitKey(wait_time) & 0xFF
            if key == ord('q'): break
            elif key == ord('r'): should_reset[0] = True
            elif key == ord('f'): should_flip[0] = True
            
            frame_count += 1
            if frame_count % args.fps == 0:
                elapsed = time.time() - start_time
                if args.debug:
                    print(f"Gen: {args.fps/elapsed:.1f} fps | H_Abs: {h_prev[0].abs().mean():.3f} (fast), {h_prev[1].abs().mean():.3f} (slow)")
                start_time = time.time()
                
    cv2.destroyAllWindows()

if __name__ == "__main__":
    generate_video()
