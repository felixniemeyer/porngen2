import os
# Suppress OpenCV Qt font warnings
os.environ["QT_LOGGING_RULES"] = "qt.qpa.fonts.warning=false;qt.qpa.fonts.critical=false"

import torch
import cv2
import time
import numpy as np
import torchvision.transforms.functional as F
from world_model import WorldModelEngine

def generate_noise_seed(height=32, width=64):
    """Generates a purely random noisy image in [-1, 1] as the seed."""
    return (torch.rand(1, 3, height, width) * 2.0) - 1.0

def generate_video():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Inference device: {device}")
    
    # 1. Configuration
    e_dim = 128
    cond_channels = 32
    latent_dim = 256
    height, width = 32, 64
    target_fps = 30
    num_frames_to_generate = 150 # 5 seconds of video at 30 fps
    
    checkpoint_path = "checkpoints/world_model_ep5.pt"
    
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found at {checkpoint_path}")
        return
        
    # 2. Load Model
    model = WorldModelEngine(
        e_dim=e_dim, 
        cond_channels=cond_channels, 
        latent_dim=latent_dim
    ).to(device)
    
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    model.eval()
    
    # 3. Load Seed Frame
    print("Using pure random noise as the seed frame.")
    x_t = generate_noise_seed(height, width).to(device) # Shape: (1, 3, 32, 64)
    
    # Initial hidden state (Zeros)
    h_prev = torch.zeros(1, latent_dim, 8, 16).to(device)
    
    print("Starting infinite realtime inference... Press 'q' in the window to quit.")
    
    # Dummy embedding (all ones) for the 'video' concept
    e_t = torch.ones((1, e_dim)).to(device)
    
    frame_count = 0
    start_time = time.time()
    
    # Target frame delay in ms for 30 FPS
    target_delay_ms = int(1000 / target_fps)
    
    with torch.no_grad():
        while True:
            frame_start = time.time()
            
            # Predict next frame
            x_t_plus_1, h_prev = model(x_t, h_prev, e_t)
            
            # Post-process for display
            out_frame = x_t_plus_1.squeeze(0).cpu().clamp(-1, 1) # (3, H, W)
            out_frame = ((out_frame + 1.0) * 127.5).byte().permute(1, 2, 0).numpy() # (H, W, 3)
            out_frame_bgr = cv2.cvtColor(out_frame, cv2.COLOR_RGB2BGR)
            
            # Scale up for better visibility on screen (e.g., 4x scale)
            display_frame = cv2.resize(out_frame_bgr, (width * 8, height * 8), interpolation=cv2.INTER_NEAREST)
            
            # Show frame
            cv2.imshow("World Model Inference", display_frame)
            
            # Calculate how long the frame took to generate
            frame_gen_time = (time.time() - frame_start) * 1000
            
            # Wait for the remaining time to hit ~30 FPS, minimum 1ms
            wait_time = max(1, target_delay_ms - int(frame_gen_time))
            
            if cv2.waitKey(wait_time) & 0xFF == ord('q'):
                break
            
            # Update input for next iteration
            x_t = x_t_plus_1
            
            # Measure realtime performance periodically
            frame_count += 1
            if frame_count % 30 == 0:
                elapsed = time.time() - start_time
                print(f"Generating at: {30 / elapsed:.2f} fps")
                start_time = time.time()
                
    cv2.destroyAllWindows()
    print("Inference stopped.")

if __name__ == "__main__":
    generate_video()