import os
import glob
import torch
import cv2
import numpy as np
import torchvision.transforms.functional as F
from torch.utils.data import Dataset, DataLoader

class Autolabeler:
    """
    Simulates an automatic labeling pipeline.
    Reads videos and extracts visual/audio/text guidance embeddings.
    """
    def __init__(self, e_dim=128, embeds_per_sec=1.0):
        self.e_dim = e_dim
        self.embeds_per_sec = embeds_per_sec
        # Future: Initialize CLIP, Audio encoder, or LLM here.
        
    def process_video(self, video_path):
        """
        Extracts embeddings for a video.
        Currently returns a dummy identity embedding (all ones) for the 'video' concept.
        """
        try:
            cap = cv2.VideoCapture(video_path)
            video_fps = cap.get(cv2.CAP_PROP_FPS)
            num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            
            if video_fps <= 0 or video_fps is None:
                video_fps = 24.0
            if num_frames <= 0:
                print(f"Warning: Could not get frame count for {video_path}, using dummy 100")
                num_frames = 100
                
            duration_sec = num_frames / video_fps
        except Exception as e:
            print(f"Error reading {video_path}: {e}")
            return None
            
        num_embeds = max(1, int(duration_sec * self.embeds_per_sec))
        
        # Generate Identity Embedding (e.g. all 1.0s to signify 'video' concept)
        # Shape: (num_embeds, e_dim)
        embeddings = torch.ones((num_embeds, self.e_dim))
        
        return embeddings, video_fps, num_frames


def run_labelling_process(video_dir='./train-videos', output_dir='./train-embeddings'):
    """
    Scans the video directory, runs the Autolabeler, and saves metadata/embeddings.
    """
    os.makedirs(output_dir, exist_ok=True)
    video_files = glob.glob(os.path.join(video_dir, '*.[mM][pP]4'))
    
    if not video_files:
        print(f"No videos found in {video_dir}. Please add some .mp4 files.")
        return

    labeler = Autolabeler(e_dim=128, embeds_per_sec=1.0)
    
    for video_path in video_files:
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        out_path = os.path.join(output_dir, f"{base_name}_meta.pt")
        
        if os.path.exists(out_path):
            print(f"Skipping {base_name}, embeddings already exist.")
            continue
            
        print(f"Labelling {base_name}...")
        result = labeler.process_video(video_path)
        if result:
            embeddings, fps, num_frames = result
            # Save the embeddings and metadata
            torch.save({
                'embeddings': embeddings,
                'fps': fps,
                'num_frames': num_frames,
                'embeds_per_sec': labeler.embeds_per_sec
            }, out_path)
            print(f"Saved {embeddings.shape[0]} embeddings for {base_name} to {out_path}.")


class WorldModelDataset(Dataset):
    """
    Loads chunks of frames and their corresponding interpolated embeddings.
    """
    def __init__(self, video_dir='./train-videos', embed_dir='./train-embeddings', 
                 seq_len=16, height=32, width=64, samples_per_video=100):
        self.video_dir = video_dir
        self.embed_dir = embed_dir
        self.seq_len = seq_len
        self.height = height
        self.width = width
        self.samples_per_video = samples_per_video
        
        self.samples = []
        for meta_path in glob.glob(os.path.join(embed_dir, '*_meta.pt')):
            base_name = os.path.basename(meta_path).replace('_meta.pt', '')
            video_path = os.path.join(video_dir, f"{base_name}.mp4")
            if os.path.exists(video_path):
                self.samples.append((video_path, meta_path))
                
        if len(self.samples) == 0:
            print(f"Warning: No valid (video, embedding) pairs found.")
                
    def __len__(self):
        # Sample N random chunks from each video per epoch
        return len(self.samples) * self.samples_per_video

    def __getitem__(self, idx):
        # Modulo to get actual video index
        video_idx = idx % len(self.samples)
        video_path, meta_path = self.samples[video_idx]
        
        meta = torch.load(meta_path, weights_only=True)
        num_frames = meta['num_frames']
        fps = meta['fps']
        embeds = meta['embeddings'] # (num_embeds, e_dim)
        embeds_per_sec = meta['embeds_per_sec']
        
        # We need seq_len + 1 frames for next-frame prediction
        chunk_size = self.seq_len + 1
        
        if num_frames <= chunk_size:
            start_frame = 0
        else:
            # Random starting frame
            start_frame = torch.randint(0, num_frames - chunk_size, (1,)).item()
            
        start_sec = start_frame / fps
        
        # Read the specific chunk using cv2
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        frames = []
        for _ in range(chunk_size):
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        cap.release()
        
        if len(frames) == 0:
            # Absolute fallback: black frames
            vframes = torch.zeros((chunk_size, self.height, self.width, 3), dtype=torch.uint8)
        else:
            # Fallback if video is too short or reading failed: pad with the last frame
            while len(frames) < chunk_size:
                frames.append(frames[-1])
            vframes = torch.tensor(np.array(frames), dtype=torch.uint8)
            
        vframes = vframes[:chunk_size] # Ensure exact size (chunk_size, H, W, C)
        
        # Convert from (T, H, W, C) to (T, C, H, W)
        vframes = vframes.permute(0, 3, 1, 2).float()
        
        # Resize to (32, 64)
        vframes = F.resize(vframes, [self.height, self.width], antialias=True)
        
        # Normalize to [-1, 1]
        vframes = (vframes / 127.5) - 1.0
        
        # Align embeddings: map each frame back to the correct embedding index
        frame_timestamps = start_sec + torch.arange(chunk_size) / fps
        embed_indices = (frame_timestamps * embeds_per_sec).long()
        embed_indices = torch.clamp(embed_indices, 0, embeds.shape[0] - 1)
        
        chunk_embeds = embeds[embed_indices] # Shape: (chunk_size, e_dim)
        
        return vframes, chunk_embeds

if __name__ == "__main__":
    print("Starting automated labelling process...")
    run_labelling_process()
    print("Labelling complete. You can now use WorldModelDataset in your training loop.")
