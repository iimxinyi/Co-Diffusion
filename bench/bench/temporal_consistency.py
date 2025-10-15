import os
import sys
import glob
import pandas as pd
from pathlib import Path

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

import torch
import numpy as np
import clip
from PIL import Image
from third_party.vbench.utils import load_video
import torch.nn.functional as F

os.environ["HF_HOME"] = "pretrained"
os.environ["TRANSFORMERS_CACHE"] = "pretrained"
os.environ["HF_DATASETS_CACHE"] = "pretrained"
os.environ["HF_MODULES_CACHE"] = "pretrained"


def load_clip_model(device, model_name='ViT-L/14'):
    model_path = 'pretrained'
    model, preprocess = clip.load(model_name, device=device, download_root=model_path)
    return model, preprocess

@torch.no_grad()
def calculate_temporal_consistency(model, preprocess, video_path, device, num_frames=16):
    video_array = load_video(video_path, num_frames=num_frames, return_tensor=False, width=224, height=224)
    
    if len(video_array) < 2:
        return 0.0, []
    
    frame_features = []
    for i, frame in enumerate(video_array):
        if isinstance(frame, np.ndarray):
            if frame.dtype == np.float32 or frame.dtype == np.float64:
                frame = (frame * 255).astype(np.uint8)
            frame = Image.fromarray(frame)
            
        processed_frame = preprocess(frame).unsqueeze(0).to(device)
        features = model.encode_image(processed_frame)
        features = F.normalize(features, dim=-1, p=2)
        frame_features.append(features)
    
    if not frame_features:
        return 0.0, []
    
    frame_features = torch.cat(frame_features, dim=0)
    
    frame_scores = []
    pairwise_similarities = []
    results = []
    
    total_similarity = 0.0
    valid_pairs = 0
    
    for i in range(len(frame_features) - 1):
        current_frame_feature = frame_features[i].unsqueeze(0)
        next_frame_feature = frame_features[i + 1].unsqueeze(0)
        
        similarity = F.cosine_similarity(current_frame_feature, next_frame_feature).item()
        similarity = max(0.0, similarity)
        
        frame_scores.append(similarity)
        pairwise_similarities.append({
            'frame_pair': (i, i + 1),
            'similarity': similarity
        })
        
        total_similarity += similarity
        valid_pairs += 1
        
        results.append({
            'frame_pair': f"{i}-{i+1}",
            'similarity': similarity
        })
    
    overall_score = total_similarity / valid_pairs if valid_pairs > 0 else 0.0
    
    return overall_score, results

def compute_temporal_consistency_single_video(video_path, device, num_frames=16):
    model, preprocess = load_clip_model(device)
    model.eval()
    
    overall_score, frame_results = calculate_temporal_consistency(model, preprocess, video_path, device, num_frames)
    
    print(f"Video: {os.path.basename(video_path)}")
    print(f"Temporal Consistency Score: {overall_score:.4f}")
    print(f"Processed {len(frame_results)} frame pairs")
    print("-" * 50)
    
    return overall_score, frame_results

def parse_video_filename(filename):
    filename = Path(filename).stem
    
    try:
        parts = filename.split('_')
        
        initial_scale = None
        snr_split = None
        remaining_scale = None
        prompt_index = None
        
        for i, part in enumerate(parts):
            if part == 'Init' and i + 1 < len(parts):
                initial_scale = parts[i + 1]
            elif part == 'SNR' and i + 1 < len(parts):
                snr_split = parts[i + 1]
            elif part == 'Rema' and i + 1 < len(parts):
                remaining_scale = parts[i + 1]
            elif part == 'Prompt' and i + 1 < len(parts):
                prompt_index = int(parts[i + 1])
        
        return initial_scale, snr_split, remaining_scale, prompt_index
    except Exception as e:
        print(f"Error parsing filename {filename}: {e}")
        return None, None, None, None

def process_all_videos_in_folder(folder_path, device, num_frames=16):
    video_files = glob.glob(os.path.join(folder_path, "*.mp4"))
    
    if not video_files:
        print(f"No MP4 files found in {folder_path}")
        return []
    
    results = []
    
    for video_path in video_files:
        filename = os.path.basename(video_path)
        
        initial_scale, snr_split, remaining_scale, prompt_index = parse_video_filename(filename)
        
        temporal_score, frame_results = compute_temporal_consistency_single_video(
            video_path, device, num_frames
        )
        
        results.append({
            'prompt': prompt_index,
            'initial_scale': initial_scale,
            'snr_split': snr_split,
            'remaining_scale': remaining_scale,
            'temporal_consistency_score': temporal_score,
            'filename': filename
        })
        
        print(f"Completed: {filename} - Temporal Consistency: {temporal_score:.4f}")

    return results

def save_results_to_excel(results, output_path):
    df = pd.DataFrame(results)
    
    columns_order = ['prompt', 'initial_scale', 'snr_split', 'remaining_scale', 
                    'temporal_consistency_score', 'filename']
    df = df[columns_order]
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    df.to_excel(output_path, index=False)
    print(f"Results saved to: {output_path}")
    
    return df


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    video_folder = "samples"
    output_excel = "results/temporal_consistency_results.xlsx"
    num_frames = 16

    results = process_all_videos_in_folder(video_folder, device, num_frames)
    
    df = save_results_to_excel(results, output_excel)
    print(f"Processed {len(results)} videos successfully.")