import os
import sys
import glob
import pandas as pd
from pathlib import Path
import cv2
import numpy as np
import torch
from tqdm import tqdm

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from third_party.RAFT.core.raft import RAFT
from third_party.RAFT.core.utils_core.utils import InputPadder
from easydict import EasyDict as edict

raft_info = {
    "pretrained": f'pretrained/raft_model/raft-things.pth',
}

class DynamicDegreeCalculator:
    def __init__(self, device):
        self.device = device
        self.model = self.load_model()
    
    def load_model(self):
        args = edict({
            "model": raft_info["pretrained"],
            "small": False,
            "mixed_precision": False,
            "alternate_corr": False
        })
        
        model = RAFT(args)
        ckpt = torch.load(args.model, map_location="cpu")
        
        new_ckpt = {k.replace('module.', ''): v for k, v in ckpt.items()}
        model.load_state_dict(new_ckpt)
        
        model.to(self.device)
        model.eval()
        return model

    def get_score(self, img, flo):
        img = img[0].permute(1,2,0).cpu().numpy()
        flo = flo[0].permute(1,2,0).cpu().numpy()

        u = flo[:,:,0]
        v = flo[:,:,1]
        rad = np.sqrt(np.square(u) + np.square(v))
        
        h, w = rad.shape
        rad_flat = rad.flatten()
        cut_index = int(h*w*0.05)

        max_rad = np.mean(abs(np.sort(-rad_flat))[:cut_index])
        return max_rad.item()

    def set_params(self, frame, count):
        scale = min(list(frame.shape)[-2:])
        self.params = {
            "thres": 6.0 * (scale / 256.0), 
            "count_num": round(4 * (count / 16.0))
        }

    def get_frames(self, video_path, num_frames=16):
        frame_list = []
        video = cv2.VideoCapture(video_path)
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        
        interval = max(1, total_frames // num_frames)
        
        frame_count = 0
        while video.isOpened():
            success, frame = video.read()
            if success:
                if frame_count % interval == 0:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame = torch.from_numpy(frame.astype(np.uint8)).permute(2, 0, 1).float()
                    frame = frame[None].to(self.device)
                    frame_list.append(frame)
                    
                    if len(frame_list) >= num_frames:
                        break
                frame_count += 1
            else:
                break
                
        video.release()
        return frame_list

    def compute_dynamic_score(self, video_path, num_frames=16):
        with torch.no_grad():
            frames = self.get_frames(video_path, num_frames)
            
            if len(frames) < 2:
                return 0.0, []
            
            self.set_params(frame=frames[0], count=len(frames))
            dynamic_scores = []
            frame_results = []
            
            for i, (image1, image2) in enumerate(zip(frames[:-1], frames[1:])):
                padder = InputPadder(image1.shape)
                image1, image2 = padder.pad(image1, image2)
                
                _, flow_up = self.model(image1, image2, iters=20, test_mode=True)
                
                max_rad = self.get_score(image1, flow_up)
                dynamic_scores.append(max_rad)
                
                frame_results.append({
                    'frame_pair_index': i,
                    'dynamic_score': max_rad
                })
            
            overall_score = np.mean(dynamic_scores) if dynamic_scores else 0.0
            
            return overall_score, frame_results

def compute_dynamic_single_video(video_path, device, num_frames=16):
    calculator = DynamicDegreeCalculator(device)
    score, frame_results = calculator.compute_dynamic_score(video_path, num_frames)
    
    # print(f"Video: {video_path}")
    # print(f"Total frame pairs processed: {len(frame_results)}")
    # print(f"Overall dynamic score: {score:.4f}")
    # print("-" * 50)
    
    return score, frame_results

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
    
    for video_path in tqdm(video_files, desc="Processing videos"):
        filename = os.path.basename(video_path)
        
        initial_scale, snr_split, remaining_scale, prompt_index = parse_video_filename(filename)
        
        if prompt_index is None:
            print(f"Invalid prompt index for file {filename}, skipping...")
            continue
        
        dynamic_score, frame_results = compute_dynamic_single_video(video_path, device, num_frames)
        
        results.append({
            'prompt': prompt_index,
            'initial_scale': initial_scale,
            'snr_split': snr_split,
            'remaining_scale': remaining_scale,
            'dynamic_degree_score': dynamic_score,
            'filename': filename
        })
            
        # print(f"Completed: {filename} - Dynamic Score: {dynamic_score:.4f}")

    return results

def save_results_to_excel(results, output_path):
    df = pd.DataFrame(results)
    
    columns_order = ['prompt', 'initial_scale', 'snr_split', 'remaining_scale', 'dynamic_degree_score', 'filename']
    df = df[columns_order]
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    df.to_excel(output_path, index=False)
    print(f"Results saved to: {output_path}")
    
    return df


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    video_folder = "samples"
    output_excel = "results/dynamic_degree_results.xlsx"
    num_frames = 16

    results = process_all_videos_in_folder(video_folder, device, num_frames)
    
    df = save_results_to_excel(results, output_excel)
    print(f"Processed {len(results)} videos successfully.")