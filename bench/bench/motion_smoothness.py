import os
import sys
import glob
import pandas as pd
from pathlib import Path
import cv2
import torch
import numpy as np
from tqdm import tqdm
from omegaconf import OmegaConf

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from third_party.amt.utils.utils import (
    img2tensor, tensor2img, check_dim_and_resize, InputPadder
)
from third_party.amt.utils.build_utils import build_from_cfg


class FrameProcess:
    def __init__(self):
        pass

    def get_frames(self, video_path):
        frame_list = []
        video = cv2.VideoCapture(video_path)
        while video.isOpened():
            success, frame = video.read()
            if success:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # convert to rgb
                frame_list.append(frame)
            else:
                break
        video.release()
        assert frame_list != []
        return frame_list 
    
    def get_frames_from_img_folder(self, img_folder):
        exts = ['jpg', 'png', 'jpeg', 'bmp', 'tif', 
                'tiff', 'JPG', 'PNG', 'JPEG', 'BMP', 
                'TIF', 'TIFF']
        frame_list = []
        imgs = sorted([p for p in glob.glob(os.path.join(img_folder, "*")) if os.path.splitext(p)[1][1:] in exts])
        for img in imgs:
            frame = cv2.imread(img, cv2.IMREAD_COLOR)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_list.append(frame)
        assert frame_list != []
        return frame_list

    def extract_frame(self, frame_list, start_from=0):
        extract = []
        for i in range(start_from, len(frame_list), 2):
            extract.append(frame_list[i])
        return extract

class MotionSmoothness:
    def __init__(self, config, ckpt, device):
        self.device = device
        self.config = config
        self.ckpt = ckpt
        self.niters = 1
        self.initialization()
        self.load_model()

    def load_model(self):
        cfg_path = self.config
        ckpt_path = self.ckpt
        network_cfg = OmegaConf.load(cfg_path).network
        network_name = network_cfg.name
        self.model = build_from_cfg(network_cfg)
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        self.model.load_state_dict(ckpt['state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()

    def initialization(self):
        if self.device == 'cuda':
            self.anchor_resolution = 1024 * 512
            self.anchor_memory = 1500 * 1024**2
            self.anchor_memory_bias = 2500 * 1024**2
            self.vram_avail = torch.cuda.get_device_properties(self.device).total_memory
            print("VRAM available: {:.1f} MB".format(self.vram_avail / 1024 ** 2))
        else:
            # Do not resize in cpu mode
            self.anchor_resolution = 8192*8192
            self.anchor_memory = 1
            self.anchor_memory_bias = 0
            self.vram_avail = 1

        self.embt = torch.tensor(1/2).float().view(1, 1, 1, 1).to(self.device)
        self.fp = FrameProcess()

    def motion_score(self, video_path):
        iters = int(self.niters)
        # get inputs
        if video_path.endswith('.mp4'):
            frames = self.fp.get_frames(video_path)
        elif os.path.isdir(video_path):
            frames = self.fp.get_frames_from_img_folder(video_path)
        else:
            raise NotImplementedError
        frame_list = self.fp.extract_frame(frames, start_from=0)
        inputs = [img2tensor(frame).to(self.device) for frame in frame_list]
        assert len(inputs) > 1, f"The number of input should be more than one (current {len(inputs)})"
        inputs = check_dim_and_resize(inputs)
        h, w = inputs[0].shape[-2:]
        scale = self.anchor_resolution / (h * w) * np.sqrt((self.vram_avail - self.anchor_memory_bias) / self.anchor_memory)
        scale = 1 if scale > 1 else scale
        scale = 1 / np.floor(1 / np.sqrt(scale) * 16) * 16
        if scale < 1:
            print(f"Due to the limited VRAM, the video will be scaled by {scale:.2f}")
        padding = int(16 / scale)
        padder = InputPadder(inputs[0].shape, padding)
        inputs = padder.pad(*inputs)

        # Frame interpolation
        for i in range(iters):
            outputs = [inputs[0]]
            for in_0, in_1 in zip(inputs[:-1], inputs[1:]):
                in_0 = in_0.to(self.device)
                in_1 = in_1.to(self.device)
                with torch.no_grad():
                    imgt_pred = self.model(in_0, in_1, self.embt, scale_factor=scale, eval=True)['imgt_pred']
                outputs += [imgt_pred.cpu(), in_1.cpu()]
            inputs = outputs

        # Calculate VFI score
        outputs = padder.unpad(*outputs)
        outputs = [tensor2img(out) for out in outputs]
        vfi_score = self.vfi_score(frames, outputs)
        norm = (255.0 - vfi_score)/255.0
        return norm

    def vfi_score(self, ori_frames, interpolate_frames):
        ori = self.fp.extract_frame(ori_frames, start_from=1)
        interpolate = self.fp.extract_frame(interpolate_frames, start_from=1)
        scores = []
        for i in range(len(interpolate)):
            scores.append(self.get_diff(ori[i], interpolate[i]))
        return np.mean(np.array(scores))

    def get_diff(self, img1, img2):
        img = cv2.absdiff(img1, img2)
        return np.mean(img)

def compute_motion_smoothness_single_video(video_path, device, submodules_list, num_frames=None):
    config = submodules_list["config"]
    ckpt = submodules_list["ckpt"]
    
    motion = MotionSmoothness(config, ckpt, device)
    
    try:
        score = motion.motion_score(video_path)
        return score, [{'video_path': video_path, 'motion_smoothness_score': score}]
    except Exception as e:
        print(f"Error processing {video_path}: {e}")
        return 0.0, [{'video_path': video_path, 'motion_smoothness_score': 0.0, 'error': str(e)}]

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

def process_all_videos_in_folder(folder_path, device, num_frames=None):
    video_files = glob.glob(os.path.join(folder_path, "*.mp4"))
    
    if not video_files:
        print(f"No MP4 files found in {folder_path}")
        return []
    
    submodules_list = {
        "config": f"pretrained/amt_model/AMT-S.yaml",
        "ckpt": f"pretrained/amt_model/amt-s.pth"
    }
    
    if not os.path.exists(submodules_list["config"]):
        print(f"Config file not found: {submodules_list['config']}")
        return []
    if not os.path.exists(submodules_list["ckpt"]):
        print(f"Checkpoint file not found: {submodules_list['ckpt']}")
        return []
    
    results = []
    
    for video_path in tqdm(video_files, desc="Processing videos"):
        filename = os.path.basename(video_path)
        
        initial_scale, snr_split, remaining_scale, prompt_index = parse_video_filename(filename)

        score, video_results = compute_motion_smoothness_single_video(
            video_path, device, submodules_list, num_frames
        )
            
        results.append({
            'prompt': prompt_index,
            'initial_scale': initial_scale,
            'snr_split': snr_split,
            'remaining_scale': remaining_scale,
            'motion_smoothness_score': score,
            'filename': filename
        })
            
        # print(f"Completed: {filename} - Motion Smoothness Score: {score:.4f}")

    return results

def save_results_to_excel(results, output_path):
    df = pd.DataFrame(results)
    
    columns_order = ['prompt', 'initial_scale', 'snr_split', 'remaining_scale', 'motion_smoothness_score', 'filename']
    df = df[columns_order]
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    df.to_excel(output_path, index=False)
    print(f"Results saved to: {output_path}")
    
    return df


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    video_folder = "samples"
    output_excel = "results/motion_smoothness_results.xlsx"

    results = process_all_videos_in_folder(video_folder, device)
    
    df = save_results_to_excel(results, output_excel)
    print(f"Processed {len(results)} videos successfully.")
        
    scores = [r['motion_smoothness_score'] for r in results if isinstance(r['motion_smoothness_score'], (int, float))]
    if scores:
        print(f"Average Motion Smoothness Score: {np.mean(scores):.4f}")
        print(f"Score Range: {min(scores):.4f} - {max(scores):.4f}")