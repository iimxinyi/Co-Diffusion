import torch
import argparse
import os
import glob
import pandas as pd
from pathlib import Path

import numpy as np
import yaml
from tqdm import tqdm

import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from third_party.DOVER.dover.datasets import ViewDecompositionDataset
from third_party.DOVER.dover.models import DOVER

download_dir = "pretrained"
torch.hub.set_dir(download_dir)


class AdaptiveScoreFuser:
    def __init__(self):
        self.aesthetic_scores = []
        self.technical_scores = []
        self.is_calibrated = False
        self.calibration_params = {}
    
    def update_calibration(self, aesthetic_raw, technical_raw):
        self.aesthetic_scores.append(aesthetic_raw)
        self.technical_scores.append(technical_raw)
    
    def calibrate(self):
        if len(self.aesthetic_scores) == 0:
            return
        
        aesthetic_mean = np.mean(self.aesthetic_scores)
        aesthetic_std = np.std(self.aesthetic_scores)
        technical_mean = np.mean(self.technical_scores)
        technical_std = np.std(self.technical_scores)
        
        aesthetic_std = aesthetic_std if aesthetic_std > 1e-8 else 1.0
        technical_std = technical_std if technical_std > 1e-8 else 1.0
        
        self.calibration_params = {
            'aesthetic_mean': aesthetic_mean,
            'aesthetic_std': aesthetic_std,
            'technical_mean': technical_mean,
            'technical_std': technical_std
        }
        self.is_calibrated = True
        print(f"Calibration completed: Aesthetic - mean={aesthetic_mean:.4f}, std={aesthetic_std:.4f}, "
              f"Technical - mean={technical_mean:.4f}, std={technical_std:.4f}")
    
    def fuse_results(self, results: list):
        a_raw, t_raw = results[0], results[1]
        
        if not self.is_calibrated:
            t = (t_raw - 0.1107) / 0.07355
            a = (a_raw + 0.08285) / 0.03774
        else:
            t = (t_raw - self.calibration_params['technical_mean']) / self.calibration_params['technical_std']
            a = (a_raw - self.calibration_params['aesthetic_mean']) / self.calibration_params['aesthetic_std']
        
        x = t * 0.6104 + a * 0.3896
        
        return {
            "aesthetic": 1 / (1 + np.exp(-a)),
            "technical": 1 / (1 + np.exp(-t)),
            "overall": 1 / (1 + np.exp(-x)),
            "aesthetic_raw": a_raw,
            "technical_raw": t_raw
        }


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-o",
        "--opt",
        type=str,
        default="third_party/DOVER/dover.yml",
        help="the option file"
    )

    parser.add_argument(
        "-m",
        "--model_path",
        type=str, 
        default="pretrained/DOVER.pth", 
        help="path to model weights"
    )

    parser.add_argument(
        "-in",
        "--input_video_dir",
        type=str,
        default="samples",
        help="the input video dir",
    )

    parser.add_argument(
        "-d",
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="the running device"
    )

    parser.add_argument(
        "-c",
        "--calibration_mode",
        type=str,
        choices=["none", "per_batch", "pre_calibrate"],
        default="pre_calibrate",
        help="Calibration mode: none(use fixed), per_batch(calibrate per batch), pre_calibrate(calibrate first then score)"
    )

    return parser.parse_args()

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

def main():
    args = parse_arguments()
    
    with open(args.opt, "r") as f:
        opt = yaml.safe_load(f)
    
    if args.model_path:
        opt["test_load_path"] = args.model_path
        # print(f"Using model from: {args.model_path}")
    
    # print("Loading DOVER model...")
    evaluator = DOVER(**opt["model"]["args"]).to(args.device)
    evaluator.load_state_dict(
        torch.load(opt["test_load_path"], map_location=args.device)
    )
    evaluator.eval()
    
    dopt = opt["data"]["val-l1080p"]["args"]
    dopt["anno_file"] = None
    dopt["data_prefix"] = args.input_video_dir

    dataset = ViewDecompositionDataset(dopt)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=1, num_workers=opt["num_workers"], pin_memory=True,
    )

    sample_types = ["aesthetic", "technical"]
    results = []
    
    score_fuser = AdaptiveScoreFuser()
    
    if args.calibration_mode == "pre_calibrate":
        print("Pre-calibration mode: collecting raw scores...")
        raw_scores = []
        for i, data in enumerate(tqdm(dataloader, desc="Collecting raw scores")):
            if len(data.keys()) == 1:
                continue
            
            video = {}
            for key in sample_types:
                if key in data:
                    video[key] = data[key].to(args.device)
                    b, c, t, h, w = video[key].shape
                    video[key] = (
                        video[key]
                        .reshape(
                            b, c, data["num_clips"][key], t // data["num_clips"][key], h, w
                        )
                        .permute(0, 2, 1, 3, 4, 5)
                        .reshape(
                            b * data["num_clips"][key], c, t // data["num_clips"][key], h, w
                        )
                    )

            with torch.no_grad():
                results_raw = evaluator(video, reduce_scores=False)
                results_list = [np.mean(l.cpu().numpy()) for l in results_raw]
                raw_scores.append(results_list)
        
        for a_raw, t_raw in raw_scores:
            score_fuser.update_calibration(a_raw, t_raw)
        score_fuser.calibrate()
        
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=1, num_workers=opt["num_workers"], pin_memory=True,
        )
    
    for i, data in enumerate(tqdm(dataloader, desc="Testing")):
        if len(data.keys()) == 1:
            continue

        video_path = data["name"][0]
        filename = os.path.basename(video_path)
        
        initial_scale, snr_split, remaining_scale, prompt_index = parse_video_filename(filename)

        video = {}
        for key in sample_types:
            if key in data:
                video[key] = data[key].to(args.device)
                b, c, t, h, w = video[key].shape
                video[key] = (
                    video[key]
                    .reshape(
                        b, c, data["num_clips"][key], t // data["num_clips"][key], h, w
                    )
                    .permute(0, 2, 1, 3, 4, 5)
                    .reshape(
                        b * data["num_clips"][key], c, t // data["num_clips"][key], h, w
                    )
                )

        with torch.no_grad():
            results_raw = evaluator(video, reduce_scores=False)
            results_list = [np.mean(l.cpu().numpy()) for l in results_raw]

        if args.calibration_mode == "per_batch":
            score_fuser.update_calibration(results_list[0], results_list[1])
            if len(score_fuser.aesthetic_scores) >= 10 and len(score_fuser.aesthetic_scores) % 10 == 0:
                score_fuser.calibrate()

        rescaled_results = score_fuser.fuse_results(results_list)
        
        results.append({
            'prompt': prompt_index,
            'initial_scale': initial_scale,
            'snr_split': snr_split,
            'remaining_scale': remaining_scale,
            'aesthetic_score': rescaled_results["aesthetic"] * 100,
            'technical_score': rescaled_results["technical"] * 100,
            'overall_score': rescaled_results["overall"] * 100,
            'aesthetic_raw': rescaled_results["aesthetic_raw"],
            'technical_raw': rescaled_results["technical_raw"],
            'filename': filename
        })
        
        print(f"Processed: {filename} - "
              f"Aesthetic: {rescaled_results['aesthetic']*100:.2f}%, "
              f"Technical: {rescaled_results['technical']*100:.2f}%, "
              f"Overall: {rescaled_results['overall']*100:.2f}%")
    
    if args.calibration_mode == "per_batch" and not score_fuser.is_calibrated:
        score_fuser.calibrate()
    
    if results:
        excel_output_path = "results/objective_quality_result.xlsx"
        
        os.makedirs(os.path.dirname(excel_output_path), exist_ok=True)
        
        df = pd.DataFrame(results)
        
        columns_order = [
            'prompt', 'initial_scale', 'snr_split', 'remaining_scale',
            'aesthetic_score', 'technical_score', 'overall_score',
            'aesthetic_raw', 'technical_raw', 'filename'
        ]
        
        existing_columns = [col for col in columns_order if col in df.columns]
        df = df[existing_columns]
        
        df.to_excel(excel_output_path, index=False)
        print(f"Results saved to: {excel_output_path}")
        print(f"Calibration mode: {args.calibration_mode}")
        print(f"Processed {len(results)} videos successfully.")
    else:
        print("No results to save.")


if __name__ == "__main__":
    main()