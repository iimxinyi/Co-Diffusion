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

os.environ["HF_HOME"] = "pretrained"
os.environ["TRANSFORMERS_CACHE"] = "pretrained"
os.environ["HF_DATASETS_CACHE"] = "pretrained"
os.environ["HF_MODULES_CACHE"] = "pretrained"

def load_clip_model(device, model_name='ViT-L/14'):
    model_path = 'pretrained'
    model, preprocess = clip.load(model_name, device=device, download_root=model_path)
    return model, preprocess

def forward_modality(model, preprocess, data, flag, device):
    if flag == 'img':
        if isinstance(data, np.ndarray):
            if data.dtype == np.float32 or data.dtype == np.float64:
                data = (data * 255).astype(np.uint8)
            data = Image.fromarray(data)
        data = preprocess(data).unsqueeze(0)
        features = model.encode_image(data.to(device))
    elif flag == 'txt':
        data = clip.tokenize([data], truncate=True)
        features = model.encode_text(data.to(device))
    else:
        raise TypeError(f"Unsupported flag: {flag}")
    return features

@torch.no_grad()
def calculate_clip_score(model, preprocess, text, image, device):
    text_features = forward_modality(model, preprocess, text, 'txt', device)
    image_features = forward_modality(model, preprocess, image, 'img', device)

    # normalize features
    text_features = text_features / text_features.norm(dim=1, keepdim=True).to(torch.float32)
    image_features = image_features / image_features.norm(dim=1, keepdim=True).to(torch.float32)

    # calculate cosine similarity
    score = (image_features * text_features).sum()
    return score.cpu().item()

def process_video(model, preprocess, video_path, prompt, device, num_frames=16):
    video_array = load_video(video_path, num_frames=num_frames, return_tensor=False, width=224, height=224)
    
    frame_scores = []
    results = []
    
    for i, frame in enumerate(video_array):
        try:
            clip_score = calculate_clip_score(model, preprocess, prompt, frame, device)
            frame_scores.append(clip_score)
            
            results.append({
                'frame_index': i,
                'clip_score': clip_score
            })
            
            # print(f"Frame {i}: CLIP Score = {clip_score:.4f}")
            # print("-" * 50)
            
        except Exception as e:
            print(f"Error processing frame {i}: {e}")
            frame_scores.append(0.0)
            results.append({
                'frame_index': i,
                'clip_score': 0.0,
                'error': str(e)
            })
    
    overall_score = np.mean(frame_scores) if frame_scores else 0.0
    
    return overall_score, results

def compute_alignment_single_video(video_path, prompt, device, num_frames=16):
    model, preprocess = load_clip_model(device)
    model.eval()
    
    overall_score, frame_results = process_video(model, preprocess, video_path, prompt, device, num_frames)
    
    # print(f"\n=== Overall Results ===")
    # print(f"Video: {video_path}")
    # print(f"Prompt: '{prompt}'")
    # print(f"Total frames processed: {len(frame_results)}")
    # print(f"Overall CLIP score: {overall_score:.4f}")
    
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
    
def process_all_videos_in_folder(folder_path, prompts, device, num_frames=16):
    video_files = glob.glob(os.path.join(folder_path, "*.mp4"))
    
    if not video_files:
        print(f"No MP4 files found in {folder_path}")
        return []
    
    results = []
    
    for video_path in video_files:
        filename = os.path.basename(video_path)
        
        initial_scale, snr_split, remaining_scale, prompt_index = parse_video_filename(filename)
        
        if prompt_index is None or prompt_index >= len(prompts):
            print(f"Invalid prompt index {prompt_index} for file {filename}, skipping...")
            continue
        
        prompt = prompts[prompt_index]
        
        clip_score, frame_results = compute_alignment_single_video(video_path, prompt, device, num_frames)
            
        results.append({
            'prompt': prompt_index,
            'initial_scale': initial_scale,
            'snr_split': snr_split,
            'remaining_scale': remaining_scale,
            'clip_score': clip_score,
            'filename': filename
        })
            
        print(f"Completed: {filename} - CLIP Score: {clip_score:.4f}")

    return results

def save_results_to_excel(results, output_path):
    df = pd.DataFrame(results)
    
    columns_order = ['prompt', 'initial_scale', 'snr_split', 'remaining_scale', 'clip_score', 'filename']
    df = df[columns_order]
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    df.to_excel(output_path, index=False)
    print(f"Results saved to: {output_path}")
    
    return df


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prompts = [
        "A fluffy-tailed squirrel perches on a moss-covered log in a sun-dappled forest clearing, its tiny paws clutching a shiny acorn. The sunlight filters through the canopy, casting playful shadows on the forest floor. The squirrel's eyes glisten with curiosity as it nibbles the nut, its whiskers twitching with each bite. Nearby, a gentle breeze rustles the autumn leaves, adding a soft, natural soundtrack to the scene. The squirrel pauses, its ears perked, listening to the distant chirping of birds, before resuming its feast, surrounded by the vibrant colors of fall foliage.",
        "A sleek, modern vehicle glides down a bustling city street, offering a dynamic view of an architectural marvel. The building, with its futuristic design, features a twisting glass facade that reflects the vibrant city lights, creating a kaleidoscope of colors. As the vehicle moves, the structure's intricate details become apparent, showcasing a blend of steel and glass that spirals upwards, defying conventional design. The surrounding urban landscape blurs slightly, emphasizing the building's unique silhouette against the evening sky. Pedestrians and other vehicles pass by, adding to the lively atmosphere of this urban scene.",
        "A stack of golden-brown pancakes, perfectly fluffy and steaming, sits on a rustic wooden table, bathed in warm morning light. Atop the stack, a generous handful of plump, juicy blueberries glisten with a light dew, their deep indigo hue contrasting beautifully with the pancakes. A drizzle of amber maple syrup cascades down the sides, pooling slightly at the base, while a dusting of powdered sugar adds a delicate touch. In the background, a soft-focus view of a cozy kitchen with vintage decor enhances the inviting, homely atmosphere, completing this mouthwatering breakfast scene.",
        "A cheerful woman with curly hair, wearing a cozy mustard sweater, sits at a rustic wooden table in a sunlit kitchen. She holds a slice of homemade apple pie on a delicate porcelain plate, its golden crust glistening with sugar crystals. The warm aroma of cinnamon and baked apples fills the air, enhancing the inviting atmosphere. Sunlight streams through a nearby window, casting a soft glow on her delighted expression. She gently lifts the pie slice with a silver fork, savoring the moment, while a steaming cup of tea and a vase of fresh daisies add charm to the scene.",
        "A beautifully arranged cabinet top showcases an eclectic mix of home decorations, including a vintage brass clock with intricate engravings, a pair of elegant porcelain vases adorned with delicate floral patterns, and a small, ornate wooden box with a polished finish. A lush, green potted plant adds a touch of nature, its leaves cascading gracefully over the edge. Nearby, a framed black-and-white photograph captures a serene landscape, while a trio of scented candles in varying heights emits a soft, warm glow. The overall composition exudes a sense of harmony and sophistication, blending classic and contemporary elements seamlessly.",
        "A majestic wolf stands in a snowy forest, its thick fur a blend of grays and whites, glistening under the soft winter sunlight. The camera captures its piercing amber eyes, reflecting intelligence and mystery, as it surveys its surroundings with a calm, regal demeanor. Its ears twitch slightly, attuned to the faintest sounds of the forest, while its breath forms gentle clouds in the crisp air. The close-up reveals the intricate details of its fur, each strand catching the light, and the subtle movements of its powerful muscles beneath. The serene, snow-draped trees provide a tranquil backdrop, enhancing the wolf's commanding presence."
    ]
    
    video_folder = "samples"
    output_excel = "results/prompt_alignment_results.xlsx"
    num_frames = 16

    results = process_all_videos_in_folder(video_folder, prompts, device, num_frames)
    
    df = save_results_to_excel(results, output_excel)
    print(f"Processed {len(results)} videos successfully.")