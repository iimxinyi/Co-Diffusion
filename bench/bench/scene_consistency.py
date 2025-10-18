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
from third_party.vbench.utils import load_video
from third_party.tag2Text.tag2text import tag2text_caption
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, ToPILImage

tag2text_info = {
    "pretrained": f'pretrained/tag2text_swin_14m.pth',  # path to tag2text_swin_14m.pth
    "image_size": 384,
    "vit": "swin_b"
}

# os.environ["HF_HOME"] = "pretrained/bert"
# os.environ["TRANSFORMERS_CACHE"] = "pretrained/bert"
# os.environ["HF_DATASETS_CACHE"] = "pretrained/bert"
# os.environ["HF_MODULES_CACHE"] = "pretrained/bert"


def tag2text_transform(n_px):
    normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return Compose([ToPILImage(), Resize((n_px, n_px), antialias=False), ToTensor(), normalize])

def get_tags(model, image_arrays):
    caption, tag_predict = model.generate(image_arrays, tag_input=None, return_tag_predict=True)
    return tag_predict

def calculate_tag_accuracy(tags, prompt):
    if isinstance(tags, str):
        tags_list = [tag.strip() for tag in tags.split('|')]
    else:
        tags_list = tags

    if not tags or len(tags) == 0:
        return 0.0
    
    prompt_lower = prompt.lower()
    found_count = 0
    
    for tag in tags_list:
        tag_lower = tag.lower()
        if tag_lower in prompt_lower:
            found_count += 1
    
    accuracy = found_count / len(tags_list)
    return accuracy

def process_video(model, video_path, prompt, device, num_frames=16):
    transform = tag2text_transform(384)
    
    video_array = load_video(video_path, num_frames=num_frames, return_tensor=False, width=384, height=384)
    
    video_tensor_list = []
    for frame in video_array:
        video_tensor_list.append(transform(frame).to(device).unsqueeze(0))
    
    video_tensor = torch.cat(video_tensor_list)
    
    frame_tags_list = get_tags(model, video_tensor)
    
    frame_accuracies = []
    results = []
    
    for i, tags in enumerate(frame_tags_list):
        frame_accuracy = calculate_tag_accuracy(tags, prompt)
        frame_accuracies.append(frame_accuracy)
        
        results.append({
            'frame_index': i,
            'tags': tags,
            'accuracy': frame_accuracy
        })
        
        # print(f"Frame {i}:")
        # print(f"Tags: {tags}")
        # print(f"Accuracy: {frame_accuracy:.4f} ({frame_accuracy*100:.2f}%)")
        # print("-" * 50)
    
    overall_accuracy = np.mean(frame_accuracies) if frame_accuracies else 0.0
    
    return overall_accuracy, results

def compute_scene_single_video(video_path, prompt, device, num_frames=16):
    model = tag2text_caption(
        pretrained=tag2text_info["pretrained"],
        image_size=tag2text_info["image_size"],
        vit=tag2text_info["vit"]
    )
    model.eval()
    model = model.to(device)
    # print("Initialize caption model success")
    
    accuracy, frame_results = process_video(model, video_path, prompt, device, num_frames)
    
    # print(f"\n=== Overall Results ===")
    # print(f"Video: {video_path}")
    # print(f"Prompt: '{prompt}'")
    # print(f"Total frames processed: {len(frame_results)}")
    # print(f"Overall accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    return accuracy, frame_results

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
        
        accuracy, frame_results = compute_scene_single_video(video_path, prompt, device, num_frames)
            
        results.append({
            'prompt': prompt_index,
            'initial_scale': initial_scale,
            'snr_split': snr_split,
            'remaining_scale': remaining_scale,
            'scene_consistency_score': accuracy,
            'filename': filename
        })
            
        print(f"Completed: {filename} - Score: {accuracy:.4f}")

    return results

def save_results_to_excel(results, output_path):
    df = pd.DataFrame(results)
    
    columns_order = ['prompt', 'initial_scale', 'snr_split', 'remaining_scale', 'scene_consistency_score', 'filename']
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
        "A majestic wolf stands in a snowy forest, its thick fur a blend of grays and whites, glistening under the soft winter sunlight. The camera captures its piercing amber eyes, reflecting intelligence and mystery, as it surveys its surroundings with a calm, regal demeanor. Its ears twitch slightly, attuned to the faintest sounds of the forest, while its breath forms gentle clouds in the crisp air. The close-up reveals the intricate details of its fur, each strand catching the light, and the subtle movements of its powerful muscles beneath. The serene, snow-draped trees provide a tranquil backdrop, enhancing the wolf's commanding presence.",
        "In a sunlit room, a fluffy ginger cat gently licks a sleek gray tabby, both nestled on a cozy windowsill. The ginger cat's fur glows warmly in the sunlight, while the tabby purrs contentedly, eyes half-closed in bliss. The room is filled with soft, golden light filtering through sheer curtains, casting delicate patterns on the wooden floor. Outside, a garden in full bloom adds a splash of color to the serene scene. The gentle grooming continues, showcasing their bond, as the tabby occasionally nuzzles back, creating a heartwarming display of feline affection.",
        "A serene pathway meanders through a tranquil park, flanked by towering, leafless trees casting intricate shadows on the ground. The path, a mix of cobblestones and earth, is bordered by patches of melting snow, revealing the vibrant green grass beneath. Sunlight filters through the branches, creating a dappled effect on the path, while the gentle sound of dripping water from the melting snow adds a soothing rhythm to the scene. In the distance, a wooden bench invites passersby to pause and enjoy the peaceful surroundings, as birds flit between the branches, heralding the arrival of spring.",
        "A solitary wooden rowboat, painted in faded blue and white, gently drifts on a tranquil lake, its surface mirroring the soft hues of the early morning sky.  The boat, with its oars neatly resting inside, rocks slightly with the gentle ripples created by a light breeze.  Surrounding the boat, the water reflects the vibrant colors of autumn leaves from nearby trees, creating a picturesque scene of serenity and solitude.  As the camera pans, the distant silhouette of misty mountains emerges, adding depth and a sense of peaceful isolation to the idyllic setting.",
        "A vibrant scene unfolds with a pair of colorful parrots perched gracefully on an ornate bird stand, their feathers a dazzling array of greens, blues, and reds, catching the sunlight.  The stand, intricately designed with swirling patterns, stands amidst a lush garden filled with blooming flowers and verdant foliage.  The parrots, with their intelligent eyes and playful demeanor, occasionally preen their feathers or engage in soft chatter, adding a lively soundtrack to the serene setting.  As a gentle breeze rustles the leaves, the parrots' feathers shimmer, creating a mesmerizing display of nature's beauty and harmony.",
        "A young woman with long, flowing hair sits gracefully on a vintage bicycle, parked on a cobblestone street lined with quaint, colorful buildings. She wears a casual white blouse and denim shorts, exuding a relaxed, summery vibe. Her attention is focused on her smartphone, held delicately in her hand, as she leans slightly forward, engrossed in her screen. The bicycle, with its classic wicker basket, adds a charming touch to the scene. Sunlight filters through the leaves of nearby trees, casting playful shadows on the ground, creating a serene and picturesque urban moment.",
        "A majestic wild bear stands amidst a lush forest, its thick fur a rich tapestry of browns and golds, glistening under the dappled sunlight filtering through the canopy. The camera captures the bear's powerful frame, focusing on its intelligent eyes that reflect the surrounding greenery. As it sniffs the air, its wet nose glistens, and its ears twitch, attuned to the forest's symphony. The bear's massive paws rest on the soft earth, leaving imprints in the mossy ground. The scene conveys a sense of raw power and serene beauty, highlighting the bear's role as a guardian of the wilderness."
    ]
    
    video_folder = "samples"
    output_excel = "results/scene_consistency_results.xlsx"
    num_frames = 16

    results = process_all_videos_in_folder(video_folder, prompts, device, num_frames)
    
    df = save_results_to_excel(results, output_excel)
    print(f"Processed {len(results)} videos successfully.")