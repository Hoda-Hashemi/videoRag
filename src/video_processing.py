import os
import subprocess
import torch
import clip
from PIL import Image
import json

def extract_frames(video_file, output_dir, interval=5):
    os.makedirs(output_dir, exist_ok=True)
    command = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_file
    ]
    video_duration = float(subprocess.check_output(command).strip())
    frame_timestamps = {}
    for t in range(0, int(video_duration), interval):
        output_path = os.path.join(output_dir, f"frame_{t:04d}.png")
        command = [
            "ffmpeg",
            "-ss", str(t),
            "-i", video_file,
            "-frames:v", "1",
            "-q:v", "2",
            output_path
        ]
        subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        frame_timestamps[output_path] = t
    with open(os.path.join(output_dir, "frame_timestamps.json"), "w") as f:
        json.dump(frame_timestamps, f)
    return frame_timestamps

def generate_image_embeddings(image_folder, output_file):
    device = "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    image_paths = sorted([
        os.path.join(image_folder, fname)
        for fname in os.listdir(image_folder)
        if fname.lower().endswith(('.png'))
    ])
    image_embeddings = []
    with torch.no_grad():
        for image_path in image_paths:
            image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
            embedding = model.encode_image(image)
            embedding /= embedding.norm(dim=-1, keepdim=True)
            image_embeddings.append(embedding.cpu())
    image_embeddings = torch.cat(image_embeddings, dim=0)
    torch.save(image_embeddings, output_file)
    return image_embeddings

