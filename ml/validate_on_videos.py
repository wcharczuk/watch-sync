#!/usr/bin/env python3
"""Validate the trained model against real watch videos.

For each video:
1. Extract frames at 2fps (enough to track second hand)
2. Crop center square (simulating the app's guide circle)
3. Run model inference
4. Check if predicted angles track smoothly at ~6°/s
5. Report statistics
"""

import math
import os
import subprocess
import sys
import tempfile

import numpy as np
import torch
from torchvision import transforms
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
from train import AngleNet

CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), "checkpoints", "best_model.pth")
VIDEOS_DIR = os.path.join(os.path.dirname(__file__), "videos")
FPS = 2  # Extract 2 frames per second


def extract_frames(video_path, output_dir, fps=2):
    """Extract frames from video using ffmpeg."""
    os.makedirs(output_dir, exist_ok=True)
    cmd = [
        "ffmpeg", "-i", video_path,
        "-vf", f"fps={fps}",
        "-q:v", "2",
        os.path.join(output_dir, "frame_%04d.jpg"),
        "-y", "-loglevel", "error",
    ]
    subprocess.run(cmd, check=True)
    frames = sorted([f for f in os.listdir(output_dir) if f.endswith(".jpg")])
    return [os.path.join(output_dir, f) for f in frames]


def crop_center_square(img):
    """Crop the center square from an image (simulating guide circle crop)."""
    w, h = img.size
    side = int(min(w, h) * 0.55)  # Match the app's 0.55 factor
    left = (w - side) // 2
    top = (h - side) // 2
    return img.crop((left, top, left + side, top + side))


def angle_diff(a, b):
    """Signed angular difference in degrees, wrapped to [-180, 180]."""
    d = a - b
    while d > 180:
        d -= 360
    while d < -180:
        d += 360
    return d


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading model...")
    model = AngleNet()
    state_dict = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize,
    ])

    videos = sorted([f for f in os.listdir(VIDEOS_DIR) if f.endswith((".MOV", ".mov", ".mp4"))])
    if not videos:
        print("No videos found!")
        return

    print(f"Found {len(videos)} videos\n")

    for video_name in videos:
        video_path = os.path.join(VIDEOS_DIR, video_name)
        print(f"=== {video_name} ===")

        with tempfile.TemporaryDirectory() as tmp_dir:
            frames = extract_frames(video_path, tmp_dir, fps=FPS)
            if len(frames) < 5:
                print(f"  Only {len(frames)} frames, skipping\n")
                continue

            angles = []
            for frame_path in frames:
                img = Image.open(frame_path).convert("RGB")
                img = crop_center_square(img)
                tensor = transform(img).unsqueeze(0).to(device)

                with torch.no_grad():
                    output = model(tensor)
                    sin_t = output[0, 0].item()
                    cos_t = output[0, 1].item()
                    angle = math.degrees(math.atan2(sin_t, cos_t))
                    if angle < 0:
                        angle += 360
                    angles.append(angle)

            # Analyze: compute frame-to-frame angular velocity
            dt = 1.0 / FPS
            velocities = []
            for i in range(1, len(angles)):
                diff = angle_diff(angles[i], angles[i - 1])
                vel = diff / dt  # degrees per second
                velocities.append(vel)

            velocities = np.array(velocities)

            # Expected: second hand moves at +6°/s (clockwise)
            # Allow for some jitter — check median velocity and consistency
            median_vel = np.median(velocities)
            mean_vel = np.mean(velocities)
            std_vel = np.std(velocities)

            # Count frames where velocity is roughly correct (3-9 °/s)
            correct_direction = np.sum((velocities > 2) & (velocities < 12))
            total = len(velocities)

            print(f"  Frames: {len(angles)}")
            print(f"  Angle range: {min(angles):.1f}° - {max(angles):.1f}°")
            print(f"  First 10 angles: {[f'{a:.1f}' for a in angles[:10]]}")
            print(f"  Angular velocity: median={median_vel:.1f}°/s  mean={mean_vel:.1f}°/s  std={std_vel:.1f}°/s")
            print(f"  Frames with ~6°/s velocity: {correct_direction}/{total} ({100*correct_direction/total:.0f}%)")
            print(f"  Expected: ~6°/s (second hand)")

            # Overall assessment
            if abs(median_vel - 6) < 3 and correct_direction / total > 0.5:
                print(f"  RESULT: GOOD — model appears to track the second hand")
            elif std_vel < 20:
                print(f"  RESULT: PARTIAL — some tracking but noisy")
            else:
                print(f"  RESULT: POOR — predictions appear random")
            print()


if __name__ == "__main__":
    main()
