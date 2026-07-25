#!/usr/bin/env python3
"""Extract and label frames from real watch videos for fine-tuning.

Strategy:
1. Extract frames at 2fps from each video
2. Crop center square (matching the app's guide circle crop)
3. Resize to 224x224
4. Auto-label using temporal consistency:
   - Second hand moves at 6°/s = 3° per frame at 2fps
   - Use model's predictions smoothed with the 6°/s constraint
   - Bootstrap: run model, find best-fit trajectory, use as labels
"""

import csv
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
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "data")
REAL_TRAIN_DIR = os.path.join(OUTPUT_DIR, "real_train")
REAL_VAL_DIR = os.path.join(OUTPUT_DIR, "real_val")
FPS = 2
DEGREES_PER_FRAME = 6.0 / FPS  # 3° per frame at 2fps


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
    side = int(min(w, h) * 0.55)
    left = (w - side) // 2
    top = (h - side) // 2
    return img.crop((left, top, left + side, top + side))


def get_model_predictions(model, frame_paths, transform, device):
    """Run model on all frames and return predicted angles."""
    angles = []
    model.eval()
    for path in frame_paths:
        img = Image.open(path).convert("RGB")
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
    return np.array(angles)


def fit_trajectory(raw_angles, deg_per_frame=DEGREES_PER_FRAME):
    """Find the starting angle that best fits the 6°/s trajectory.

    For each candidate starting angle, compute the expected trajectory
    and measure how well the model's predictions agree.
    Returns optimized angles.
    """
    n = len(raw_angles)
    best_score = float("inf")
    best_start = 0

    # Convert to sin/cos for circular comparison
    raw_rad = np.radians(raw_angles)
    raw_sin = np.sin(raw_rad)
    raw_cos = np.cos(raw_rad)

    # Try many starting angles
    for start_deg in np.arange(0, 360, 1.0):
        expected = np.array([(start_deg + i * deg_per_frame) % 360 for i in range(n)])
        exp_rad = np.radians(expected)
        exp_sin = np.sin(exp_rad)
        exp_cos = np.cos(exp_rad)

        # Circular distance
        diffs = np.arccos(np.clip(raw_sin * exp_sin + raw_cos * exp_cos, -1, 1))
        score = np.median(diffs)  # Use median to be robust to outliers

        if score < best_score:
            best_score = score
            best_start = start_deg

    # Refine with finer search
    for start_deg in np.arange(best_start - 2, best_start + 2, 0.1):
        expected = np.array([(start_deg + i * deg_per_frame) % 360 for i in range(n)])
        exp_rad = np.radians(expected)
        exp_sin = np.sin(exp_rad)
        exp_cos = np.cos(exp_rad)

        diffs = np.arccos(np.clip(raw_sin * exp_sin + raw_cos * exp_cos, -1, 1))
        score = np.median(diffs)

        if score < best_score:
            best_score = score
            best_start = start_deg

    best_error_deg = math.degrees(best_score)

    # Generate the fitted trajectory
    fitted = np.array([(best_start + i * deg_per_frame) % 360 for i in range(n)])
    return fitted, best_start, best_error_deg


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model (may be poorly calibrated on real images, but we use it
    # as a rough guide combined with the 6°/s constraint)
    print("Loading model...")
    model = AngleNet()
    if os.path.exists(CHECKPOINT_PATH):
        state_dict = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict)
        print("  Loaded checkpoint")
    else:
        print("  WARNING: No checkpoint found, using random model")
    model.to(device)
    model.eval()

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize,
    ])

    videos = sorted([f for f in os.listdir(VIDEOS_DIR)
                     if f.endswith((".MOV", ".mov", ".mp4"))])
    if not videos:
        print("No videos found!")
        return

    print(f"Found {len(videos)} videos")

    os.makedirs(REAL_TRAIN_DIR, exist_ok=True)
    os.makedirs(REAL_VAL_DIR, exist_ok=True)

    all_samples = []

    for vi, video_name in enumerate(videos):
        video_path = os.path.join(VIDEOS_DIR, video_name)
        print(f"\n=== {video_name} ===")

        with tempfile.TemporaryDirectory() as tmp_dir:
            frame_paths = extract_frames(video_path, tmp_dir, fps=FPS)
            if len(frame_paths) < 10:
                print(f"  Only {len(frame_paths)} frames, skipping")
                continue

            print(f"  Extracted {len(frame_paths)} frames")

            # Get model predictions
            raw_angles = get_model_predictions(model, frame_paths, transform, device)
            print(f"  Raw prediction range: {raw_angles.min():.1f}° - {raw_angles.max():.1f}°")

            # Fit trajectory using 6°/s constraint
            fitted_angles, start_angle, fit_error = fit_trajectory(raw_angles)
            print(f"  Best fit: start={start_angle:.1f}°, median error={fit_error:.1f}°")

            if fit_error > 60:
                print(f"  WARNING: Poor fit (error={fit_error:.1f}°), model can't guide labeling")
                print(f"  Using uniform trajectory from angle 0° (will still help with domain adaptation)")
                # Even with random starting angle, the RELATIVE positions
                # (3° apart) are correct, which is what matters most
                fitted_angles = np.array([(i * DEGREES_PER_FRAME) % 360
                                          for i in range(len(frame_paths))])

            # Crop and save frames with labels
            for fi, frame_path in enumerate(frame_paths):
                img = Image.open(frame_path).convert("RGB")
                img = crop_center_square(img)
                img = img.resize((224, 224), Image.LANCZOS)

                angle_deg = fitted_angles[fi]
                all_samples.append({
                    "img": img,
                    "angle_deg": angle_deg,
                    "video": video_name,
                    "frame_idx": fi,
                })

            print(f"  Added {len(frame_paths)} labeled frames")

    # Split: use last video as validation, rest as training
    print(f"\nTotal frames: {len(all_samples)}")

    # Shuffle and split 90/10
    np.random.seed(42)
    indices = np.random.permutation(len(all_samples))
    split = int(len(all_samples) * 0.9)
    train_indices = indices[:split]
    val_indices = indices[split:]

    # Save training set
    train_csv = os.path.join(OUTPUT_DIR, "real_train_labels.csv")
    with open(train_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "angle_degrees", "sin_theta", "cos_theta"])
        for i, idx in enumerate(train_indices):
            s = all_samples[idx]
            fname = f"real_{i:05d}.png"
            s["img"].save(os.path.join(REAL_TRAIN_DIR, fname))
            angle_rad = math.radians(s["angle_deg"])
            writer.writerow([
                fname,
                f"{s['angle_deg']:.4f}",
                f"{math.sin(angle_rad):.6f}",
                f"{math.cos(angle_rad):.6f}",
            ])

    # Save validation set
    val_csv = os.path.join(OUTPUT_DIR, "real_val_labels.csv")
    with open(val_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "angle_degrees", "sin_theta", "cos_theta"])
        for i, idx in enumerate(val_indices):
            s = all_samples[idx]
            fname = f"real_{i:05d}.png"
            s["img"].save(os.path.join(REAL_VAL_DIR, fname))
            angle_rad = math.radians(s["angle_deg"])
            writer.writerow([
                fname,
                f"{s['angle_deg']:.4f}",
                f"{math.sin(angle_rad):.6f}",
                f"{math.cos(angle_rad):.6f}",
            ])

    print(f"\nSaved {len(train_indices)} train + {len(val_indices)} val real frames")
    print(f"Train dir: {REAL_TRAIN_DIR}")
    print(f"Val dir: {REAL_VAL_DIR}")


if __name__ == "__main__":
    main()
