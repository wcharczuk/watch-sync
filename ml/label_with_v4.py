#!/usr/bin/env python3
"""Label real watch frames using the v4 stabilized kymograph pipeline.

Unlike extract_real_frames.py (which bootstraps labels from the model itself),
this uses the physics-based v4 pipeline (stabilization + background subtraction +
kymograph + per-frame verification) as an independent labeling source.

For each GOOD/OK video:
1. Run v4 stabilization + kymograph pipeline at 30fps
2. Extract per-frame angles from verified frames only (match > 50%, vel error < 1°/s)
3. Crop center 55%, resize to 224x224, save with validated angle
4. Hold out 1 full video for validation (tests generalization)
5. Output to data/real_v4_train/ and data/real_v4_val/ with CSV labels

Usage:
    python label_with_v4.py
    python label_with_v4.py --val-video IMG_7710.MOV   # specify holdout video
"""

import argparse
import csv
import math
import os

import cv2
import numpy as np
from PIL import Image

# Import v4 pipeline functions
from validate_stabilized_v4 import (
    extract_frames,
    stabilize_phase_corr,
    stabilize_orb,
    apply_shift,
    apply_transform,
    compute_radial_profile,
    radon_search,
    refine_velocity,
    FPS,
    N_ANGLES,
)

VIDEOS_DIR = os.path.join(os.path.dirname(__file__), "videos")
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
TRAIN_DIR = os.path.join(DATA_DIR, "real_v4_train")
VAL_DIR = os.path.join(DATA_DIR, "real_v4_val")


def process_video_for_labels(video_path, video_name):
    """Run v4 pipeline on a video and return per-frame (angle, frame_image) pairs.

    Returns list of (frame_bgr, angle_degrees) for verified frames only.
    """
    print(f"\n{'='*60}")
    print(f"  {video_name}")
    print(f"{'='*60}")

    frames = extract_frames(video_path, max_frames=450, target_fps=FPS)
    if len(frames) < 120:
        print("  Too few frames")
        return []

    h, w = frames[0].shape[:2]
    side = int(min(w, h) * 0.55)
    x0, y0 = (w - side) // 2, (h - side) // 2
    print(f"  {len(frames)} frames, crop={side}x{side}")

    # Crop and convert to grayscale
    cropped_bgr = [f[y0:y0+side, x0:x0+side] for f in frames]
    grays = [cv2.cvtColor(c, cv2.COLOR_BGR2GRAY) for c in cropped_bgr]

    # Stabilization (same as v4)
    shifts_pc = stabilize_phase_corr(grays)
    shifts_orb, transforms_orb = stabilize_orb(grays)

    stabilized_orb = [apply_transform(grays[i], transforms_orb[i])
                      for i in range(len(grays))]
    stabilized_pc = [apply_shift(grays[i], *shifts_pc[i])
                     for i in range(len(grays))]

    # Measure residual quality
    def measure_residual_shift(stab_list):
        side_s = stab_list[0].shape[0]
        sc = min(1.0, 400.0 / side_s)
        smalls = [cv2.resize(s, None, fx=sc, fy=sc) for s in stab_list[:100]]
        h_s, w_s = smalls[0].shape
        hann = cv2.createHanningWindow((w_s, h_s), cv2.CV_64F)
        ref_f = np.float64(smalls[0]) * hann
        residuals = []
        for i in range(1, len(smalls)):
            s, _ = cv2.phaseCorrelate(ref_f, np.float64(smalls[i]) * hann)
            residuals.append(np.sqrt(s[0]**2 + s[1]**2) / sc)
        return np.mean(residuals) if residuals else 999

    res_pc = measure_residual_shift(stabilized_pc)
    res_orb = measure_residual_shift(stabilized_orb)

    if res_orb < res_pc * 0.8:
        stabilized = stabilized_orb
        stab_method = "ORB"
    else:
        stabilized = stabilized_pc
        stab_method = "PC"
    print(f"  Stabilization: {stab_method} (PC={res_pc:.2f}px, ORB={res_orb:.2f}px)")

    # Build background
    bg_frames = stabilized[30:min(180, len(stabilized))]
    background = np.median(np.array(bg_frames, dtype=np.float32), axis=0)

    cx, cy = side / 2.0, side / 2.0
    radius = side / 2.0

    # Try multiple ring/blur configurations (same as v4)
    rings = [
        ("outer", 0.50, 0.85, 25),
        ("mid",   0.30, 0.65, 25),
        ("inner", 0.20, 0.50, 20),
    ]
    blur_sizes = [0, 3, 7]

    best_result = None

    for ring_name, r_in_f, r_out_f, n_rad in rings:
        r_inner = radius * r_in_f
        r_outer = radius * r_out_f

        for blur_k in blur_sizes:
            kymo = np.zeros((len(stabilized), N_ANGLES))
            for i, stab in enumerate(stabilized):
                res2d = np.abs(np.float64(stab) - background)
                if blur_k > 0:
                    res2d = cv2.GaussianBlur(res2d, (blur_k, blur_k), 0)
                kymo[i] = compute_radial_profile(res2d, cx, cy, r_inner,
                                                  r_outer, N_ANGLES, n_rad)

            kymo_slice = kymo[30:]
            if len(kymo_slice) < 60:
                continue

            results = radon_search(kymo_slice, FPS, N_ANGLES,
                                   vel_min=4.5, vel_max=7.5)
            if not results:
                continue

            best_vel_c, best_snr, best_score, best_peak, ref_t = results[0]
            vel_r, peak_r, ref_r = refine_velocity(kymo_slice, best_vel_c,
                                                    FPS, N_ANGLES)

            # Per-frame verification with tighter threshold
            match, ts, angs, frame_indices = verify_angles_with_indices(
                kymo_slice, vel_r, peak_r, ref_r, FPS, N_ANGLES
            )

            vel_err = abs(abs(vel_r) - 6.0)
            quality = best_snr * match

            if best_result is None or quality > best_result[0]:
                best_result = (quality, ring_name, blur_k, vel_r, best_snr,
                               match, vel_err, frame_indices, angs, kymo_slice,
                               peak_r, ref_r)

    if best_result is None:
        print("  FAILED — no detection")
        return []

    (quality, ring_name, blur_k, vel_r, snr, match, vel_err,
     frame_indices, angs, kymo_slice, peak_r, ref_r) = best_result

    # Quality gate: only use GOOD/OK videos
    if vel_err > 1.0 or match < 0.50:
        print(f"  SKIPPED: vel_err={vel_err:.2f}°/s, match={match:.0%}, SNR={snr:.0f}")
        return []

    print(f"  GOOD: vel={vel_r:+.2f}°/s, match={match:.0%}, SNR={snr:.0f}, "
          f"ring={ring_name}, blur={blur_k}")

    # Now extract per-frame angles for ALL frames using the fitted model
    # (not just the verified subset — but only keep frames that pass verification)
    labeled_frames = []

    n_t = len(kymo_slice)
    for fi in range(n_t):
        # Predict angle for this frame from the kymograph model
        dt = (fi - ref_r) / FPS
        pred_deg = (peak_r * 360.0 / N_ANGLES + vel_r * dt) % 360
        pred_idx = int(round(pred_deg * N_ANGLES / 360.0)) % N_ANGLES

        # Refine against actual kymograph peak
        best_idx = pred_idx
        best_val = kymo_slice[fi, pred_idx]
        for j in range(-12, 13):
            idx = (pred_idx + j) % N_ANGLES
            if kymo_slice[fi, idx] > best_val:
                best_val = kymo_slice[fi, idx]
                best_idx = idx

        actual_deg = best_idx * 360.0 / N_ANGLES
        dev = actual_deg - pred_deg
        if dev > 180: dev -= 360
        if dev < -180: dev += 360

        # Only keep well-verified frames (deviation < 3°)
        if abs(dev) < 3.0:
            # Map kymo_slice frame index back to original frame index
            # (kymo_slice starts at frame 30)
            orig_fi = fi + 30
            if orig_fi < len(cropped_bgr):
                labeled_frames.append((cropped_bgr[orig_fi], actual_deg))

    print(f"  Verified frames: {len(labeled_frames)}/{n_t}")
    return labeled_frames


def verify_angles_with_indices(kymo, vel, peak_idx, ref_t, fps, n_angles):
    """Per-frame verification. Returns match rate, timestamps, angles, frame indices."""
    n_t = kymo.shape[0]
    good = 0
    total = 0
    ts_good = []
    ang_good = []
    idx_good = []

    for fi in range(n_t):
        dt = (fi - ref_t) / fps
        pred_deg = (peak_idx * 360.0 / n_angles + vel * dt) % 360
        pred_idx = int(round(pred_deg * n_angles / 360.0)) % n_angles

        best_idx = pred_idx
        best_val = kymo[fi, pred_idx]
        for j in range(-12, 13):
            idx = (pred_idx + j) % n_angles
            if kymo[fi, idx] > best_val:
                best_val = kymo[fi, idx]
                best_idx = idx

        actual_deg = best_idx * 360.0 / n_angles
        dev = actual_deg - pred_deg
        if dev > 180: dev -= 360
        if dev < -180: dev += 360

        total += 1
        if abs(dev) < 3.0:
            good += 1
            ts_good.append(fi / fps)
            ang_good.append(actual_deg)
            idx_good.append(fi)

    return good / max(total, 1), ts_good, ang_good, idx_good


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--val-video", type=str, default=None,
                        help="Video filename to hold out for validation (default: last video)")
    args = parser.parse_args()

    videos = sorted([f for f in os.listdir(VIDEOS_DIR)
                     if f.upper().endswith((".MOV", ".MP4"))])
    if not videos:
        print("No videos found!")
        return

    val_video = args.val_video or videos[-1]
    print(f"Videos: {len(videos)}")
    print(f"Validation holdout: {val_video}")

    os.makedirs(TRAIN_DIR, exist_ok=True)
    os.makedirs(VAL_DIR, exist_ok=True)

    train_samples = []
    val_samples = []

    for video_name in videos:
        video_path = os.path.join(VIDEOS_DIR, video_name)
        labeled = process_video_for_labels(video_path, video_name)

        if not labeled:
            continue

        if video_name == val_video:
            val_samples.extend(labeled)
            print(f"  → Validation set ({len(labeled)} frames)")
        else:
            train_samples.extend(labeled)
            print(f"  → Training set ({len(labeled)} frames)")

    print(f"\n{'='*60}")
    print(f"Total: {len(train_samples)} train, {len(val_samples)} val")

    # Save training frames
    train_csv = os.path.join(DATA_DIR, "real_v4_train_labels.csv")
    with open(train_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "angle_degrees", "sin_theta", "cos_theta"])
        for i, (frame_bgr, angle_deg) in enumerate(train_samples):
            # Crop center, resize to 224x224
            img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb).resize((224, 224), Image.LANCZOS)

            fname = f"v4_{i:05d}.png"
            pil_img.save(os.path.join(TRAIN_DIR, fname))

            angle_rad = math.radians(angle_deg)
            writer.writerow([
                fname,
                f"{angle_deg:.4f}",
                f"{math.sin(angle_rad):.6f}",
                f"{math.cos(angle_rad):.6f}",
            ])

    # Save validation frames
    val_csv = os.path.join(DATA_DIR, "real_v4_val_labels.csv")
    with open(val_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "angle_degrees", "sin_theta", "cos_theta"])
        for i, (frame_bgr, angle_deg) in enumerate(val_samples):
            img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb).resize((224, 224), Image.LANCZOS)

            fname = f"v4_{i:05d}.png"
            pil_img.save(os.path.join(VAL_DIR, fname))

            angle_rad = math.radians(angle_deg)
            writer.writerow([
                fname,
                f"{angle_deg:.4f}",
                f"{math.sin(angle_rad):.6f}",
                f"{math.cos(angle_rad):.6f}",
            ])

    print(f"\nSaved to:")
    print(f"  Train: {TRAIN_DIR} ({len(train_samples)} frames)")
    print(f"  Val:   {VAL_DIR} ({len(val_samples)} frames)")
    print(f"  Train CSV: {train_csv}")
    print(f"  Val CSV:   {val_csv}")


if __name__ == "__main__":
    main()
