#!/usr/bin/env python3
"""Quick diagnostic: measure camera shake and check if stabilization helps.

Extracts frames from sample videos and measures:
1. Frame-to-frame translation (camera shake magnitude)
2. Whether background subtraction after stabilization reveals the second hand
3. Saves diagnostic images for visual inspection
"""

import os
import subprocess
import tempfile
import sys

import cv2
import numpy as np
from pathlib import Path

VIDEOS_DIR = os.path.join(os.path.dirname(__file__), "videos")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "diagnostics")


def extract_frames_cv2(video_path, max_frames=450, target_fps=30):
    """Extract frames using cv2, downscale to 1080p."""
    cap = cv2.VideoCapture(video_path)
    native_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_skip = max(1, int(round(native_fps / target_fps)))

    frames = []
    frame_idx = 0
    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_skip == 0:
            # Downscale to 1080p if larger
            h, w = frame.shape[:2]
            if h > 1080:
                scale = 1080.0 / h
                frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
            frames.append(frame)
        frame_idx += 1

    cap.release()
    return frames


def estimate_translation(ref_gray, cur_gray):
    """Estimate translation between two frames using phase correlation."""
    # Ensure same size
    h, w = ref_gray.shape
    # Window to reduce edge effects
    hann = cv2.createHanningWindow((w, h), cv2.CV_64F)

    ref_f = np.float64(ref_gray) * hann
    cur_f = np.float64(cur_gray) * hann

    shift, response = cv2.phaseCorrelate(ref_f, cur_f)
    return shift, response  # (dx, dy), confidence


def stabilize_frame(frame, dx, dy):
    """Shift frame by (dx, dy) to compensate for camera motion."""
    h, w = frame.shape[:2]
    M = np.float32([[1, 0, -dx], [0, 1, -dy]])
    return cv2.warpAffine(frame, M, (w, h), borderMode=cv2.BORDER_REFLECT)


def extract_annular_ring(gray, cx, cy, r_inner, r_outer, n_angles=720):
    """Extract brightness along an annular ring, return 1D signal."""
    signal = np.zeros(n_angles)
    n_samples = max(3, int((r_outer - r_inner) / 2))

    for i in range(n_angles):
        angle_rad = np.radians(i * 360.0 / n_angles)
        dx = np.sin(angle_rad)
        dy = -np.cos(angle_rad)

        total = 0.0
        count = 0
        for s in range(n_samples):
            r = r_inner + s * (r_outer - r_inner) / max(1, n_samples - 1)
            px = cx + dx * r
            py = cy + dy * r
            ix, iy = int(round(px)), int(round(py))
            if 0 <= ix < gray.shape[1] and 0 <= iy < gray.shape[0]:
                total += gray[iy, ix]
                count += 1
        signal[i] = total / max(count, 1)

    return signal


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    videos = sorted([f for f in os.listdir(VIDEOS_DIR)
                     if f.upper().endswith((".MOV", ".MP4"))])
    if not videos:
        print("No videos found!")
        return

    print(f"Found {len(videos)} videos")
    print(f"Output directory: {OUTPUT_DIR}\n")

    for vi, video_name in enumerate(videos):
        video_path = os.path.join(VIDEOS_DIR, video_name)
        print(f"=== {video_name} ===")

        # Extract 15 seconds of frames at 30fps
        frames = extract_frames_cv2(video_path, max_frames=450, target_fps=30)
        print(f"  Extracted {len(frames)} frames, shape: {frames[0].shape}")

        if len(frames) < 60:
            print("  Too few frames, skipping\n")
            continue

        h, w = frames[0].shape[:2]
        print(f"  Frame size: {w}x{h}")

        # Crop center 55% (match existing approach)
        side = int(min(w, h) * 0.55)
        x0 = (w - side) // 2
        y0 = (h - side) // 2

        # Convert to grayscale and crop
        grays = []
        for f in frames:
            crop = f[y0:y0+side, x0:x0+side]
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            grays.append(gray)

        crop_h, crop_w = grays[0].shape
        print(f"  Cropped watch region: {crop_w}x{crop_h}")

        # === 1. Measure camera shake ===
        print("\n  --- Camera Shake Analysis ---")
        shifts_x = []
        shifts_y = []
        ref_gray = grays[0]
        for i in range(1, min(300, len(grays))):
            shift, resp = estimate_translation(
                np.float64(ref_gray), np.float64(grays[i]))
            shifts_x.append(shift[0])
            shifts_y.append(shift[1])

        shifts_x = np.array(shifts_x)
        shifts_y = np.array(shifts_y)
        total_shift = np.sqrt(shifts_x**2 + shifts_y**2)

        print(f"  Shift from frame 0 (pixels):")
        print(f"    Mean: {np.mean(total_shift):.1f}px")
        print(f"    Max:  {np.max(total_shift):.1f}px")
        print(f"    Std:  {np.std(total_shift):.1f}px")

        # Frame-to-frame shifts (jitter)
        jitter_x = np.diff(shifts_x)
        jitter_y = np.diff(shifts_y)
        jitter = np.sqrt(jitter_x**2 + jitter_y**2)
        print(f"  Frame-to-frame jitter:")
        print(f"    Mean: {np.mean(jitter):.2f}px")
        print(f"    Max:  {np.max(jitter):.2f}px")
        print(f"    Std:  {np.std(jitter):.2f}px")

        # At this crop size, how many degrees per pixel?
        approx_radius = crop_w / 2
        deg_per_pixel = 360.0 / (2 * np.pi * approx_radius * 0.6)  # at 60% radius
        print(f"  Angular resolution: {deg_per_pixel:.2f}°/pixel at 60% radius")
        print(f"  Shake as angular error: ±{np.std(total_shift) * deg_per_pixel:.1f}°")

        # === 2. Stabilize frames ===
        print("\n  --- Stabilization ---")
        stabilized_grays = [grays[0].copy()]
        cumulative_dx, cumulative_dy = 0.0, 0.0

        for i in range(1, len(grays)):
            # Estimate shift from reference (frame 0)
            shift, resp = estimate_translation(
                np.float64(grays[0]), np.float64(grays[i]))
            dx, dy = shift

            # Stabilize
            stab = stabilize_frame(grays[i], dx, dy)
            stabilized_grays.append(stab)

        # Verify stabilization worked
        post_shifts = []
        for i in range(1, min(300, len(stabilized_grays))):
            shift, _ = estimate_translation(
                np.float64(stabilized_grays[0]),
                np.float64(stabilized_grays[i]))
            post_shifts.append(np.sqrt(shift[0]**2 + shift[1]**2))
        post_shifts = np.array(post_shifts)
        print(f"  Post-stabilization residual shift:")
        print(f"    Mean: {np.mean(post_shifts):.2f}px")
        print(f"    Max:  {np.max(post_shifts):.2f}px")

        # === 3. Background subtraction ===
        print("\n  --- Background Subtraction ---")
        # Use frames 30-120 (1-4 seconds) to build median background
        bg_start, bg_end = 30, min(150, len(stabilized_grays))
        bg_stack = np.array(stabilized_grays[bg_start:bg_end], dtype=np.float32)
        background = np.median(bg_stack, axis=0).astype(np.uint8)

        # Save background image
        bg_path = os.path.join(OUTPUT_DIR, f"{video_name}_background.png")
        cv2.imwrite(bg_path, background)
        print(f"  Saved background: {bg_path}")

        # Compute residuals for a few sample frames
        sample_frames = [60, 90, 120, 150, 180]
        max_residual_val = 0
        for sf in sample_frames:
            if sf >= len(stabilized_grays):
                continue
            residual = cv2.absdiff(stabilized_grays[sf], background)
            max_residual_val = max(max_residual_val, np.max(residual))

            # Save residual (enhanced for visibility)
            residual_enhanced = cv2.normalize(residual, None, 0, 255,
                                              cv2.NORM_MINMAX)
            res_path = os.path.join(OUTPUT_DIR,
                                    f"{video_name}_residual_f{sf}.png")
            cv2.imwrite(res_path, residual_enhanced)

        # Save original frame for comparison
        orig_path = os.path.join(OUTPUT_DIR, f"{video_name}_frame0.png")
        cv2.imwrite(orig_path, grays[0])

        print(f"  Max residual value: {max_residual_val}")
        print(f"  Saved sample residuals to {OUTPUT_DIR}")

        # === 4. Build kymograph ===
        print("\n  --- Kymograph Analysis ---")
        cx, cy = crop_w / 2, crop_h / 2
        r_inner = approx_radius * 0.3
        r_outer = approx_radius * 0.7
        n_angles = 720

        kymograph = []
        for i in range(len(stabilized_grays)):
            ring = extract_annular_ring(stabilized_grays[i], cx, cy,
                                        r_inner, r_outer, n_angles)
            kymograph.append(ring)

        kymograph = np.array(kymograph)  # shape: (n_frames, n_angles)

        # Background-subtract the kymograph
        kymo_bg = np.median(kymograph[bg_start:bg_end], axis=0)
        kymo_residual = np.abs(kymograph - kymo_bg)

        # Save kymograph images
        kymo_img = cv2.normalize(kymograph, None, 0, 255,
                                 cv2.NORM_MINMAX).astype(np.uint8)
        kymo_path = os.path.join(OUTPUT_DIR, f"{video_name}_kymograph.png")
        cv2.imwrite(kymo_path, kymo_img)

        kymo_res_img = cv2.normalize(kymo_residual, None, 0, 255,
                                     cv2.NORM_MINMAX).astype(np.uint8)
        kymo_res_path = os.path.join(OUTPUT_DIR,
                                     f"{video_name}_kymograph_residual.png")
        cv2.imwrite(kymo_res_path, kymo_res_img)
        print(f"  Saved kymograph: {kymo_path}")
        print(f"  Saved kymograph residual: {kymo_res_path}")

        # === 5. Try to detect second hand via de-rotation on stabilized data ===
        print("\n  --- De-rotation on Stabilized Data ---")
        # Use median-subtracted radial profiles
        n_frames = len(kymograph)
        median_window = 90  # 3 seconds at 30fps

        best_score = 0
        best_vel = 0
        best_angle = 0

        for vel in [+6.0, -6.0]:
            # De-rotate and stack the residual kymograph
            stack = np.zeros(n_angles)
            count = 0
            ref_frame = n_frames // 2

            for fi in range(median_window, n_frames):
                dt = (fi - ref_frame) / 30.0  # 30 fps
                shift_deg = vel * dt
                shift_idx = int(round(shift_deg * n_angles / 360.0))
                shifted = np.roll(kymo_residual[fi], -shift_idx)
                stack += shifted
                count += 1

            if count > 0:
                stack /= count

            # Analyze
            peak_idx = np.argmax(stack)
            peak_val = stack[peak_idx]
            median_val = np.median(stack)
            mad = np.median(np.abs(stack - median_val))

            # Second peak (excluding main)
            mask = stack.copy()
            for j in range(-20, 21):
                mask[(peak_idx + j) % n_angles] = 0
            second_peak = np.max(mask)

            dominance = (peak_val - median_val) / max(second_peak - median_val, 0.001)
            snr = peak_val / max(median_val, 0.001)

            # Match rate check
            good = 0
            total = 0
            ref_angle_deg = peak_idx * 360.0 / n_angles
            for fi in range(median_window, n_frames):
                dt = (fi - ref_frame) / 30.0
                predicted_deg = (ref_angle_deg + vel * dt) % 360
                predicted_idx = int(round(predicted_deg * n_angles / 360.0)) % n_angles

                # Find local peak
                best_local = predicted_idx
                best_local_val = kymo_residual[fi, predicted_idx]
                for j in range(-10, 11):
                    idx = (predicted_idx + j) % n_angles
                    if kymo_residual[fi, idx] > best_local_val:
                        best_local_val = kymo_residual[fi, idx]
                        best_local = idx

                dev = (best_local - predicted_idx) * 360.0 / n_angles
                if dev > 180: dev -= 360
                if dev < -180: dev += 360
                total += 1
                if abs(dev) < 3.0:
                    good += 1

            match_rate = good / max(total, 1)
            score = dominance * match_rate

            print(f"  vel={vel:+.0f}°/s: dom={dominance:.1f}x, SNR={snr:.1f}x, "
                  f"match={match_rate:.0%}, score={score:.2f}")

            if score > best_score:
                best_score = score
                best_vel = vel
                best_angle = ref_angle_deg

        if best_score > 0.5:
            print(f"  >> DETECTED second hand at {best_vel:+.0f}°/s, "
                  f"score={best_score:.2f}")
        else:
            print(f"  >> WEAK/NO detection, best score={best_score:.2f}")

        # === 6. Try Radon-like analysis on kymograph ===
        print("\n  --- Radon Line Detection on Kymograph ---")
        # The second hand should appear as a diagonal line in the kymograph
        # with slope = 6°/s * (n_angles/360) indices per frame
        # At 30fps, that's 6/30 = 0.2°/frame = 0.4 idx/frame

        # Search over a range of slopes
        kymo_to_analyze = kymo_residual[median_window:]
        n_t = kymo_to_analyze.shape[0]
        best_radon_score = 0
        best_radon_vel = 0

        for vel_dps in np.arange(-8, 8.1, 0.25):
            # For each starting angle, sum along the line
            slope_idx_per_frame = vel_dps * n_angles / (360.0 * 30.0)

            # Compute the sum for ALL starting angles at once (vectorized)
            line_sums = np.zeros(n_angles)
            for t in range(n_t):
                shift = int(round(slope_idx_per_frame * t))
                line_sums += np.roll(kymo_to_analyze[t], -shift)
            line_sums /= n_t

            # Score = peak / background
            peak = np.max(line_sums)
            # Exclude peak region for background
            peak_idx = np.argmax(line_sums)
            bg_mask = line_sums.copy()
            for j in range(-20, 21):
                bg_mask[(peak_idx + j) % n_angles] = np.nan
            bg = np.nanmedian(bg_mask)
            score = (peak - bg) / max(bg, 0.001)

            if score > best_radon_score:
                best_radon_score = score
                best_radon_vel = vel_dps

        print(f"  Best velocity: {best_radon_vel:+.1f}°/s, "
              f"score={best_radon_score:.2f}")
        vel_err = abs(abs(best_radon_vel) - 6)
        if vel_err < 1.5 and best_radon_score > 0.3:
            print(f"  >> GOOD: Radon detected second hand at "
                  f"{best_radon_vel:+.1f}°/s")
        elif vel_err < 3 and best_radon_score > 0.15:
            print(f"  >> PARTIAL: Radon suggests hand at "
                  f"{best_radon_vel:+.1f}°/s")
        else:
            print(f"  >> POOR: vel={best_radon_vel:+.1f}°/s, "
                  f"score={best_radon_score:.2f}")

        print()

    print("=" * 60)
    print(f"Diagnostic images saved to: {OUTPUT_DIR}")
    print("Check the _kymograph_residual.png files — the second hand should")
    print("appear as a DIAGONAL line (slope = ~6°/s).")
    print("Static hands appear as VERTICAL lines.")


if __name__ == "__main__":
    main()
