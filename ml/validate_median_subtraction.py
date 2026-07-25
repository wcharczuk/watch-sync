#!/usr/bin/env python3
"""Test temporal median subtraction for second hand detection.

Key insight: compute the median radial profile over a sliding window (~3s).
Static features (markers, hour/minute hands) appear at the median value and
cancel out. The second hand is only at each position for ~2 frames out of 90,
so the median ignores it. Subtracting the median from the current frame
reveals only the moving second hand.

This fundamentally solves the static-feature confusion problem.
"""

import math
import os
import subprocess
import tempfile
from collections import deque

import numpy as np
from PIL import Image

VIDEOS_DIR = os.path.join(os.path.dirname(__file__), "videos")
FPS = 30
BUFFER_SIZE = 300
NUM_RAYS = 720
SAMPLES_PER_RAY = 50
INNER_R_FRAC = 0.2
OUTER_R_FRAC = 0.9

MEDIAN_WINDOW = 90  # 3.0s — second hand at each 0.5° bin for ~2 frames out of 90


def extract_frames(video_path, output_dir, fps=30):
    os.makedirs(output_dir, exist_ok=True)
    cmd = [
        "ffmpeg", "-i", video_path,
        "-vf", f"fps={fps}",
        "-q:v", "2",
        os.path.join(output_dir, "frame_%06d.jpg"),
        "-y", "-loglevel", "error",
    ]
    subprocess.run(cmd, check=True)
    frames = sorted([f for f in os.listdir(output_dir) if f.endswith(".jpg")])
    return [os.path.join(output_dir, f) for f in frames]


def crop_center_square(img):
    w, h = img.size
    side = int(min(w, h) * 0.55)
    left = (w - side) // 2
    top = (h - side) // 2
    return img.crop((left, top, left + side, top + side))


def compute_radial_profile(buf):
    center = BUFFER_SIZE / 2.0
    inner_r = center * INNER_R_FRAC
    outer_r = center * OUTER_R_FRAC
    step_r = (outer_r - inner_r) / (SAMPLES_PER_RAY - 1)
    profile = np.zeros(NUM_RAYS)
    for i in range(NUM_RAYS):
        angle_rad = math.radians(i * 360.0 / NUM_RAYS)
        dx, dy = math.sin(angle_rad), -math.cos(angle_rad)
        total = 0.0
        for s in range(SAMPLES_PER_RAY):
            r = inner_r + s * step_r
            px, py = center + dx * r, center + dy * r
            fx = max(0, min(px, BUFFER_SIZE - 2))
            fy = max(0, min(py, BUFFER_SIZE - 2))
            ix, iy = int(fx), int(fy)
            ddx, ddy = fx - ix, fy - iy
            total += (buf[iy, ix] * (1 - ddx) * (1 - ddy) +
                      buf[iy, ix + 1] * ddx * (1 - ddy) +
                      buf[iy + 1, ix] * (1 - ddx) * ddy +
                      buf[iy + 1, ix + 1] * ddx * ddy)
        profile[i] = total / SAMPLES_PER_RAY
    return profile


def find_residual_peak(residual):
    """Find the dominant peak in the median-subtracted residual profile.

    The second hand shows up as a strong negative dip (dark hand against
    background) or positive peak (bright hand). We look at absolute residual.
    """
    n = len(residual)
    abs_res = np.abs(residual)

    # Smooth to reduce noise
    kernel = 5
    smoothed = np.zeros(n)
    for i in range(n):
        total = 0
        for k in range(-kernel, kernel + 1):
            total += abs_res[(i + k) % n]
        smoothed[i] = total / (2 * kernel + 1)

    # Find all local maxima
    peaks = []
    for i in range(n):
        if (smoothed[i] > smoothed[(i - 1) % n] and
                smoothed[i] > smoothed[(i + 1) % n]):
            peaks.append((i, smoothed[i]))

    if not peaks:
        return None, None, smoothed

    # Significance threshold
    median_val = np.median(smoothed)
    mad = np.median(np.abs(smoothed - median_val))
    threshold = median_val + 3 * max(mad, 0.5)

    # Filter significant peaks
    sig_peaks = [(i, v) for i, v in peaks if v > threshold]
    if not sig_peaks:
        return None, None, smoothed

    # NMS: keep top peaks with minimum 10-index separation
    sig_peaks.sort(key=lambda x: -x[1])
    kept = []
    for idx, val in sig_peaks:
        if any(min(abs(idx - ki), n - abs(idx - ki)) < 10 for ki, _ in kept):
            continue
        kept.append((idx, val))
        if len(kept) >= 5:
            break

    if not kept:
        return None, None, smoothed

    # Refine top peak with centroid
    peak_idx, peak_val = kept[0]
    window = 10
    w_sum, w_total = 0.0, 0.0
    for j in range(-window, window + 1):
        idx = (peak_idx + j) % n
        w = max(smoothed[idx] - threshold, 0)
        w_sum += j * w
        w_total += w

    if w_total > 0:
        refined = peak_idx + w_sum / w_total
    else:
        refined = peak_idx

    angle = (refined % n) * 360.0 / n
    return angle, peak_val, smoothed


def main():
    videos = sorted([f for f in os.listdir(VIDEOS_DIR)
                     if f.endswith((".MOV", ".mov", ".mp4"))])
    if not videos:
        print("No videos found!")
        return

    print(f"Found {len(videos)} videos")
    print(f"Median window: {MEDIAN_WINDOW} frames ({MEDIAN_WINDOW / FPS:.1f}s)\n")

    results = []

    for video_name in videos:
        video_path = os.path.join(VIDEOS_DIR, video_name)
        print(f"=== {video_name} ===")

        with tempfile.TemporaryDirectory() as tmp_dir:
            frame_paths = extract_frames(video_path, tmp_dir, fps=FPS)
            max_frames = FPS * 6  # Need extra for median window fill
            frame_paths = frame_paths[:max_frames]

            if len(frame_paths) < MEDIAN_WINDOW + 30:
                print(f"  Only {len(frame_paths)} frames, skipping\n")
                continue

            print(f"  Processing {len(frame_paths)} frames ({len(frame_paths) / FPS:.1f}s)...")

            # Pre-compute all radial profiles
            profiles = []
            for fp in frame_paths:
                img = Image.open(fp).convert("RGB")
                img = crop_center_square(img)
                buf = np.array(img.convert("L").resize(
                    (BUFFER_SIZE, BUFFER_SIZE), Image.LANCZOS), dtype=np.float64)
                profiles.append(compute_radial_profile(buf))

            profiles_arr = np.array(profiles)  # (N, 720)

            detected = []

            for fi in range(MEDIAN_WINDOW, len(profiles)):
                time_s = fi / FPS

                # Compute median profile over window
                window_start = fi - MEDIAN_WINDOW
                window_profiles = profiles_arr[window_start:fi]  # exclude current frame
                median_profile = np.median(window_profiles, axis=0)

                # Residual: current - median
                residual = profiles_arr[fi] - median_profile

                # Remove global offset (camera brightness changes)
                residual -= np.mean(residual)

                angle, strength, smoothed = find_residual_peak(residual)

                if angle is not None:
                    detected.append((time_s, angle, strength))

                    if fi % 30 == 0:
                        print(f"  t={time_s:.1f}s: angle={angle:.1f}°, "
                              f"strength={strength:.1f}")
                elif fi % 30 == 0:
                    print(f"  t={time_s:.1f}s: no significant peak")

            if len(detected) < 10:
                print(f"\n  RESULT: FAILED — only {len(detected)} detections")
                results.append(("FAILED", video_name, 0))
                print()
                continue

            # Analyze velocity
            times = np.array([d[0] for d in detected])
            angles = np.array([d[1] for d in detected])

            # Unwrap
            for i in range(1, len(angles)):
                diff = angles[i] - angles[i - 1]
                if diff > 180: diff -= 360
                if diff < -180: diff += 360
                angles[i] = angles[i - 1] + diff

            # Linear regression
            mean_t = np.mean(times)
            mean_a = np.mean(angles)
            slope = np.sum((times - mean_t) * (angles - mean_a)) / max(
                np.sum((times - mean_t) ** 2), 1e-10)
            predicted = mean_a + slope * (times - mean_t)
            rms = np.sqrt(np.mean((angles - predicted) ** 2))

            coverage = len(detected) / (len(profiles) - MEDIAN_WINDOW)

            print(f"\n  Detections: {len(detected)} ({coverage:.0%} coverage)")
            print(f"  Velocity: {slope:.2f}°/s (expected ±6°/s)")
            print(f"  RMS residual: {rms:.2f}°")

            vel_err = abs(abs(slope) - 6)
            if vel_err < 2 and rms < 15:
                print(f"  RESULT: GOOD — {slope:.1f}°/s")
                results.append(("GOOD", video_name, slope))
            elif vel_err < 4 and rms < 25:
                print(f"  RESULT: PARTIAL — {slope:.1f}°/s")
                results.append(("PARTIAL", video_name, slope))
            else:
                print(f"  RESULT: POOR — {slope:.1f}°/s, RMS={rms:.1f}°")
                results.append(("POOR", video_name, slope))

        print()

    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for status, name, vel in results:
        print(f"  {status:10s} {name:20s} vel={vel:+.1f}°/s")
    good = sum(1 for s, _, _ in results if s in ("GOOD", "PARTIAL"))
    print(f"\n  {good}/{len(results)} videos: GOOD or PARTIAL")


if __name__ == "__main__":
    main()
