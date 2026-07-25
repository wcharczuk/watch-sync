#!/usr/bin/env python3
"""Combine temporal median subtraction with backward matching.

1. Median subtraction removes static features (markers, hour/minute hands)
2. Find peaks in the residual profile
3. For each peak, verify it was at the expected position in past frames
   (backward matching at ±6°/s)
4. Track confirmed detections to get velocity estimate

This combines the best of both approaches:
- Median subtraction eliminates static feature confusion
- Backward matching confirms movement at the expected velocity
"""

import math
import os
import subprocess
import tempfile

import numpy as np
from PIL import Image

VIDEOS_DIR = os.path.join(os.path.dirname(__file__), "videos")
FPS = 30
BUFFER_SIZE = 300
NUM_RAYS = 720
SAMPLES_PER_RAY = 50
INNER_R_FRAC = 0.2
OUTER_R_FRAC = 0.9

MEDIAN_WINDOW = 90  # 3.0s for background estimation

# Backward matching on residual peaks
LOOKBACK_FRAMES = [5, 10, 15, 20, 30]  # 0.17s to 1.0s
EXPECTED_VEL = 6.0  # °/s
MATCH_TOLERANCE = 2.5  # degrees
MIN_MATCHES = 3  # out of 5 lookbacks


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


def find_residual_peaks(residual):
    """Find peaks in the median-subtracted residual."""
    n = len(residual)
    abs_res = np.abs(residual)

    # Smooth
    kernel = 3
    smoothed = np.zeros(n)
    for i in range(n):
        total = 0
        for k in range(-kernel, kernel + 1):
            total += abs_res[(i + k) % n]
        smoothed[i] = total / (2 * kernel + 1)

    # Threshold
    median_val = np.median(smoothed)
    mad = np.median(np.abs(smoothed - median_val))
    threshold = median_val + 2.0 * max(mad, 0.5)

    # Find local maxima
    raw = []
    for i in range(n):
        if (smoothed[i] > smoothed[(i - 1) % n] and
                smoothed[i] > smoothed[(i + 1) % n] and
                smoothed[i] > threshold):
            raw.append((i, smoothed[i]))

    raw.sort(key=lambda x: -x[1])

    # NMS
    kept = []
    for idx, val in raw:
        if any(min(abs(idx - ki), n - abs(idx - ki)) < 8 for ki, _ in kept):
            continue
        kept.append((idx, val))
        if len(kept) >= 10:
            break

    return kept  # list of (index, score)


def angle_dist(a, b):
    d = abs(a - b) % 360
    return min(d, 360 - d)


def backward_match_residual(current_peaks, past_residual_peaks_list, direction):
    """For each current peak, check backward matches on residual peaks.

    direction: +1 for CW, -1 for CCW
    Returns list of (angle, match_count, score) for matching peaks.
    """
    results = []

    for idx, score in current_peaks:
        angle = idx * 360.0 / NUM_RAYS
        matches = 0

        for past_peaks, lookback_frames in past_residual_peaks_list:
            if past_peaks is None:
                continue
            dt = lookback_frames / FPS
            expected_past = (angle - direction * EXPECTED_VEL * dt) % 360

            for pidx, pscore in past_peaks:
                past_angle = pidx * 360.0 / NUM_RAYS
                if angle_dist(past_angle, expected_past) < MATCH_TOLERANCE:
                    matches += 1
                    break

        if matches >= MIN_MATCHES:
            results.append((angle, matches, score))

    return results


def main():
    videos = sorted([f for f in os.listdir(VIDEOS_DIR)
                     if f.endswith((".MOV", ".mov", ".mp4"))])
    if not videos:
        print("No videos found!")
        return

    print(f"Found {len(videos)} videos")
    print(f"Median window: {MEDIAN_WINDOW} frames ({MEDIAN_WINDOW / FPS:.1f}s)")
    print(f"Lookbacks: {LOOKBACK_FRAMES} frames")
    print(f"Tolerance: ±{MATCH_TOLERANCE}°, need {MIN_MATCHES}/{len(LOOKBACK_FRAMES)} matches")
    print(f"Testing both +6°/s and -6°/s\n")

    results = []

    for video_name in videos:
        video_path = os.path.join(VIDEOS_DIR, video_name)
        print(f"=== {video_name} ===")

        with tempfile.TemporaryDirectory() as tmp_dir:
            frame_paths = extract_frames(video_path, tmp_dir, fps=FPS)
            max_frames = FPS * 6
            frame_paths = frame_paths[:max_frames]

            needed = MEDIAN_WINDOW + max(LOOKBACK_FRAMES) + 10
            if len(frame_paths) < needed:
                print(f"  Only {len(frame_paths)} frames, need {needed}, skipping\n")
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

            profiles_arr = np.array(profiles)

            # Compute residual peaks for each frame (after median window fills)
            all_residual_peaks = [None] * len(profiles)
            for fi in range(MEDIAN_WINDOW, len(profiles)):
                window_profiles = profiles_arr[fi - MEDIAN_WINDOW:fi]
                median_profile = np.median(window_profiles, axis=0)
                residual = profiles_arr[fi] - median_profile
                residual -= np.mean(residual)
                all_residual_peaks[fi] = find_residual_peaks(residual)

            # Backward matching on residual peaks
            detected_cw = []
            detected_ccw = []

            start_frame = MEDIAN_WINDOW + max(LOOKBACK_FRAMES)
            for fi in range(start_frame, len(profiles)):
                time_s = fi / FPS
                current = all_residual_peaks[fi]
                if current is None:
                    continue

                # Past residual peaks
                past_list = []
                for lb in LOOKBACK_FRAMES:
                    past_fi = fi - lb
                    if past_fi >= MEDIAN_WINDOW and all_residual_peaks[past_fi] is not None:
                        past_list.append((all_residual_peaks[past_fi], lb))
                    else:
                        past_list.append((None, lb))

                for direction, det_list, label in [
                    (+1, detected_cw, "CW"),
                    (-1, detected_ccw, "CCW"),
                ]:
                    matches = backward_match_residual(current, past_list, direction)
                    if matches:
                        best = max(matches, key=lambda x: (x[1], x[2]))
                        det_list.append((time_s, best[0]))

                if fi % 30 == 0:
                    cw_a = detected_cw[-1][1] if detected_cw and detected_cw[-1][0] == time_s else None
                    ccw_a = detected_ccw[-1][1] if detected_ccw and detected_ccw[-1][0] == time_s else None
                    print(f"  t={time_s:.1f}s: "
                          f"CW={f'{cw_a:.1f}°' if cw_a is not None else 'none':>8s}, "
                          f"CCW={f'{ccw_a:.1f}°' if ccw_a is not None else 'none':>8s}")

            # Analyze both directions
            best_result = None
            for det_list, label in [(detected_cw, "CW(+6)"), (detected_ccw, "CCW(-6)")]:
                if len(det_list) < 10:
                    print(f"\n  {label}: only {len(det_list)} detections")
                    continue

                times = np.array([t for t, a in det_list])
                angles = np.array([a for t, a in det_list])

                # Unwrap
                for i in range(1, len(angles)):
                    diff = angles[i] - angles[i - 1]
                    if diff > 180: diff -= 360
                    if diff < -180: diff += 360
                    angles[i] = angles[i - 1] + diff

                mt = np.mean(times)
                ma = np.mean(angles)
                slope = np.sum((times - mt) * (angles - ma)) / max(
                    np.sum((times - mt) ** 2), 1e-10)
                pred = ma + slope * (times - mt)
                rms = np.sqrt(np.mean((angles - pred) ** 2))

                total_frames = len(profiles) - start_frame
                coverage = len(det_list) / total_frames
                vel_err = abs(abs(slope) - 6)

                print(f"\n  {label}: {len(det_list)} detections ({coverage:.0%} coverage)")
                print(f"    Velocity: {slope:.2f}°/s, RMS: {rms:.2f}°")

                if best_result is None or vel_err < best_result[0]:
                    best_result = (vel_err, slope, rms, label, coverage)

            if best_result:
                vel_err, slope, rms, label, coverage = best_result
                if vel_err < 2 and coverage > 0.3:
                    print(f"\n  RESULT: GOOD — {label} at {slope:.1f}°/s ({coverage:.0%} coverage)")
                    results.append(("GOOD", video_name, slope))
                elif vel_err < 4 and coverage > 0.2:
                    print(f"\n  RESULT: PARTIAL — {label} at {slope:.1f}°/s ({coverage:.0%} coverage)")
                    results.append(("PARTIAL", video_name, slope))
                else:
                    print(f"\n  RESULT: POOR — best {label} at {slope:.1f}°/s ({coverage:.0%})")
                    results.append(("POOR", video_name, slope))
            else:
                print(f"\n  RESULT: FAILED — no detections")
                results.append(("FAILED", video_name, 0))

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
