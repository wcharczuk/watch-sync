#!/usr/bin/env python3
"""Trajectory matched filter for second hand detection.

For each frame, at each candidate angle θ, integrate the residual energy
along the expected trajectory θ(t) = θ₀ ± 6°/s * t through past frames.
The second hand's angle will have consistently high residual energy along
its trajectory; static noise won't.

This avoids fragile frame-to-frame tracking entirely.
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

MEDIAN_WINDOW = 90  # 3.0s
TRAJECTORY_LOOKBACK = 30  # 1.0s of trajectory integration
TRAJECTORY_STEP = 3  # sample every 3rd frame (every 0.1s)
EXPECTED_VEL = 6.0  # °/s


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


def compute_trajectory_score(abs_residuals, fi, direction):
    """Compute trajectory matched filter score for each angle.

    For each angle θ (index i), sum abs_residual[fi-dt][(i - dir*vel*dt/FPS * NUM_RAYS/360)]
    over several past frames dt.

    Returns score array of shape (NUM_RAYS,).
    """
    n = NUM_RAYS
    score = np.zeros(n)

    lookback_frames = list(range(TRAJECTORY_STEP, TRAJECTORY_LOOKBACK + 1, TRAJECTORY_STEP))

    for lb in lookback_frames:
        past_fi = fi - lb
        if past_fi < 0 or abs_residuals[past_fi] is None:
            continue

        dt = lb / FPS
        # Offset in indices for this lookback
        offset_deg = direction * EXPECTED_VEL * dt
        offset_idx = int(round(offset_deg * n / 360.0))

        past_residual = abs_residuals[past_fi]

        # Shift past residual by offset and add to score
        # score[i] += past_residual[(i - offset_idx) % n]
        for i in range(n):
            score[i] += past_residual[(i - offset_idx) % n]

    return score


def find_trajectory_peak(score):
    """Find the best angle from the trajectory score."""
    n = len(score)

    # Smooth
    kernel = 5
    smoothed = np.zeros(n)
    for i in range(n):
        total = 0
        for k in range(-kernel, kernel + 1):
            total += score[(i + k) % n]
        smoothed[i] = total / (2 * kernel + 1)

    # Significance check
    median_val = np.median(smoothed)
    mad = np.median(np.abs(smoothed - median_val))
    threshold = median_val + 4 * max(mad, 0.1)

    peak_idx = np.argmax(smoothed)
    peak_val = smoothed[peak_idx]

    if peak_val < threshold:
        return None, peak_val, median_val

    # Centroid refinement
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
    return angle, peak_val, median_val


def main():
    videos = sorted([f for f in os.listdir(VIDEOS_DIR)
                     if f.endswith((".MOV", ".mov", ".mp4"))])
    if not videos:
        print("No videos found!")
        return

    print(f"Found {len(videos)} videos")
    print(f"Median window: {MEDIAN_WINDOW} frames ({MEDIAN_WINDOW / FPS:.1f}s)")
    print(f"Trajectory: {TRAJECTORY_LOOKBACK} frames ({TRAJECTORY_LOOKBACK / FPS:.1f}s), "
          f"step={TRAJECTORY_STEP}")
    print(f"Testing both ±{EXPECTED_VEL}°/s\n")

    results = []

    for video_name in videos:
        video_path = os.path.join(VIDEOS_DIR, video_name)
        print(f"=== {video_name} ===")

        with tempfile.TemporaryDirectory() as tmp_dir:
            frame_paths = extract_frames(video_path, tmp_dir, fps=FPS)
            max_frames = FPS * 6
            frame_paths = frame_paths[:max_frames]

            needed = MEDIAN_WINDOW + TRAJECTORY_LOOKBACK + 10
            if len(frame_paths) < needed:
                print(f"  Only {len(frame_paths)} frames, need {needed}, skipping\n")
                continue

            print(f"  Processing {len(frame_paths)} frames ({len(frame_paths) / FPS:.1f}s)...")

            # Pre-compute profiles
            profiles = []
            for fp in frame_paths:
                img = Image.open(fp).convert("RGB")
                img = crop_center_square(img)
                buf = np.array(img.convert("L").resize(
                    (BUFFER_SIZE, BUFFER_SIZE), Image.LANCZOS), dtype=np.float64)
                profiles.append(compute_radial_profile(buf))

            profiles_arr = np.array(profiles)

            # Pre-compute abs residuals for all frames after median window
            abs_residuals = [None] * len(profiles)
            for fi in range(MEDIAN_WINDOW, len(profiles)):
                window_profiles = profiles_arr[fi - MEDIAN_WINDOW:fi]
                median_profile = np.median(window_profiles, axis=0)
                residual = profiles_arr[fi] - median_profile
                residual -= np.mean(residual)
                abs_residuals[fi] = np.abs(residual)

            # Detect second hand using trajectory filter
            detected_cw = []
            detected_ccw = []

            start_frame = MEDIAN_WINDOW + TRAJECTORY_LOOKBACK
            for fi in range(start_frame, len(profiles)):
                time_s = fi / FPS

                for direction, det_list, label in [
                    (+1, detected_cw, "CW"),
                    (-1, detected_ccw, "CCW"),
                ]:
                    score = compute_trajectory_score(abs_residuals, fi, direction)
                    angle, peak_val, median_val = find_trajectory_peak(score)

                    if angle is not None:
                        det_list.append((time_s, angle))

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
                if vel_err < 2 and rms < 10:
                    print(f"\n  RESULT: GOOD — {label} at {slope:.1f}°/s, RMS={rms:.1f}°")
                    results.append(("GOOD", video_name, slope))
                elif vel_err < 3 and rms < 15:
                    print(f"\n  RESULT: PARTIAL — {label} at {slope:.1f}°/s, RMS={rms:.1f}°")
                    results.append(("PARTIAL", video_name, slope))
                else:
                    print(f"\n  RESULT: POOR — best {label} at {slope:.1f}°/s, RMS={rms:.1f}°")
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
