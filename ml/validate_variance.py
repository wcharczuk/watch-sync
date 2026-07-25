#!/usr/bin/env python3
"""Test temporal variance approach for second hand detection.

Instead of tracking peaks, detect the second hand by finding which
angles have high temporal variance in the radial brightness profile.
Only moving features produce variance; static hands/markers don't.

The second hand sweeps ~6° per second, creating a band of high
variance. The center of this band tracks the hand's current position.
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
BG_WINDOW = 20

VARIANCE_BUFFER_FRAMES = 45  # 1.5s of profiles for variance calc


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


def to_grayscale_buffer(img):
    gray = img.convert("L").resize((BUFFER_SIZE, BUFFER_SIZE), Image.LANCZOS)
    return np.array(gray, dtype=np.float64)


def compute_radial_profile(buf):
    center = BUFFER_SIZE / 2.0
    radius = center
    inner_r = radius * INNER_R_FRAC
    outer_r = radius * OUTER_R_FRAC
    step_r = (outer_r - inner_r) / (SAMPLES_PER_RAY - 1)

    profile = np.zeros(NUM_RAYS)
    for i in range(NUM_RAYS):
        angle_deg = i * 360.0 / NUM_RAYS
        angle_rad = math.radians(angle_deg)
        dx = math.sin(angle_rad)
        dy = -math.cos(angle_rad)

        total = 0.0
        for s in range(SAMPLES_PER_RAY):
            r = inner_r + s * step_r
            px = center + dx * r
            py = center + dy * r
            # Bilinear interpolation
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


def find_variance_peak(variance_profile):
    """Find the angle with maximum temporal variance (= moving hand)."""
    n = len(variance_profile)

    # Smooth the variance profile to reduce noise
    kernel_size = 5
    smoothed = np.zeros(n)
    for i in range(n):
        total = 0
        for k in range(-kernel_size, kernel_size + 1):
            total += variance_profile[(i + k) % n]
        smoothed[i] = total / (2 * kernel_size + 1)

    # Find the peak
    peak_idx = np.argmax(smoothed)
    peak_val = smoothed[peak_idx]

    # Check if it's significant (above median + 3*MAD)
    median_var = np.median(smoothed)
    mad = np.median(np.abs(smoothed - median_var))
    threshold = median_var + 5 * max(mad, 0.1)

    if peak_val < threshold:
        return None, smoothed

    # Refine: compute centroid around peak using values above threshold
    # This gives sub-index precision
    window = 20  # ±10 indices = ±5°
    weighted_sum = 0.0
    weight_total = 0.0
    for j in range(-window, window + 1):
        idx = (peak_idx + j) % n
        w = max(smoothed[idx] - threshold, 0)
        weighted_sum += j * w
        weight_total += w

    if weight_total > 0:
        refined_idx = peak_idx + weighted_sum / weight_total
    else:
        refined_idx = peak_idx

    angle_deg = (refined_idx % n) * 360.0 / n
    return angle_deg, smoothed


def main():
    videos = sorted([f for f in os.listdir(VIDEOS_DIR)
                     if f.endswith((".MOV", ".mov", ".mp4"))])
    if not videos:
        print("No videos found!")
        return

    print(f"Found {len(videos)} videos")
    print(f"Variance buffer: {VARIANCE_BUFFER_FRAMES} frames ({VARIANCE_BUFFER_FRAMES/FPS:.1f}s)\n")

    results = []

    for video_name in videos:
        video_path = os.path.join(VIDEOS_DIR, video_name)
        print(f"=== {video_name} ===")

        with tempfile.TemporaryDirectory() as tmp_dir:
            frames = extract_frames(video_path, tmp_dir, fps=FPS)
            max_frames = FPS * 5
            frames = frames[:max_frames]

            if len(frames) < VARIANCE_BUFFER_FRAMES + 10:
                print(f"  Only {len(frames)} frames, skipping\n")
                continue

            print(f"  Processing {len(frames)} frames ({len(frames)/FPS:.1f}s)...")

            profile_buffer = deque(maxlen=VARIANCE_BUFFER_FRAMES)
            detected_angles = []

            for fi, frame_path in enumerate(frames):
                time_s = fi / FPS

                img = Image.open(frame_path).convert("RGB")
                img = crop_center_square(img)
                buf = to_grayscale_buffer(img)

                profile = compute_radial_profile(buf)
                profile_buffer.append(profile)

                if len(profile_buffer) < VARIANCE_BUFFER_FRAMES:
                    continue

                # Compute temporal variance at each angle
                profiles_arr = np.array(profile_buffer)  # (N, 720)
                variance = np.var(profiles_arr, axis=0)  # (720,)

                angle, smoothed = find_variance_peak(variance)

                if angle is not None:
                    detected_angles.append((time_s, angle))

                    if fi % 30 == 0:
                        max_var = np.max(smoothed)
                        med_var = np.median(smoothed)
                        print(f"  t={time_s:.1f}s: angle={angle:.1f}°, "
                              f"max_var={max_var:.1f}, median_var={med_var:.1f}, "
                              f"ratio={max_var/max(med_var,0.01):.1f}x")
                elif fi % 30 == 0:
                    print(f"  t={time_s:.1f}s: no significant variance peak")

            if len(detected_angles) < 10:
                print(f"\n  RESULT: FAILED — only {len(detected_angles)} detected angles")
                results.append(("FAILED", video_name, 0))
                print()
                continue

            # Analyze detected angles: compute velocity
            times = np.array([a[0] for a in detected_angles])
            angles = np.array([a[1] for a in detected_angles])

            # Unwrap angles
            for i in range(1, len(angles)):
                diff = angles[i] - angles[i - 1]
                if diff > 180: diff -= 360
                if diff < -180: diff += 360
                angles[i] = angles[i - 1] + diff

            # Linear regression for velocity
            mean_t = np.mean(times)
            mean_a = np.mean(angles)
            slope = np.sum((times - mean_t) * (angles - mean_a)) / max(np.sum((times - mean_t) ** 2), 1e-10)

            predicted = mean_a + slope * (times - mean_t)
            residuals = angles - predicted
            rms_error = np.sqrt(np.mean(residuals ** 2))

            print(f"\n  Detected angles: {len(detected_angles)} frames")
            print(f"  Velocity: {slope:.2f}°/s (expected ±6°/s)")
            print(f"  RMS residual: {rms_error:.2f}°")
            print(f"  First 5 angles: {[f'{a:.1f}' for _, a in detected_angles[:5]]}")
            print(f"  Last 5 angles: {[f'{a:.1f}' for _, a in detected_angles[-5:]]}")

            if abs(abs(slope) - 6) < 2:
                print(f"  RESULT: GOOD — tracks second hand at {slope:.1f}°/s")
                results.append(("GOOD", video_name, slope))
            elif abs(abs(slope) - 6) < 4:
                print(f"  RESULT: PARTIAL — velocity {slope:.1f}°/s is close")
                results.append(("PARTIAL", video_name, slope))
            else:
                print(f"  RESULT: POOR — velocity {slope:.1f}°/s doesn't match")
                results.append(("POOR", video_name, slope))

        print()

    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for status, name, vel in results:
        print(f"  {status:10s} {name:20s} vel={vel:+.1f}°/s")
    good = sum(1 for s, _, _ in results if s == "GOOD")
    print(f"\n  {good}/{len(results)} videos: GOOD")


if __name__ == "__main__":
    main()
