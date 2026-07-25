#!/usr/bin/env python3
"""De-rotation v2: Fixed velocity at ±6°/s, try both raw and median-subtracted.

Instead of searching over velocity, use the known 6°/s speed.
Try de-rotation on both raw profiles and median-subtracted residuals.
Also try frame-differenced profiles (consecutive frame differences).
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

MEDIAN_WINDOW = 90


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


def derotate_and_stack(signal_frames, frame_indices, vel_dps):
    """De-rotate signals by velocity and stack.

    signal_frames: list of (frame_index, 1D_signal) pairs
    vel_dps: velocity in degrees per second
    Returns stacked profile.
    """
    n = NUM_RAYS
    if not signal_frames:
        return np.zeros(n)

    ref_fi = signal_frames[len(signal_frames) // 2][0]
    stack = np.zeros(n)

    for fi, signal in signal_frames:
        dt = (fi - ref_fi) / FPS
        shift_deg = vel_dps * dt
        shift_idx = shift_deg * n / 360.0
        shift_int = int(round(shift_idx))
        shifted = np.roll(signal, -shift_int)
        stack += shifted

    stack /= len(signal_frames)
    return stack, ref_fi


def analyze_stack(stack, label):
    """Find peak, compute dominance, return (angle, dominance, peak_val)."""
    n = len(stack)

    # Smooth
    kernel = 5
    sm = np.zeros(n)
    for i in range(n):
        total = 0
        for k in range(-kernel, kernel + 1):
            total += stack[(i + k) % n]
        sm[i] = total / (2 * kernel + 1)

    peak_idx = np.argmax(sm)
    peak_val = sm[peak_idx]

    median_val = np.median(sm)
    mad = np.median(np.abs(sm - median_val))

    # Second peak with wide exclusion
    mask = sm.copy()
    for j in range(-20, 21):
        mask[(peak_idx + j) % n] = 0
    second_peak = np.max(mask)

    dominance = (peak_val - median_val) / max(second_peak - median_val, 0.01)

    # Centroid
    threshold = median_val + 2 * max(mad, 0.1)
    window = 15
    w_sum, w_total = 0.0, 0.0
    for j in range(-window, window + 1):
        idx = (peak_idx + j) % n
        w = max(sm[idx] - threshold, 0)
        w_sum += j * w
        w_total += w

    refined = peak_idx + (w_sum / w_total if w_total > 0 else 0)
    angle = (refined % n) * 360.0 / n

    snr = peak_val / max(median_val, 0.01)

    return angle, dominance, snr, peak_val


def verify_detections(abs_residuals, start_frame, vel_dps, ref_frame, ref_angle):
    """Verify the de-rotation result by checking per-frame residual peaks."""
    n = NUM_RAYS
    good = 0
    total = 0
    deviations = []

    for fi in range(start_frame, len(abs_residuals)):
        r = abs_residuals[fi]
        if r is None:
            continue
        total += 1
        dt = (fi - ref_frame) / FPS
        predicted_deg = (ref_angle + vel_dps * dt) % 360
        predicted_idx = int(round(predicted_deg * n / 360.0)) % n

        # Find local peak
        best_idx = predicted_idx
        best_val = r[predicted_idx]
        for j in range(-10, 11):
            idx = (predicted_idx + j) % n
            if r[idx] > best_val:
                best_val = r[idx]
                best_idx = idx

        local_angle = best_idx * 360.0 / n
        dev = local_angle - predicted_deg
        if dev > 180: dev -= 360
        if dev < -180: dev += 360
        deviations.append(dev)
        if abs(dev) < 3.0:
            good += 1

    match_rate = good / total if total > 0 else 0
    rms = np.sqrt(np.mean(np.array(deviations) ** 2)) if deviations else 999
    return match_rate, rms


def main():
    videos = sorted([f for f in os.listdir(VIDEOS_DIR)
                     if f.endswith((".MOV", ".mov", ".mp4"))])
    if not videos:
        print("No videos found!")
        return

    print(f"Found {len(videos)} videos")
    print(f"Testing: raw profiles, median-subtracted, and frame-differenced")
    print(f"Fixed velocities: +6°/s and -6°/s\n")

    results = []

    for video_name in videos:
        video_path = os.path.join(VIDEOS_DIR, video_name)
        print(f"=== {video_name} ===")

        with tempfile.TemporaryDirectory() as tmp_dir:
            frame_paths = extract_frames(video_path, tmp_dir, fps=FPS)
            max_frames = FPS * 6
            frame_paths = frame_paths[:max_frames]

            if len(frame_paths) < MEDIAN_WINDOW + 30:
                print(f"  Only {len(frame_paths)} frames, skipping\n")
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

            # Prepare three signal types
            # 1. Median-subtracted residuals (abs)
            abs_residuals = [None] * len(profiles)
            residual_signals = []
            for fi in range(MEDIAN_WINDOW, len(profiles)):
                window_profiles = profiles_arr[fi - MEDIAN_WINDOW:fi]
                median_profile = np.median(window_profiles, axis=0)
                residual = profiles_arr[fi] - median_profile
                residual -= np.mean(residual)
                abs_res = np.abs(residual)
                abs_residuals[fi] = abs_res
                residual_signals.append((fi, abs_res))

            # 2. Frame differences (abs)
            diff_signals = []
            for fi in range(1, len(profiles)):
                diff = np.abs(profiles_arr[fi] - profiles_arr[fi - 1])
                diff -= np.mean(diff)
                diff = np.maximum(diff, 0)
                diff_signals.append((fi, diff))

            # 3. Raw profiles (with mean subtracted — highlights hands as deviations)
            raw_signals = []
            for fi in range(MEDIAN_WINDOW, len(profiles)):
                p = profiles_arr[fi].copy()
                p -= np.mean(p)
                raw_signals.append((fi, np.abs(p)))

            # Try all combinations
            best_combo = None

            for signal_name, signals, start in [
                ("median_residual", residual_signals, MEDIAN_WINDOW),
                ("frame_diff", diff_signals[MEDIAN_WINDOW:], MEDIAN_WINDOW),
                ("raw_abs_dev", raw_signals, MEDIAN_WINDOW),
            ]:
                if len(signals) < 30:
                    continue

                for vel in [+6.0, -6.0]:
                    stack, ref_frame = derotate_and_stack(signals, None, vel)
                    angle, dominance, snr, peak_val = analyze_stack(stack, f"{signal_name}@{vel}")

                    # Verify
                    match_rate, rms = verify_detections(
                        abs_residuals, start, vel, ref_frame, angle)

                    score = dominance * match_rate  # combined metric

                    if best_combo is None or score > best_combo[6]:
                        best_combo = (signal_name, vel, angle, dominance, snr,
                                      match_rate, score, ref_frame, rms)

                    if dominance > 1.3 or match_rate > 0.7:
                        print(f"  {signal_name} @ {vel:+.0f}°/s: "
                              f"dom={dominance:.1f}x, SNR={snr:.1f}x, "
                              f"match={match_rate:.0%}, angle={angle:.1f}°")

            if best_combo is None:
                print(f"\n  RESULT: FAILED")
                results.append(("FAILED", video_name, 0))
                print()
                continue

            name, vel, angle, dom, snr, match, score, ref, rms = best_combo
            print(f"\n  Best: {name} @ {vel:+.0f}°/s")
            print(f"    angle={angle:.1f}°, dom={dom:.1f}x, SNR={snr:.1f}x")
            print(f"    match={match:.0%}, RMS={rms:.1f}°, score={score:.2f}")

            # Now refine: search narrowly around the best velocity
            best_refined = best_combo
            for test_vel in np.arange(vel - 1.0, vel + 1.05, 0.1):
                if name == "median_residual":
                    sigs = residual_signals
                elif name == "frame_diff":
                    sigs = diff_signals[MEDIAN_WINDOW:]
                else:
                    sigs = raw_signals

                stack, ref_frame = derotate_and_stack(sigs, None, test_vel)
                angle2, dom2, snr2, _ = analyze_stack(stack, "")
                match2, rms2 = verify_detections(
                    abs_residuals, MEDIAN_WINDOW, test_vel, ref_frame, angle2)
                score2 = dom2 * match2
                if score2 > best_refined[6]:
                    best_refined = (name, test_vel, angle2, dom2, snr2,
                                    match2, score2, ref_frame, rms2)

            name, vel, angle, dom, snr, match, score, ref, rms = best_refined
            print(f"\n  Refined: {vel:+.1f}°/s, dom={dom:.1f}x, match={match:.0%}, RMS={rms:.1f}°")

            vel_err = abs(abs(vel) - 6)
            if vel_err < 2 and match > 0.5:
                print(f"  RESULT: GOOD — {vel:+.1f}°/s")
                results.append(("GOOD", video_name, vel))
            elif vel_err < 3 and match > 0.3:
                print(f"  RESULT: PARTIAL — {vel:+.1f}°/s")
                results.append(("PARTIAL", video_name, vel))
            else:
                print(f"  RESULT: POOR — {vel:+.1f}°/s, match={match:.0%}")
                results.append(("POOR", video_name, vel))

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
