#!/usr/bin/env python3
"""De-rotation stacking approach for second hand detection.

For each candidate velocity v (±6°/s), shift each frame's residual profile
by -v*t to cancel the hand's motion. Stack (sum) all shifted profiles.
If v matches the hand, it appears as a fixed strong peak in the stack.

Then for each frame, the hand angle = stack_peak + v * t.

This uses ALL frames globally, maximizing signal-to-noise ratio.
The second hand contributes coherently; noise averages out.
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


def shift_profile(profile, shift_indices):
    """Circular shift a profile by shift_indices (can be fractional)."""
    n = len(profile)
    shift_int = int(math.floor(shift_indices))
    frac = shift_indices - shift_int

    if abs(frac) < 1e-6:
        return np.roll(profile, -shift_int)

    # Linear interpolation for fractional shift
    shifted = np.zeros(n)
    for i in range(n):
        src = (i + shift_int) % n
        src_next = (src + 1) % n
        shifted[i] = profile[src] * (1 - frac) + profile[src_next] * frac
    return shifted


def derotate_and_stack(abs_residuals, start_frame, vel_dps, ref_frame=None):
    """De-rotate residuals by velocity and stack.

    Returns the stacked profile and the reference frame index used.
    """
    n = NUM_RAYS
    valid_frames = [(fi, r) for fi, r in enumerate(abs_residuals)
                    if r is not None and fi >= start_frame]

    if not valid_frames:
        return np.zeros(n), 0

    if ref_frame is None:
        ref_frame = valid_frames[len(valid_frames) // 2][0]

    stack = np.zeros(n)
    count = 0

    for fi, residual in valid_frames:
        dt = (fi - ref_frame) / FPS
        shift_deg = vel_dps * dt
        shift_idx = shift_deg * n / 360.0
        shifted = shift_profile(residual, shift_idx)
        stack += shifted
        count += 1

    if count > 0:
        stack /= count

    return stack, ref_frame


def find_stack_peak(stack):
    """Find the dominant peak in the de-rotated stack."""
    n = len(stack)

    # Smooth
    kernel = 5
    smoothed = np.zeros(n)
    for i in range(n):
        total = 0
        for k in range(-kernel, kernel + 1):
            total += stack[(i + k) % n]
        smoothed[i] = total / (2 * kernel + 1)

    # Significance
    median_val = np.median(smoothed)
    mad = np.median(np.abs(smoothed - median_val))
    threshold = median_val + 3 * max(mad, 0.1)

    peak_idx = np.argmax(smoothed)
    peak_val = smoothed[peak_idx]

    if peak_val < threshold:
        return None, peak_val / max(median_val, 0.01), smoothed

    # Second highest peak (for dominance ratio)
    # Mask around the peak
    mask = smoothed.copy()
    for j in range(-20, 21):
        mask[(peak_idx + j) % n] = 0
    second_peak = np.max(mask)
    dominance = peak_val / max(second_peak, 0.01)

    # Centroid refinement
    window = 15
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
    return angle, dominance, smoothed


def main():
    videos = sorted([f for f in os.listdir(VIDEOS_DIR)
                     if f.endswith((".MOV", ".mov", ".mp4"))])
    if not videos:
        print("No videos found!")
        return

    print(f"Found {len(videos)} videos")
    print(f"Median window: {MEDIAN_WINDOW} frames ({MEDIAN_WINDOW / FPS:.1f}s)")
    print(f"Testing velocities: +6°/s, -6°/s, and fine grid around those\n")

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

            # Pre-compute abs residuals
            abs_residuals = [None] * len(profiles)
            for fi in range(MEDIAN_WINDOW, len(profiles)):
                window_profiles = profiles_arr[fi - MEDIAN_WINDOW:fi]
                median_profile = np.median(window_profiles, axis=0)
                residual = profiles_arr[fi] - median_profile
                residual -= np.mean(residual)
                abs_residuals[fi] = np.abs(residual)

            # Try coarse velocity grid first
            best_overall = None
            coarse_velocities = [v for v in np.arange(-8, 9, 0.5) if abs(v) > 0.5]

            for vel in coarse_velocities:
                stack, ref_frame = derotate_and_stack(
                    abs_residuals, MEDIAN_WINDOW, vel)
                angle, dominance, _ = find_stack_peak(stack)

                if angle is not None:
                    if best_overall is None or dominance > best_overall[2]:
                        best_overall = (vel, angle, dominance, ref_frame)

            if best_overall is None:
                print(f"  No peak found at any velocity")
                results.append(("FAILED", video_name, 0))
                print()
                continue

            coarse_vel, coarse_angle, coarse_dom, ref_frame = best_overall
            print(f"  Coarse best: vel={coarse_vel:.1f}°/s, angle={coarse_angle:.1f}°, "
                  f"dominance={coarse_dom:.1f}x")

            # Fine search around coarse best
            fine_velocities = np.arange(coarse_vel - 0.5, coarse_vel + 0.55, 0.1)
            best_fine = None

            for vel in fine_velocities:
                stack, _ = derotate_and_stack(
                    abs_residuals, MEDIAN_WINDOW, vel, ref_frame=ref_frame)
                angle, dominance, _ = find_stack_peak(stack)

                if angle is not None:
                    if best_fine is None or dominance > best_fine[2]:
                        best_fine = (vel, angle, dominance)

            if best_fine:
                vel, angle_at_ref, dominance = best_fine
            else:
                vel, angle_at_ref, dominance = coarse_vel, coarse_angle, coarse_dom

            print(f"  Fine best: vel={vel:.1f}°/s, angle_at_ref={angle_at_ref:.1f}°, "
                  f"dominance={dominance:.1f}x")

            # Generate per-frame detections from the global model
            detected = []
            for fi in range(MEDIAN_WINDOW, len(profiles)):
                time_s = fi / FPS
                dt = (fi - ref_frame) / FPS
                predicted_angle = (angle_at_ref + vel * dt) % 360
                detected.append((time_s, predicted_angle))

            # Since all detections are on a perfect line by construction,
            # measure how well this model explains the per-frame residual data
            # by computing the average residual strength at the predicted position
            avg_strength = 0
            avg_off_strength = 0
            count = 0
            for fi in range(MEDIAN_WINDOW, len(profiles)):
                r = abs_residuals[fi]
                if r is None:
                    continue
                dt = (fi - ref_frame) / FPS
                predicted_idx = int(round((angle_at_ref + vel * dt) * NUM_RAYS / 360.0)) % NUM_RAYS
                # Average over ±3 indices
                on_sum = 0
                for j in range(-3, 4):
                    on_sum += r[(predicted_idx + j) % NUM_RAYS]
                avg_strength += on_sum / 7

                # Compare with off-peak average
                off_sum = 0
                for j in range(20, 50):
                    off_sum += r[(predicted_idx + j) % NUM_RAYS]
                    off_sum += r[(predicted_idx - j) % NUM_RAYS]
                avg_off_strength += off_sum / 60
                count += 1

            if count > 0:
                avg_strength /= count
                avg_off_strength /= count

            snr = avg_strength / max(avg_off_strength, 0.01)
            print(f"  On-peak avg: {avg_strength:.1f}, off-peak avg: {avg_off_strength:.1f}, "
                  f"SNR: {snr:.1f}x")

            # Also verify by computing per-frame residual peak near predicted position
            # and measuring how often it's within tolerance
            good_frames = 0
            deviations = []
            for fi in range(MEDIAN_WINDOW, len(profiles)):
                r = abs_residuals[fi]
                if r is None:
                    continue
                dt = (fi - ref_frame) / FPS
                predicted_deg = (angle_at_ref + vel * dt) % 360
                predicted_idx = int(round(predicted_deg * NUM_RAYS / 360.0)) % NUM_RAYS

                # Find local peak near predicted position
                best_local = predicted_idx
                best_val = r[predicted_idx]
                for j in range(-10, 11):
                    idx = (predicted_idx + j) % NUM_RAYS
                    if r[idx] > best_val:
                        best_val = r[idx]
                        best_local = idx

                local_angle = best_local * 360.0 / NUM_RAYS
                dev = local_angle - predicted_deg
                if dev > 180: dev -= 360
                if dev < -180: dev += 360
                deviations.append(dev)
                if abs(dev) < 3.0:
                    good_frames += 1

            total = len(deviations)
            match_rate = good_frames / total if total > 0 else 0
            rms_dev = np.sqrt(np.mean(np.array(deviations) ** 2)) if deviations else 999

            print(f"  Local peak match: {good_frames}/{total} ({match_rate:.0%}) within ±3°")
            print(f"  RMS deviation from prediction: {rms_dev:.1f}°")

            vel_err = abs(abs(vel) - 6)
            if vel_err < 2 and snr > 1.3 and match_rate > 0.5:
                print(f"\n  RESULT: GOOD — {vel:.1f}°/s, SNR={snr:.1f}, match={match_rate:.0%}")
                results.append(("GOOD", video_name, vel))
            elif vel_err < 3 and snr > 1.2 and match_rate > 0.3:
                print(f"\n  RESULT: PARTIAL — {vel:.1f}°/s, SNR={snr:.1f}")
                results.append(("PARTIAL", video_name, vel))
            else:
                print(f"\n  RESULT: POOR — {vel:.1f}°/s, SNR={snr:.1f}, match={match_rate:.0%}")
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
