#!/usr/bin/env python3
"""Improved stabilized kymograph analysis.

Key improvements over diagnose_shake.py:
1. Higher resolution (2x - 668px crop instead of 333px)
2. Outer-ring sampling (0.5R-0.85R) - only second hand, not hour/minute
3. Column normalization - removes static features from kymograph
4. Multiple ring radii - try inner, middle, outer
5. Robust velocity refinement
"""

import os
import sys
import time

import cv2
import numpy as np

VIDEOS_DIR = os.path.join(os.path.dirname(__file__), "videos")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "diagnostics_v2")

FPS = 30
N_ANGLES = 720
CROP_FRAC = 0.55


def extract_frames_highres(video_path, max_frames=450, target_fps=30):
    """Extract frames, scale to ~2x the previous resolution."""
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
            h, w = frame.shape[:2]
            # Scale to 2160p max height (2x previous 1080p)
            if h > 2160:
                scale = 2160.0 / h
                frame = cv2.resize(frame, (int(w * scale), int(h * scale)),
                                   interpolation=cv2.INTER_AREA)
            frames.append(frame)
        frame_idx += 1

    cap.release()
    return frames


def stabilize_frames(grays, ref_idx=0):
    """Stabilize all frames relative to reference using phase correlation.

    Returns list of (dx, dy) shifts for each frame.
    """
    ref = np.float64(grays[ref_idx])
    h, w = ref.shape
    hann = cv2.createHanningWindow((w, h), cv2.CV_64F)
    ref_windowed = ref * hann

    shifts = [(0.0, 0.0)] * len(grays)
    for i in range(len(grays)):
        if i == ref_idx:
            continue
        cur_windowed = np.float64(grays[i]) * hann
        shift, response = cv2.phaseCorrelate(ref_windowed, cur_windowed)
        shifts[i] = (shift[0], shift[1])

    return shifts


def apply_shift(gray, dx, dy):
    """Shift frame by (dx, dy)."""
    h, w = gray.shape
    M = np.float32([[1, 0, -dx], [0, 1, -dy]])
    return cv2.warpAffine(gray, M, (w, h), borderMode=cv2.BORDER_REFLECT)


def extract_annular_ring(gray, cx, cy, r_inner, r_outer, n_angles=720,
                         n_radial_samples=20):
    """Extract brightness along annular ring. Returns 1D signal."""
    h, w = gray.shape
    signal = np.zeros(n_angles)

    for i in range(n_angles):
        angle_rad = np.radians(i * 360.0 / n_angles)
        dx = np.sin(angle_rad)
        dy = -np.cos(angle_rad)

        total = 0.0
        count = 0
        for s in range(n_radial_samples):
            frac = s / max(1, n_radial_samples - 1)
            r = r_inner + frac * (r_outer - r_inner)
            px = cx + dx * r
            py = cy + dy * r

            # Bilinear interpolation
            ix, iy = int(px), int(py)
            if 0 <= ix < w - 1 and 0 <= iy < h - 1:
                fx = px - ix
                fy = py - iy
                val = (gray[iy, ix] * (1 - fx) * (1 - fy) +
                       gray[iy, ix + 1] * fx * (1 - fy) +
                       gray[iy + 1, ix] * (1 - fx) * fy +
                       gray[iy + 1, ix + 1] * fx * fy)
                total += val
                count += 1

        signal[i] = total / max(count, 1)

    return signal


def build_kymograph(stabilized_grays, shifts, cx, cy, r_inner_frac,
                    r_outer_frac, n_angles=720):
    """Build kymograph from stabilized frames at given radii."""
    h, w = stabilized_grays[0].shape
    radius = min(w, h) / 2.0
    r_inner = radius * r_inner_frac
    r_outer = radius * r_outer_frac
    n_radial = max(5, int((r_outer - r_inner) / 2))

    kymograph = np.zeros((len(stabilized_grays), n_angles))
    for i, gray in enumerate(stabilized_grays):
        # Apply stabilization shift
        dx, dy = shifts[i]
        stab = apply_shift(gray, dx, dy)
        ring = extract_annular_ring(stab, cx, cy, r_inner, r_outer,
                                    n_angles, n_radial)
        kymograph[i] = ring

    return kymograph


def normalize_kymograph(kymo):
    """Remove static features by subtracting column-wise median."""
    col_median = np.median(kymo, axis=0)
    residual = kymo - col_median[np.newaxis, :]
    # Take absolute value (hand could be brighter or darker than background)
    return np.abs(residual)


def radon_line_search(kymo_norm, fps=30, n_angles=720,
                      vel_range=(-10, 10), vel_step=0.25):
    """Search for diagonal lines in kymograph.

    Returns list of (velocity_dps, score, peak_angle_idx) sorted by score.
    """
    n_t, n_a = kymo_norm.shape
    results = []

    for vel_dps in np.arange(vel_range[0], vel_range[1] + vel_step/2, vel_step):
        slope_idx_per_frame = vel_dps * n_a / (360.0 * fps)

        # Vectorized: de-rotate all frames and average
        line_sums = np.zeros(n_a)
        for t in range(n_t):
            shift = int(round(slope_idx_per_frame * t))
            line_sums += np.roll(kymo_norm[t], -shift)
        line_sums /= n_t

        # Score: peak prominence vs background
        peak_idx = np.argmax(line_sums)
        peak_val = line_sums[peak_idx]

        # Background: exclude peak region
        mask = line_sums.copy()
        for j in range(-25, 26):
            mask[(peak_idx + j) % n_a] = np.nan
        bg_median = np.nanmedian(mask)
        bg_mad = np.nanmedian(np.abs(mask - np.nanmedian(mask)))

        score = (peak_val - bg_median) / max(bg_mad, 0.001)
        results.append((vel_dps, score, peak_idx))

    results.sort(key=lambda x: -x[1])
    return results


def verify_per_frame(kymo_norm, velocity_dps, ref_angle_idx, ref_frame,
                     fps=30, n_angles=720):
    """Verify detection by checking per-frame consistency."""
    n_t = kymo_norm.shape[0]
    good = 0
    total = 0
    deviations = []

    for fi in range(n_t):
        dt = (fi - ref_frame) / fps
        predicted_deg = (ref_angle_idx * 360.0 / n_angles + velocity_dps * dt) % 360
        predicted_idx = int(round(predicted_deg * n_angles / 360.0)) % n_angles

        # Search in ±15 bins around predicted
        best_idx = predicted_idx
        best_val = kymo_norm[fi, predicted_idx]
        for j in range(-15, 16):
            idx = (predicted_idx + j) % n_angles
            if kymo_norm[fi, idx] > best_val:
                best_val = kymo_norm[fi, idx]
                best_idx = idx

        dev = (best_idx - predicted_idx) * 360.0 / n_angles
        if dev > 180:
            dev -= 360
        if dev < -180:
            dev += 360
        deviations.append(dev)
        total += 1
        if abs(dev) < 3.0:
            good += 1

    match_rate = good / max(total, 1)
    rms = np.sqrt(np.mean(np.array(deviations) ** 2)) if deviations else 999
    return match_rate, rms


def refine_velocity(kymo_norm, coarse_vel, coarse_angle_idx,
                    fps=30, n_angles=720):
    """Refine velocity estimate around the coarse value."""
    best_score = 0
    best_vel = coarse_vel
    best_angle = coarse_angle_idx

    n_t = kymo_norm.shape[0]
    ref_frame = n_t // 2

    for vel in np.arange(coarse_vel - 1.0, coarse_vel + 1.05, 0.05):
        slope = vel * n_angles / (360.0 * fps)
        line_sums = np.zeros(n_angles)
        for t in range(n_t):
            shift = int(round(slope * t))
            line_sums += np.roll(kymo_norm[t], -shift)
        line_sums /= n_t

        peak_idx = np.argmax(line_sums)
        peak_val = line_sums[peak_idx]
        mask = line_sums.copy()
        for j in range(-25, 26):
            mask[(peak_idx + j) % n_angles] = np.nan
        bg = np.nanmedian(mask)
        score = peak_val - bg

        if score > best_score:
            best_score = score
            best_vel = vel
            best_angle = peak_idx

    return best_vel, best_angle


def process_video(video_path, video_name):
    """Process a single video and return results."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"\n{'='*60}")
    print(f"  {video_name}")
    print(f"{'='*60}")

    # Extract frames at higher resolution
    t0 = time.time()
    frames = extract_frames_highres(video_path, max_frames=450, target_fps=FPS)
    print(f"  Extracted {len(frames)} frames in {time.time()-t0:.1f}s")
    print(f"  Frame shape: {frames[0].shape}")

    if len(frames) < 120:
        print("  Too few frames, skipping")
        return None

    h, w = frames[0].shape[:2]

    # Crop center
    side = int(min(w, h) * CROP_FRAC)
    x0 = (w - side) // 2
    y0 = (h - side) // 2
    print(f"  Watch crop: {side}x{side} pixels")

    # Convert to grayscale + crop
    grays = []
    for f in frames:
        crop = f[y0:y0+side, x0:x0+side]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        grays.append(gray)

    # Save first frame
    cv2.imwrite(os.path.join(OUTPUT_DIR, f"{video_name}_frame0.png"), grays[0])

    # Stabilize using phase correlation on downscaled images (faster)
    t0 = time.time()
    scale_for_stab = min(1.0, 400.0 / side)
    small_grays = [cv2.resize(g, None, fx=scale_for_stab, fy=scale_for_stab)
                   for g in grays]
    shifts_small = stabilize_frames(small_grays, ref_idx=0)
    # Scale shifts back to full resolution
    shifts = [(dx / scale_for_stab, dy / scale_for_stab)
              for dx, dy in shifts_small]
    print(f"  Stabilized in {time.time()-t0:.1f}s")

    # Report stabilization quality
    shift_mags = [np.sqrt(dx**2 + dy**2) for dx, dy in shifts]
    print(f"  Camera shake: mean={np.mean(shift_mags):.1f}px, "
          f"max={np.max(shift_mags):.1f}px")

    # Compute angular resolution
    approx_radius = side / 2.0
    deg_per_pixel = 360.0 / (2 * np.pi * approx_radius * 0.6)
    print(f"  Angular resolution: {deg_per_pixel:.3f}°/pixel at 60% radius")

    cx, cy = side / 2.0, side / 2.0

    # Try multiple ring radii
    ring_configs = [
        ("outer",  0.55, 0.85),  # Only second hand reaches here
        ("middle", 0.35, 0.65),  # All hands, but more signal
        ("inner",  0.20, 0.45),  # Inner region
    ]

    best_overall = None

    for ring_name, r_inner_frac, r_outer_frac in ring_configs:
        # Build kymograph
        t0 = time.time()
        kymo = build_kymograph(grays, shifts, cx, cy,
                               r_inner_frac, r_outer_frac, N_ANGLES)
        # Column-normalize (remove static features)
        kymo_norm = normalize_kymograph(kymo)
        elapsed = time.time() - t0

        # Radon line search
        results = radon_line_search(kymo_norm, fps=FPS, n_angles=N_ANGLES)

        # Find best result near ±6°/s (skip 0°/s which is static hands)
        best_near6 = None
        for vel, score, peak_idx in results:
            if abs(abs(vel) - 6) < 3.0:  # Within 3°/s of expected
                best_near6 = (vel, score, peak_idx)
                break

        # Also get the overall best (including 0°/s)
        top_vel, top_score, top_idx = results[0]

        print(f"\n  Ring '{ring_name}' ({r_inner_frac:.2f}R-{r_outer_frac:.2f}R) "
              f"[{elapsed:.1f}s]:")
        print(f"    Top velocity: {top_vel:+.1f}°/s (score={top_score:.1f})")

        if best_near6:
            vel6, score6, idx6 = best_near6

            # Refine velocity
            vel_refined, idx_refined = refine_velocity(
                kymo_norm, vel6, idx6, fps=FPS, n_angles=N_ANGLES)

            # Verify per-frame consistency
            ref_frame = len(kymo_norm) // 2
            match_rate, rms = verify_per_frame(
                kymo_norm, vel_refined, idx_refined, ref_frame,
                fps=FPS, n_angles=N_ANGLES)

            print(f"    Near-6 velocity: {vel_refined:+.2f}°/s "
                  f"(score={score6:.1f}, match={match_rate:.0%}, "
                  f"rms={rms:.1f}°)")

            combined_score = score6 * match_rate

            if best_overall is None or combined_score > best_overall[0]:
                best_overall = (combined_score, ring_name, vel_refined,
                                score6, match_rate, rms)

            # Save kymograph images for the outer ring
            if ring_name == "outer":
                kymo_img = cv2.normalize(kymo, None, 0, 255,
                                         cv2.NORM_MINMAX).astype(np.uint8)
                cv2.imwrite(os.path.join(OUTPUT_DIR,
                            f"{video_name}_kymo_outer.png"), kymo_img)

                kymo_norm_img = cv2.normalize(kymo_norm, None, 0, 255,
                                              cv2.NORM_MINMAX).astype(np.uint8)
                cv2.imwrite(os.path.join(OUTPUT_DIR,
                            f"{video_name}_kymo_outer_norm.png"), kymo_norm_img)
        else:
            print(f"    No velocity near ±6°/s found")

    # Summary
    print(f"\n  --- RESULT ---")
    if best_overall:
        score, ring, vel, raw_score, match, rms = best_overall
        vel_err = abs(abs(vel) - 6.0)

        if vel_err < 1.0 and match > 0.50:
            status = "GOOD"
        elif vel_err < 2.0 and match > 0.35:
            status = "PARTIAL"
        else:
            status = "POOR"

        print(f"  {status}: vel={vel:+.2f}°/s, match={match:.0%}, "
              f"rms={rms:.1f}°, ring={ring}, score={raw_score:.1f}")
        return (status, video_name, vel, match, rms, ring)
    else:
        print(f"  FAILED: no signal near 6°/s in any ring")
        return ("FAILED", video_name, 0, 0, 999, "none")


def main():
    videos = sorted([f for f in os.listdir(VIDEOS_DIR)
                     if f.upper().endswith((".MOV", ".MP4"))])
    if not videos:
        print("No videos found!")
        return

    print(f"Stabilized Kymograph Analysis v2")
    print(f"Improvements: higher res, outer ring, column normalization")
    print(f"Found {len(videos)} videos\n")

    results = []
    for video_name in videos:
        video_path = os.path.join(VIDEOS_DIR, video_name)
        result = process_video(video_path, video_name)
        if result:
            results.append(result)

    # Summary table
    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")
    print(f"{'Status':<10} {'Video':<20} {'Vel':>8} {'Match':>7} "
          f"{'RMS':>6} {'Ring':<8}")
    print(f"{'-'*10} {'-'*20} {'-'*8} {'-'*7} {'-'*6} {'-'*8}")
    for status, name, vel, match, rms, ring in results:
        print(f"{status:<10} {name:<20} {vel:>+7.2f} {match:>6.0%} "
              f"{rms:>5.1f}° {ring:<8}")

    good = sum(1 for r in results if r[0] in ("GOOD", "PARTIAL"))
    print(f"\n{good}/{len(results)} videos: GOOD or PARTIAL")


if __name__ == "__main__":
    main()
