#!/usr/bin/env python3
"""Stabilized kymograph analysis v3.

Fixes from v2:
- Fixed reference frame bug (Radon de-rotation and verification now consistent)
- Uses 2D background subtraction for per-frame angle extraction (as in v1)
- Combines Radon velocity detection with 2D residual angle refinement
- Better ring radius selection
"""

import os
import time

import cv2
import numpy as np

VIDEOS_DIR = os.path.join(os.path.dirname(__file__), "videos")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "diagnostics_v3")

FPS = 30
N_ANGLES = 720


def extract_frames(video_path, max_frames=450, target_fps=30, max_height=2160):
    """Extract frames, optionally downscale."""
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
            if h > max_height:
                scale = max_height / h
                frame = cv2.resize(frame, (int(w * scale), int(h * scale)),
                                   interpolation=cv2.INTER_AREA)
            frames.append(frame)
        frame_idx += 1

    cap.release()
    return frames


def stabilize_phase_corr(grays, ref_idx=0):
    """Phase correlation stabilization. Returns list of (dx, dy) shifts."""
    # Work on downscaled images for speed
    side = grays[0].shape[0]
    scale = min(1.0, 400.0 / side)
    small_ref = cv2.resize(grays[ref_idx], None, fx=scale, fy=scale)
    h, w = small_ref.shape
    hann = cv2.createHanningWindow((w, h), cv2.CV_64F)
    ref_f = np.float64(small_ref) * hann

    shifts = [(0.0, 0.0)] * len(grays)
    for i in range(len(grays)):
        if i == ref_idx:
            continue
        small_cur = cv2.resize(grays[i], None, fx=scale, fy=scale)
        cur_f = np.float64(small_cur) * hann
        shift, _ = cv2.phaseCorrelate(ref_f, cur_f)
        shifts[i] = (shift[0] / scale, shift[1] / scale)

    return shifts


def apply_shift(gray, dx, dy):
    h, w = gray.shape
    M = np.float32([[1, 0, -dx], [0, 1, -dy]])
    return cv2.warpAffine(gray, M, (w, h), borderMode=cv2.BORDER_REFLECT)


def compute_radial_profile(gray, cx, cy, r_inner, r_outer,
                           n_angles=720, n_radial=20):
    """Compute average brightness along radial rays in an annular region."""
    h, w = gray.shape
    profile = np.zeros(n_angles)
    gray_f = np.float64(gray)

    for i in range(n_angles):
        angle_rad = np.radians(i * 360.0 / n_angles)
        dx = np.sin(angle_rad)
        dy = -np.cos(angle_rad)

        total = 0.0
        count = 0
        for s in range(n_radial):
            frac = s / max(1, n_radial - 1)
            r = r_inner + frac * (r_outer - r_inner)
            px = cx + dx * r
            py = cy + dy * r
            ix, iy = int(px), int(py)
            if 0 <= ix < w - 1 and 0 <= iy < h - 1:
                fx = px - ix
                fy = py - iy
                val = (gray_f[iy, ix] * (1 - fx) * (1 - fy) +
                       gray_f[iy, ix + 1] * fx * (1 - fy) +
                       gray_f[iy + 1, ix] * (1 - fx) * fy +
                       gray_f[iy + 1, ix + 1] * fx * fy)
                total += val
                count += 1

        profile[i] = total / max(count, 1)

    return profile


def compute_residual_profile(residual_img, cx, cy, r_inner, r_outer,
                             n_angles=720, n_radial=20):
    """Compute radial profile on a 2D residual image."""
    return compute_radial_profile(residual_img, cx, cy, r_inner, r_outer,
                                  n_angles, n_radial)


def radon_search(kymo_residual, fps=30, n_angles=720,
                 vel_range=(-10, 10), vel_step=0.25):
    """De-rotation search. Returns sorted list of (vel, score, peak_idx, ref_frame)."""
    n_t, n_a = kymo_residual.shape
    ref_t = n_t // 2
    results = []

    for vel in np.arange(vel_range[0], vel_range[1] + vel_step / 2, vel_step):
        slope = vel * n_a / (360.0 * fps)

        # De-rotate relative to ref_t
        stack = np.zeros(n_a)
        for t in range(n_t):
            shift = int(round(slope * (t - ref_t)))
            stack += np.roll(kymo_residual[t], -shift)
        stack /= n_t

        peak_idx = np.argmax(stack)
        peak_val = stack[peak_idx]

        # Background: exclude peak
        mask = stack.copy()
        for j in range(-25, 26):
            mask[(peak_idx + j) % n_a] = np.nan
        bg = np.nanmedian(mask)
        mad = np.nanmedian(np.abs(mask[~np.isnan(mask)] - bg))

        score = (peak_val - bg) / max(mad, 0.001)
        results.append((vel, score, peak_idx, ref_t))

    results.sort(key=lambda x: -x[1])
    return results


def refine_velocity(kymo_residual, coarse_vel, coarse_peak_idx,
                    fps=30, n_angles=720):
    """Fine-grained velocity search around coarse estimate."""
    n_t = kymo_residual.shape[0]
    ref_t = n_t // 2
    best_score = -1
    best_vel = coarse_vel
    best_peak = coarse_peak_idx

    for vel in np.arange(coarse_vel - 1.0, coarse_vel + 1.01, 0.05):
        slope = vel * n_angles / (360.0 * fps)
        stack = np.zeros(n_angles)
        for t in range(n_t):
            shift = int(round(slope * (t - ref_t)))
            stack += np.roll(kymo_residual[t], -shift)
        stack /= n_t

        peak_idx = np.argmax(stack)
        peak_val = stack[peak_idx]
        mask = stack.copy()
        for j in range(-25, 26):
            mask[(peak_idx + j) % n_angles] = np.nan
        bg = np.nanmedian(mask)
        score = peak_val - bg

        if score > best_score:
            best_score = score
            best_vel = vel
            best_peak = peak_idx

    return best_vel, best_peak, ref_t


def verify_and_extract_angles(kymo_residual, velocity, ref_angle_idx,
                              ref_frame, fps=30, n_angles=720):
    """Verify detection per-frame and extract angle time series."""
    n_t = kymo_residual.shape[0]
    angles = []
    timestamps = []
    good = 0
    total = 0

    for fi in range(n_t):
        dt = (fi - ref_frame) / fps
        predicted_deg = (ref_angle_idx * 360.0 / n_angles + velocity * dt) % 360
        predicted_idx = int(round(predicted_deg * n_angles / 360.0)) % n_angles

        # Search ±15 bins around predicted
        best_idx = predicted_idx
        best_val = kymo_residual[fi, predicted_idx]
        for j in range(-15, 16):
            idx = (predicted_idx + j) % n_angles
            if kymo_residual[fi, idx] > best_val:
                best_val = kymo_residual[fi, idx]
                best_idx = idx

        actual_deg = best_idx * 360.0 / n_angles
        dev = actual_deg - predicted_deg
        if dev > 180:
            dev -= 360
        if dev < -180:
            dev += 360

        total += 1
        if abs(dev) < 3.0:
            good += 1
            angles.append(actual_deg)
            timestamps.append(fi / fps)

    match_rate = good / max(total, 1)
    return match_rate, timestamps, angles


def process_video(video_path, video_name):
    """Process one video. Returns result tuple."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"\n{'='*60}")
    print(f"  {video_name}")
    print(f"{'='*60}")

    # Extract frames
    frames = extract_frames(video_path, max_frames=450, target_fps=FPS,
                            max_height=2160)
    print(f"  {len(frames)} frames, shape={frames[0].shape}")

    if len(frames) < 120:
        print("  Too few frames")
        return None

    h, w = frames[0].shape[:2]
    side = int(min(w, h) * 0.55)
    x0 = (w - side) // 2
    y0 = (h - side) // 2

    # Grayscale + crop
    grays = [cv2.cvtColor(f[y0:y0+side, x0:x0+side], cv2.COLOR_BGR2GRAY)
             for f in frames]
    print(f"  Crop: {side}x{side}")

    # Stabilize
    t0 = time.time()
    shifts = stabilize_phase_corr(grays, ref_idx=0)
    shift_mags = [np.sqrt(dx**2 + dy**2) for dx, dy in shifts]
    print(f"  Stabilized in {time.time()-t0:.1f}s "
          f"(shake: mean={np.mean(shift_mags):.1f}px, "
          f"max={np.max(shift_mags):.1f}px)")

    # Apply stabilization
    stabilized = [apply_shift(grays[i], *shifts[i]) for i in range(len(grays))]

    # Build 2D background (temporal median)
    t0 = time.time()
    bg_stack = np.array(stabilized[30:min(180, len(stabilized))], dtype=np.float32)
    background = np.median(bg_stack, axis=0).astype(np.uint8)
    print(f"  Background built in {time.time()-t0:.1f}s")

    # Save diagnostics
    cv2.imwrite(os.path.join(OUTPUT_DIR, f"{video_name}_frame0.png"), grays[0])
    cv2.imwrite(os.path.join(OUTPUT_DIR, f"{video_name}_background.png"),
                background)

    # Compute 2D residuals for sample frames
    for sf in [60, 120, 180]:
        if sf < len(stabilized):
            res = cv2.absdiff(stabilized[sf], background)
            res_enhanced = cv2.normalize(res, None, 0, 255, cv2.NORM_MINMAX)
            cv2.imwrite(os.path.join(OUTPUT_DIR,
                        f"{video_name}_residual2d_f{sf}.png"), res_enhanced)

    cx, cy = side / 2.0, side / 2.0
    radius = side / 2.0

    # Ring configurations to try
    ring_configs = [
        ("outer",  0.50, 0.85, 25),
        ("mid",    0.30, 0.65, 25),
        ("inner",  0.20, 0.50, 20),
    ]

    best_overall = None

    for ring_name, r_in_frac, r_out_frac, n_radial in ring_configs:
        r_inner = radius * r_in_frac
        r_outer = radius * r_out_frac

        # === Approach A: Kymograph from RAW stabilized frames ===
        # Build kymograph
        kymo_raw = np.zeros((len(stabilized), N_ANGLES))
        for i, stab in enumerate(stabilized):
            kymo_raw[i] = compute_radial_profile(stab, cx, cy, r_inner, r_outer,
                                                  N_ANGLES, n_radial)

        # Background-subtract kymograph (temporal median per column)
        kymo_bg = np.median(kymo_raw[30:min(180, len(stabilized))], axis=0)
        kymo_residual_a = np.abs(kymo_raw - kymo_bg)

        # === Approach B: Kymograph from 2D RESIDUAL images ===
        kymo_residual_b = np.zeros((len(stabilized), N_ANGLES))
        for i, stab in enumerate(stabilized):
            res2d = np.abs(np.float64(stab) - np.float64(background))
            kymo_residual_b[i] = compute_radial_profile(
                res2d.astype(np.float64), cx, cy, r_inner, r_outer,
                N_ANGLES, n_radial)

        # Try both approaches
        for approach_name, kymo_res in [("1d_sub", kymo_residual_a),
                                         ("2d_sub", kymo_residual_b)]:
            # Skip early frames (before background is reliable)
            start = 30
            kymo_slice = kymo_res[start:]

            if len(kymo_slice) < 60:
                continue

            # Radon velocity search
            radon_results = radon_search(kymo_slice, fps=FPS, n_angles=N_ANGLES)

            # Find best near ±6°/s
            best_near6 = None
            for vel, score, peak_idx, ref_t in radon_results:
                if abs(abs(vel) - 6) < 3.0:
                    best_near6 = (vel, score, peak_idx, ref_t)
                    break

            if best_near6 is None:
                continue

            vel_coarse, score_coarse, peak_coarse, ref_t = best_near6

            # Refine velocity
            vel_fine, peak_fine, ref_t = refine_velocity(
                kymo_slice, vel_coarse, peak_coarse, fps=FPS, n_angles=N_ANGLES)

            # Verify per-frame
            match_rate, ts, angles = verify_and_extract_angles(
                kymo_slice, vel_fine, peak_fine, ref_t, fps=FPS, n_angles=N_ANGLES)

            # Combined quality score
            quality = score_coarse * match_rate

            vel_err = abs(abs(vel_fine) - 6.0)

            if best_overall is None or quality > best_overall[0]:
                best_overall = (quality, ring_name, approach_name, vel_fine,
                                score_coarse, match_rate, ts, angles)

            if match_rate > 0.3 or score_coarse > 10:
                print(f"  {ring_name}/{approach_name}: vel={vel_fine:+.2f}°/s, "
                      f"radon={score_coarse:.0f}, match={match_rate:.0%}")

        # Save kymograph for outer ring
        if ring_name == "outer":
            kymo_img = cv2.normalize(kymo_residual_a, None, 0, 255,
                                     cv2.NORM_MINMAX).astype(np.uint8)
            cv2.imwrite(os.path.join(OUTPUT_DIR,
                        f"{video_name}_kymo_outer.png"), kymo_img)
            kymo_img_b = cv2.normalize(kymo_residual_b, None, 0, 255,
                                       cv2.NORM_MINMAX).astype(np.uint8)
            cv2.imwrite(os.path.join(OUTPUT_DIR,
                        f"{video_name}_kymo_outer_2d.png"), kymo_img_b)

    # Result
    print(f"\n  --- RESULT ---")
    if best_overall:
        quality, ring, approach, vel, radon, match, ts, angles = best_overall
        vel_err = abs(abs(vel) - 6.0)

        if vel_err < 1.0 and match > 0.60:
            status = "GOOD"
        elif vel_err < 1.5 and match > 0.40:
            status = "OK"
        elif vel_err < 2.0 and match > 0.30:
            status = "PARTIAL"
        else:
            status = "POOR"

        print(f"  {status}: vel={vel:+.2f}°/s, match={match:.0%}, "
              f"radon={radon:.0f}, ring={ring}, approach={approach}")

        # If we have enough matched angles, try drift estimation
        if len(angles) > 30:
            # Unwrap angles
            unwrapped = [angles[0]]
            for a in angles[1:]:
                diff = a - unwrapped[-1]
                if diff > 180:
                    diff -= 360
                if diff < -180:
                    diff += 360
                unwrapped.append(unwrapped[-1] + diff)

            ts_arr = np.array(ts)
            ang_arr = np.array(unwrapped)

            # Linear regression
            mean_t = np.mean(ts_arr)
            mean_a = np.mean(ang_arr)
            slope = (np.sum((ts_arr - mean_t) * (ang_arr - mean_a)) /
                     max(np.sum((ts_arr - mean_t)**2), 1e-10))

            residuals = ang_arr - (mean_a + slope * (ts_arr - mean_t))
            rms = np.sqrt(np.mean(residuals**2))

            print(f"  Measured velocity: {slope:.3f}°/s "
                  f"(RMS residual: {rms:.2f}°)")

        return (status, video_name, vel, match, radon, ring, approach)
    else:
        print(f"  FAILED: no signal near ±6°/s")
        return ("FAILED", video_name, 0, 0, 0, "none", "none")


def main():
    videos = sorted([f for f in os.listdir(VIDEOS_DIR)
                     if f.upper().endswith((".MOV", ".MP4"))])
    if not videos:
        print("No videos found!")
        return

    print(f"Stabilized Kymograph Analysis v3")
    print(f"Fixed reference frame, 2D residuals, higher resolution")
    print(f"Found {len(videos)} videos\n")

    results = []
    for video_name in videos:
        video_path = os.path.join(VIDEOS_DIR, video_name)
        result = process_video(video_path, video_name)
        if result:
            results.append(result)

    # Summary
    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")
    print(f"{'Status':<10} {'Video':<20} {'Vel':>8} {'Match':>7} "
          f"{'Radon':>7} {'Ring':<8} {'Method':<8}")
    print(f"{'-'*10} {'-'*20} {'-'*8} {'-'*7} {'-'*7} {'-'*8} {'-'*8}")
    for status, name, vel, match, radon, ring, approach in results:
        print(f"{status:<10} {name:<20} {vel:>+7.2f} {match:>6.0%} "
              f"{radon:>6.0f} {ring:<8} {approach:<8}")

    good = sum(1 for r in results if r[0] in ("GOOD", "OK"))
    partial = sum(1 for r in results if r[0] == "PARTIAL")
    print(f"\n{good}/{len(results)} GOOD/OK, {partial}/{len(results)} PARTIAL")


if __name__ == "__main__":
    main()
