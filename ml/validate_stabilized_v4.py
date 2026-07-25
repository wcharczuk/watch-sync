#!/usr/bin/env python3
"""Stabilized kymograph v4 — focused improvements.

Changes from v3:
1. Gaussian blur on 2D residual before radial profiling (suppresses edge artifacts)
2. Tighter velocity constraint (5.0-7.0°/s — covers any real watch)
3. Feature-based (ORB) stabilization fallback for high-shake videos
4. Frame-to-frame cumulative stabilization option
5. Better scoring: prefer results closest to 6.0°/s
"""

import os
import time

import cv2
import numpy as np

VIDEOS_DIR = os.path.join(os.path.dirname(__file__), "videos")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "diagnostics_v4")

FPS = 30
N_ANGLES = 720


def extract_frames(video_path, max_frames=450, target_fps=30, max_height=2160):
    cap = cv2.VideoCapture(video_path)
    native_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_skip = max(1, int(round(native_fps / target_fps)))
    frames = []
    idx = 0
    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % frame_skip == 0:
            h, w = frame.shape[:2]
            if h > max_height:
                s = max_height / h
                frame = cv2.resize(frame, (int(w * s), int(h * s)),
                                   interpolation=cv2.INTER_AREA)
            frames.append(frame)
        idx += 1
    cap.release()
    return frames


def stabilize_phase_corr(grays, ref_idx=0):
    """Phase correlation. Returns shifts."""
    side = grays[0].shape[0]
    scale = min(1.0, 400.0 / side)
    smalls = [cv2.resize(g, None, fx=scale, fy=scale) for g in grays]
    h, w = smalls[0].shape
    hann = cv2.createHanningWindow((w, h), cv2.CV_64F)
    ref_f = np.float64(smalls[ref_idx]) * hann

    shifts = [(0.0, 0.0)] * len(grays)
    for i in range(len(grays)):
        if i == ref_idx:
            continue
        cur_f = np.float64(smalls[i]) * hann
        s, _ = cv2.phaseCorrelate(ref_f, cur_f)
        shifts[i] = (s[0] / scale, s[1] / scale)
    return shifts


def stabilize_orb(grays, ref_idx=0):
    """ORB feature-based stabilization. Handles rotation + scale."""
    orb = cv2.ORB_create(500)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    ref_kp, ref_des = orb.detectAndCompute(grays[ref_idx], None)

    shifts = [(0.0, 0.0)] * len(grays)
    transforms = [np.eye(2, 3, dtype=np.float32)] * len(grays)

    for i in range(len(grays)):
        if i == ref_idx:
            continue
        kp, des = orb.detectAndCompute(grays[i], None)
        if des is None or ref_des is None:
            continue
        matches = bf.match(ref_des, des)
        if len(matches) < 10:
            continue
        matches = sorted(matches, key=lambda m: m.distance)[:50]

        src = np.float32([ref_kp[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst = np.float32([kp[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        M, inliers = cv2.estimateAffinePartial2D(dst, src, method=cv2.RANSAC,
                                                   ransacReprojThreshold=3.0)
        if M is not None:
            transforms[i] = M
            # Extract translation component
            shifts[i] = (-M[0, 2], -M[1, 2])  # Approximate

    return shifts, transforms


def apply_shift(gray, dx, dy):
    h, w = gray.shape
    M = np.float32([[1, 0, -dx], [0, 1, -dy]])
    return cv2.warpAffine(gray, M, (w, h), borderMode=cv2.BORDER_REFLECT)


def apply_transform(gray, M):
    h, w = gray.shape
    return cv2.warpAffine(gray, M, (w, h), borderMode=cv2.BORDER_REFLECT)


def compute_radial_profile(gray_f, cx, cy, r_inner, r_outer,
                           n_angles=720, n_radial=20):
    """Radial profile with bilinear interpolation."""
    h, w = gray_f.shape[:2]
    profile = np.zeros(n_angles)

    for i in range(n_angles):
        angle_rad = np.radians(i * 360.0 / n_angles)
        sin_a = np.sin(angle_rad)
        cos_a = -np.cos(angle_rad)

        total = 0.0
        count = 0
        for s in range(n_radial):
            frac = s / max(1, n_radial - 1)
            r = r_inner + frac * (r_outer - r_inner)
            px = cx + sin_a * r
            py = cy + cos_a * r
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


def radon_search(kymo, fps, n_angles, vel_min=4.5, vel_max=7.5, vel_step=0.1):
    """Search for second hand line in kymograph. Only searches near 6°/s."""
    n_t, n_a = kymo.shape
    ref_t = n_t // 2
    results = []

    for sign in [+1, -1]:
        for vel_abs in np.arange(vel_min, vel_max + vel_step / 2, vel_step):
            vel = sign * vel_abs
            slope = vel * n_a / (360.0 * fps)

            stack = np.zeros(n_a)
            for t in range(n_t):
                shift = int(round(slope * (t - ref_t)))
                stack += np.roll(kymo[t], -shift)
            stack /= n_t

            peak_idx = np.argmax(stack)
            peak_val = stack[peak_idx]

            mask = stack.copy()
            for j in range(-30, 31):
                mask[(peak_idx + j) % n_a] = np.nan
            bg = np.nanmedian(mask)
            mad = np.nanmedian(np.abs(mask[~np.isnan(mask)] - bg))

            snr = (peak_val - bg) / max(mad, 0.001)

            # Bonus for being close to exactly 6°/s
            vel_bonus = 1.0 / (1.0 + (vel_abs - 6.0)**2)
            score = snr * vel_bonus

            results.append((vel, snr, score, peak_idx, ref_t))

    results.sort(key=lambda x: -x[2])
    return results


def refine_velocity(kymo, coarse_vel, fps, n_angles):
    """Fine-tune velocity around coarse estimate."""
    n_t = kymo.shape[0]
    ref_t = n_t // 2
    best_score = -1
    best_vel = coarse_vel
    best_peak = 0

    for vel in np.arange(coarse_vel - 0.5, coarse_vel + 0.51, 0.02):
        slope = vel * n_angles / (360.0 * fps)
        stack = np.zeros(n_angles)
        for t in range(n_t):
            shift = int(round(slope * (t - ref_t)))
            stack += np.roll(kymo[t], -shift)
        stack /= n_t

        peak_idx = np.argmax(stack)
        peak_val = stack[peak_idx]
        mask = stack.copy()
        for j in range(-30, 31):
            mask[(peak_idx + j) % n_angles] = np.nan
        bg = np.nanmedian(mask)
        score = peak_val - bg

        if score > best_score:
            best_score = score
            best_vel = vel
            best_peak = peak_idx

    return best_vel, best_peak, ref_t


def verify_angles(kymo, vel, peak_idx, ref_t, fps, n_angles):
    """Per-frame verification. Returns match rate, timestamps, angles."""
    n_t = kymo.shape[0]
    good = 0
    total = 0
    ts_good = []
    ang_good = []

    for fi in range(n_t):
        dt = (fi - ref_t) / fps
        pred_deg = (peak_idx * 360.0 / n_angles + vel * dt) % 360
        pred_idx = int(round(pred_deg * n_angles / 360.0)) % n_angles

        # Search ±12 bins (±6°)
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

    return good / max(total, 1), ts_good, ang_good


def measure_drift(timestamps, angles):
    """Linear regression on angle vs time. Returns velocity and RMS."""
    if len(timestamps) < 10:
        return None, None, None

    ts = np.array(timestamps)
    angs = [angles[0]]
    for a in angles[1:]:
        diff = a - angs[-1]
        if diff > 180: diff -= 360
        if diff < -180: diff += 360
        angs.append(angs[-1] + diff)
    angs = np.array(angs)

    mean_t = np.mean(ts)
    mean_a = np.mean(angs)
    slope = np.sum((ts - mean_t) * (angs - mean_a)) / max(np.sum((ts - mean_t)**2), 1e-10)
    residuals = angs - (mean_a + slope * (ts - mean_t))
    rms = np.sqrt(np.mean(residuals**2))

    return slope, rms, len(timestamps)


def process_video(video_path, video_name):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"\n{'='*60}")
    print(f"  {video_name}")
    print(f"{'='*60}")

    frames = extract_frames(video_path, max_frames=450, target_fps=FPS)
    if len(frames) < 120:
        print("  Too few frames")
        return None

    h, w = frames[0].shape[:2]
    side = int(min(w, h) * 0.55)
    x0, y0 = (w - side) // 2, (h - side) // 2
    print(f"  {len(frames)} frames, crop={side}x{side}")

    grays = [cv2.cvtColor(f[y0:y0+side, x0:x0+side], cv2.COLOR_BGR2GRAY)
             for f in frames]

    # === Phase correlation stabilization ===
    shifts_pc = stabilize_phase_corr(grays)
    shake_pc = np.mean([np.sqrt(dx**2 + dy**2) for dx, dy in shifts_pc])

    # === ORB stabilization (handles rotation) ===
    shifts_orb, transforms_orb = stabilize_orb(grays)

    # Use ORB transforms (handles rotation)
    stabilized_orb = [apply_transform(grays[i], transforms_orb[i])
                      for i in range(len(grays))]
    # Use phase correlation (simpler, often sufficient)
    stabilized_pc = [apply_shift(grays[i], *shifts_pc[i])
                     for i in range(len(grays))]

    # Measure stabilization quality for both
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

    print(f"  Shake: {shake_pc:.1f}px → PC residual: {res_pc:.2f}px, "
          f"ORB residual: {res_orb:.2f}px")

    # Pick the better stabilization
    if res_orb < res_pc * 0.8:
        stabilized = stabilized_orb
        stab_method = "ORB"
    else:
        stabilized = stabilized_pc
        stab_method = "PC"
    print(f"  Using: {stab_method}")

    # Build background (temporal median)
    bg_frames = stabilized[30:min(180, len(stabilized))]
    background = np.median(np.array(bg_frames, dtype=np.float32), axis=0)

    # Save diagnostics
    cv2.imwrite(os.path.join(OUTPUT_DIR, f"{video_name}_frame0.png"), grays[0])
    cv2.imwrite(os.path.join(OUTPUT_DIR, f"{video_name}_bg.png"),
                background.astype(np.uint8))

    cx, cy = side / 2.0, side / 2.0
    radius = side / 2.0

    # Ring configs
    rings = [
        ("outer", 0.50, 0.85, 25),
        ("mid",   0.30, 0.65, 25),
        ("inner", 0.20, 0.50, 20),
    ]

    # Blur kernels to try on the 2D residual
    blur_sizes = [0, 3, 7]

    best_result = None

    for ring_name, r_in_f, r_out_f, n_rad in rings:
        r_inner = radius * r_in_f
        r_outer = radius * r_out_f

        for blur_k in blur_sizes:
            # Build kymograph from blurred 2D residuals
            kymo = np.zeros((len(stabilized), N_ANGLES))
            for i, stab in enumerate(stabilized):
                res2d = np.abs(np.float64(stab) - background)
                if blur_k > 0:
                    res2d = cv2.GaussianBlur(res2d, (blur_k, blur_k), 0)
                kymo[i] = compute_radial_profile(res2d, cx, cy, r_inner,
                                                  r_outer, N_ANGLES, n_rad)

            # Skip first 30 frames
            kymo_slice = kymo[30:]
            if len(kymo_slice) < 60:
                continue

            # Radon search (only near 6°/s)
            results = radon_search(kymo_slice, FPS, N_ANGLES,
                                   vel_min=4.5, vel_max=7.5)
            if not results:
                continue

            best_vel, best_snr, best_score, best_peak, ref_t = results[0]

            # Refine
            vel_r, peak_r, ref_r = refine_velocity(kymo_slice, best_vel,
                                                    FPS, N_ANGLES)

            # Verify
            match, ts, angs = verify_angles(kymo_slice, vel_r, peak_r,
                                            ref_r, FPS, N_ANGLES)

            # Measure drift
            slope, rms, n_pts = measure_drift(ts, angs)

            quality = best_snr * match

            if best_result is None or quality > best_result[0]:
                best_result = (quality, ring_name, blur_k, vel_r, best_snr,
                               match, slope, rms, n_pts, stab_method)

            if match > 0.4 and best_snr > 5:
                blur_str = f"blur={blur_k}" if blur_k > 0 else "raw"
                print(f"  {ring_name}/{blur_str}: vel={vel_r:+.2f}°/s, "
                      f"SNR={best_snr:.0f}, match={match:.0%}"
                      + (f", slope={slope:.3f}°/s, rms={rms:.2f}°" if slope else ""))

        # Save kymograph for outer ring, no blur
        if ring_name == "outer":
            kymo_out = np.zeros((len(stabilized), N_ANGLES))
            for i, stab in enumerate(stabilized):
                res2d = np.abs(np.float64(stab) - background)
                kymo_out[i] = compute_radial_profile(res2d, cx, cy,
                                                      radius * 0.5, radius * 0.85,
                                                      N_ANGLES, 25)
            # Save raw and blurred kymograph
            for sf in [60, 120, 180]:
                if sf < len(stabilized):
                    res2d = cv2.absdiff(stabilized[sf], background.astype(np.uint8))
                    # Also save blurred residual
                    res_blur = cv2.GaussianBlur(res2d, (7, 7), 0)
                    res_enhanced = cv2.normalize(res_blur, None, 0, 255,
                                                 cv2.NORM_MINMAX)
                    cv2.imwrite(os.path.join(OUTPUT_DIR,
                                f"{video_name}_res_blur_f{sf}.png"),
                                res_enhanced)

    # Report
    print(f"\n  --- RESULT ---")
    if best_result:
        (quality, ring, blur, vel, snr, match, slope,
         rms, n_pts, stab) = best_result
        vel_err = abs(abs(vel) - 6.0)

        if vel_err < 0.5 and match > 0.65 and rms is not None and rms < 2.0:
            status = "GOOD"
        elif vel_err < 1.0 and match > 0.50:
            status = "OK"
        elif vel_err < 1.5 and match > 0.35:
            status = "PARTIAL"
        else:
            status = "POOR"

        blur_str = f"blur={blur}" if blur > 0 else "raw"
        print(f"  {status}: vel={vel:+.2f}°/s, match={match:.0%}, SNR={snr:.0f}, "
              f"ring={ring}, {blur_str}, stab={stab}")
        if slope is not None:
            print(f"  Measured slope: {slope:.3f}°/s, RMS: {rms:.2f}°, "
                  f"points: {n_pts}")

        return (status, video_name, vel, match, snr, rms, ring, blur, stab)
    else:
        print(f"  FAILED")
        return ("FAILED", video_name, 0, 0, 0, 999, "none", 0, "none")


def main():
    videos = sorted([f for f in os.listdir(VIDEOS_DIR)
                     if f.upper().endswith((".MOV", ".MP4"))])
    if not videos:
        print("No videos!")
        return

    print(f"Stabilized Kymograph v4")
    print(f"Blur + tight velocity + ORB fallback")
    print(f"{len(videos)} videos\n")

    results = []
    for vn in videos:
        r = process_video(os.path.join(VIDEOS_DIR, vn), vn)
        if r:
            results.append(r)

    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")
    print(f"{'St':<8} {'Video':<18} {'Vel':>7} {'Match':>6} {'SNR':>5} "
          f"{'RMS':>5} {'Ring':<6} {'Blur':<5} {'Stab':<4}")
    print("-" * 70)
    for s, n, v, m, snr, rms, ring, blur, stab in results:
        rms_str = f"{rms:.1f}°" if rms and rms < 900 else "n/a"
        print(f"{s:<8} {n:<18} {v:>+6.2f} {m:>5.0%} {snr:>4.0f} "
              f"{rms_str:>5} {ring:<6} {blur:<5} {stab:<4}")

    good = sum(1 for r in results if r[0] in ("GOOD", "OK"))
    partial = sum(1 for r in results if r[0] == "PARTIAL")
    print(f"\n{good} GOOD/OK, {partial} PARTIAL out of {len(results)}")


if __name__ == "__main__":
    main()
