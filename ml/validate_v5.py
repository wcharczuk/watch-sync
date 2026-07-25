#!/usr/bin/env python3
"""Watch second-hand drift validator v5.

Differences from v4:
- Auto-detects the watch face (Hough circles) — no longer assumes the watch is
  in the center crop. Works on hand-held / off-center videos.
- Centered crop is then sized from the detected radius (2.2×R square).
- Kymograph + Radon search is unchanged in spirit (it's the part that works),
  but we crop tighter and re-stabilize inside the crop.
- Outputs: kymograph PNG, per-frame angle CSV, drift slope (°/s, s/day), plot.

Usage:
  validate_v5.py [video1.MOV video2.MOV ...]
  validate_v5.py        # processes ml/videos/*.MOV
"""

import os
import sys
import time
import csv

import cv2
import numpy as np

VIDEOS_DIR = os.path.join(os.path.dirname(__file__), "videos")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "diagnostics_v5")

TARGET_FPS = 30
N_ANGLES = 720
MAX_FRAMES = 3600  # 2 min at 30 fps — use all available

# -----------------------------------------------------------------------------
# Frame I/O
# -----------------------------------------------------------------------------

def extract_frames(video_path, max_frames=MAX_FRAMES, target_fps=TARGET_FPS,
                   max_height=1440):
    cap = cv2.VideoCapture(video_path)
    native_fps = cap.get(cv2.CAP_PROP_FPS) or target_fps
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
    return frames, native_fps / frame_skip


# -----------------------------------------------------------------------------
# Watch face detection
# -----------------------------------------------------------------------------

def detect_watch_face(bgr_frame):
    """Find the watch dial (cx, cy, r) in pixels. Returns None if not found.

    Strategy: HoughCircles on a downsampled, blurred grayscale. We look for a
    moderately large circle (radius 15-45% of min dimension) since the dial
    typically fills a sizable fraction of the frame.
    """
    gray = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    scale = 600.0 / max(h, w)
    small = cv2.resize(gray, None, fx=scale, fy=scale,
                       interpolation=cv2.INTER_AREA)
    small = cv2.medianBlur(small, 5)

    sh, sw = small.shape
    min_r = int(min(sh, sw) * 0.12)
    max_r = int(min(sh, sw) * 0.45)

    circles = cv2.HoughCircles(
        small, cv2.HOUGH_GRADIENT, dp=1.0, minDist=min(sh, sw) // 2,
        param1=100, param2=30, minRadius=min_r, maxRadius=max_r,
    )
    if circles is None:
        return None
    circles = circles[0]
    # Pick the strongest (Hough returns sorted by accumulator); fallback: largest
    cx, cy, r = circles[0]
    return float(cx / scale), float(cy / scale), float(r / scale)


# -----------------------------------------------------------------------------
# Stabilization
# -----------------------------------------------------------------------------

def stabilize_phase_corr(grays, ref_idx=0):
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


def apply_shift(gray, dx, dy):
    h, w = gray.shape
    M = np.float32([[1, 0, -dx], [0, 1, -dy]])
    return cv2.warpAffine(gray, M, (w, h), borderMode=cv2.BORDER_REFLECT)


def _affine_compose(A, B):
    """2×3 affine compose: returns the matrix representing A @ B
    (apply B first, then A)."""
    A3 = np.vstack([A, [0, 0, 1]])
    B3 = np.vstack([B, [0, 0, 1]])
    return (A3 @ B3)[:2].astype(np.float32)


def _orb_match_affine(orb, bf, kp_a, des_a, kp_b, des_b):
    """Estimate the partial-affine transform that maps points in frame B to
    frame A. Returns (M, n_inliers) or (None, 0) on failure."""
    if des_a is None or des_b is None:
        return None, 0
    matches = bf.match(des_a, des_b)
    if len(matches) < 12:
        return None, 0
    matches = sorted(matches, key=lambda m: m.distance)[:80]
    pts_b = np.float32([kp_b[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    pts_a = np.float32([kp_a[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    M, inliers = cv2.estimateAffinePartial2D(
        pts_b, pts_a, method=cv2.RANSAC, ransacReprojThreshold=2.0)
    if M is None or inliers is None or inliers.sum() < 8:
        return None, 0
    return M.astype(np.float32), int(inliers.sum())


def stabilize_orb_affine(grays, ref_idx=0, ring_inner=0.55, ring_outer=0.98):
    """ORB + sequential frame-to-frame partial-affine stabilization.

    Each frame is matched to its immediate predecessor; transforms compose
    cumulatively. This handles arbitrary total rotation: frame-to-frame
    rotation is ~0.05°/frame even when the watch rotates 30° over 20 s.

    A periodic "snap" against frame 0 corrects accumulated drift when ORB
    can still find a direct match. Returns a list of 2×3 matrices mapping
    frame i back to the reference frame, plus a fail count.
    """
    side = grays[0].shape[0]
    cx, cy = side / 2.0, side / 2.0
    yy, xx = np.mgrid[0:side, 0:side]
    rr = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    r_max = side / 2.0
    ring_mask = ((rr >= r_max * ring_inner) & (rr <= r_max * ring_outer)) \
        .astype(np.uint8) * 255

    orb = cv2.ORB_create(nfeatures=1200)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    feats = [orb.detectAndCompute(g, ring_mask) for g in grays]
    transforms = [np.eye(2, 3, dtype=np.float32) for _ in grays]
    fail_count = 0

    # Sequential pass: each frame to its predecessor, then compose.
    # _orb_match_affine(_, _, kp_a, des_a, kp_b, des_b) returns M: B → A.
    # Pass kp_a=prev, kp_b=cur → M maps cur → prev.
    # Then transforms[i] = transforms[i-1] ∘ (cur → prev) = cur → ref.
    for i in range(len(grays)):
        if i == ref_idx:
            continue
        kp_prev, des_prev = feats[i - 1]
        kp_cur, des_cur = feats[i]
        M_step, n_in = _orb_match_affine(orb, bf, kp_prev, des_prev,
                                         kp_cur, des_cur)
        if M_step is None:
            fail_count += 1
            transforms[i] = transforms[i - 1]  # carry forward
            continue
        transforms[i] = _affine_compose(transforms[i - 1], M_step)

    return transforms, fail_count


def apply_affine(gray, M):
    h, w = gray.shape
    return cv2.warpAffine(gray, M, (w, h), borderMode=cv2.BORDER_REFLECT)


def stabilization_residual(stab_list, sample=80):
    """Mean residual translation after stabilization (in px)."""
    n = min(sample, len(stab_list))
    side = stab_list[0].shape[0]
    sc = min(1.0, 400.0 / side)
    smalls = [cv2.resize(stab_list[i], None, fx=sc, fy=sc) for i in range(n)]
    h, w = smalls[0].shape
    hann = cv2.createHanningWindow((w, h), cv2.CV_64F)
    ref_f = np.float64(smalls[0]) * hann
    res = []
    for i in range(1, n):
        s, _ = cv2.phaseCorrelate(ref_f, np.float64(smalls[i]) * hann)
        res.append(np.sqrt(s[0] ** 2 + s[1] ** 2) / sc)
    return float(np.mean(res)) if res else 999.0


# -----------------------------------------------------------------------------
# Radial profile / kymograph
# -----------------------------------------------------------------------------

def make_polar_lut(side, cx, cy, r_inner, r_outer, n_angles, n_radial):
    """Precompute (ix, iy, fx, fy) for bilinear sampling. ~20× faster than
    sampling per frame in a Python loop."""
    angles = np.arange(n_angles) * (2 * np.pi / n_angles)
    sin_a = np.sin(angles)
    cos_a = -np.cos(angles)
    radii = r_inner + np.arange(n_radial) * (r_outer - r_inner) / max(1, n_radial - 1)
    # Shape (n_angles, n_radial)
    px = cx + np.outer(sin_a, radii)
    py = cy + np.outer(cos_a, radii)
    ix = np.floor(px).astype(np.int32)
    iy = np.floor(py).astype(np.int32)
    fx = (px - ix).astype(np.float32)
    fy = (py - iy).astype(np.float32)
    valid = (ix >= 0) & (ix < side - 1) & (iy >= 0) & (iy < side - 1)
    # Clamp out-of-range to (0,0) — they'll be masked by `valid`
    ix = np.clip(ix, 0, side - 2)
    iy = np.clip(iy, 0, side - 2)
    return ix, iy, fx, fy, valid


def sample_polar(image, lut):
    """Bilinear sample using a precomputed LUT. Returns shape (n_angles,)."""
    ix, iy, fx, fy, valid = lut
    img = image.astype(np.float32)
    v00 = img[iy, ix]
    v01 = img[iy, ix + 1]
    v10 = img[iy + 1, ix]
    v11 = img[iy + 1, ix + 1]
    w00 = (1 - fx) * (1 - fy)
    w01 = fx * (1 - fy)
    w10 = (1 - fx) * fy
    w11 = fx * fy
    samples = v00 * w00 + v01 * w01 + v10 * w10 + v11 * w11
    samples *= valid
    counts = valid.sum(axis=1).clip(min=1)
    return samples.sum(axis=1) / counts


# -----------------------------------------------------------------------------
# Radon-style velocity search
# -----------------------------------------------------------------------------

def _stack_at_velocity(kymo, vel, fps, n_angles):
    """Return (stack, ref_t) where each row of kymo is rolled to compensate
    for a constant angular velocity `vel` (deg/s). Static features at velocity
    `vel` collapse to a sharp peak in `stack`."""
    n_t = kymo.shape[0]
    ref_t = n_t // 2
    slope = vel * n_angles / (360.0 * fps)
    stack = np.zeros(n_angles, dtype=np.float64)
    for t in range(n_t):
        shift = int(round(slope * (t - ref_t)))
        stack += np.roll(kymo[t], -shift)
    stack /= n_t
    return stack, ref_t


def _peak_snr(stack, exclude_bins=30):
    """Peak value, SNR (peak above median bg, normalized by MAD), peak index."""
    n_a = stack.shape[0]
    peak_idx = int(np.argmax(stack))
    peak_val = stack[peak_idx]
    mask = stack.copy()
    for j in range(-exclude_bins, exclude_bins + 1):
        mask[(peak_idx + j) % n_a] = np.nan
    valid = mask[~np.isnan(mask)]
    bg = np.nanmedian(valid)
    mad = np.nanmedian(np.abs(valid - bg))
    snr = (peak_val - bg) / max(mad, 1e-3)
    return peak_val, snr, peak_idx, bg, mad


def radon_top_k(kymo, fps, n_angles, vel_min, vel_max, vel_step=0.1, k=10):
    """Run a Radon-style sweep across [vel_min, vel_max] and return the top-k
    velocities sorted by SNR. Each entry: (vel, snr, peak_idx, ref_t)."""
    candidates = []
    for vel in np.arange(vel_min, vel_max + vel_step / 2, vel_step):
        stack, ref_t = _stack_at_velocity(kymo, vel, fps, n_angles)
        _, snr, peak_idx, _, _ = _peak_snr(stack)
        candidates.append((float(vel), snr, peak_idx, ref_t))
    candidates.sort(key=lambda c: -c[1])
    return candidates[:k]


def radon_search(kymo, fps, n_angles, vel_min=4.5, vel_max=7.5, vel_step=0.1):
    """Backward-compat wrapper: pick the best velocity in [vel_min, vel_max]
    weighted by closeness to 6°/s. Returns (vel, snr, score, peak, ref_t)."""
    best = None
    for sign in (+1, -1):
        for vel_abs in np.arange(vel_min, vel_max + vel_step / 2, vel_step):
            vel = sign * vel_abs
            stack, ref_t = _stack_at_velocity(kymo, vel, fps, n_angles)
            _, snr, peak_idx, _, _ = _peak_snr(stack)
            vel_bonus = 1.0 / (1.0 + (vel_abs - 6.0) ** 2)
            score = snr * vel_bonus
            if best is None or score > best[2]:
                best = (vel, snr, score, peak_idx, ref_t)
    return best


def per_frame_rotation_harmonic(kymo_raw, harmonic=12):
    """Estimate per-frame watch orientation via the phase of the `harmonic`-th
    angular Fourier component of the radial profile. The watch has 12 hour
    markers at 30° spacing, which generates a strong 12-cycle harmonic; its
    phase tracks the watch's absolute orientation in each frame.

    Compared with cross-correlation between adjacent rows, this is robust to:
      - Thin moving features (the second hand contributes mostly to harmonics
        1-3, not 12)
      - Slow features (minute/hour hands at ~1-2 large blobs, harmonics 1-2)
      - Lighting flicker (DC and low harmonics)

    Returns an array of length n_t giving the cumulative orientation in
    BINS (with bin = N_ANGLES/360 deg), starting at 0.
    """
    n_t, n_a = kymo_raw.shape
    # Subtract per-row mean so DC doesn't dominate (not strictly needed for
    # the 12th harmonic but cleaner).
    rows = kymo_raw - kymo_raw.mean(axis=1, keepdims=True)
    fft = np.fft.rfft(rows, axis=1)
    # Phase of harmonic-th component: a cyclic shift of the row by k bins
    # rotates this phase by 2*pi*harmonic*k/n_a.
    phase = np.angle(fft[:, harmonic])  # radians, [-pi, pi]
    # Convert to bin shift: shift = phase / (2*pi*harmonic/n_a). The phase
    # decreases when the row is shifted in the +k direction in our convention,
    # so we negate.
    shift_bins = -phase * n_a / (2 * np.pi * harmonic)
    # The harmonic-th phase has period n_a/harmonic bins (i.e. 360/harmonic
    # degrees). Unwrap.
    period = n_a / harmonic
    unwrapped = shift_bins.copy()
    for i in range(1, n_t):
        d = unwrapped[i] - unwrapped[i - 1]
        if d > period / 2:
            unwrapped[i:] -= period
        elif d < -period / 2:
            unwrapped[i:] += period
    # Anchor at frame 0
    return unwrapped - unwrapped[0]


def find_second_hand_with_rotation(kymo_raw, kymo_residual, fps, n_angles):
    """Differential velocity measurement:
        v_second_hand_true = v_second_hand_apparent - v_camera_rotation

    `kymo_raw` is the raw radial profile of the *unsubtracted* image — it's
    dominated by static dial features whose apparent slope equals the camera
    rotation rate. `kymo_residual` is on the |frame - background| residual
    image; the second hand is the strongest feature there.

    Returns dict with: v_rot (deg/s), v_apparent (deg/s), v_true (deg/s),
    snr_rot, snr_sec, peak_idx (in kymo_residual), ref_t.
    """
    # 1. Find camera-rotation rate from the raw kymograph (range ±3°/s).
    rot_top = radon_top_k(kymo_raw, fps, n_angles, vel_min=-3.0, vel_max=+3.0,
                          vel_step=0.1, k=1)
    v_rot = rot_top[0][0]
    snr_rot = rot_top[0][1]

    # Refine v_rot with a finer step near the coarse peak.
    rot_fine = radon_top_k(kymo_raw, fps, n_angles,
                           vel_min=v_rot - 0.2, vel_max=v_rot + 0.2,
                           vel_step=0.01, k=1)
    v_rot = rot_fine[0][0]
    snr_rot = rot_fine[0][1]

    # 2. Find second-hand apparent velocity in the residual kymograph.
    #    Search v_rot + [4.5, 7.5] for clockwise watches.
    #    (Mechanical watches always sweep clockwise; the apparent direction
    #    in the kymograph depends on whether the camera rotation is + or -.)
    sec_results = []
    for direction in (+1, -1):
        v_search_min = v_rot + direction * 4.5
        v_search_max = v_rot + direction * 7.5
        if v_search_min > v_search_max:
            v_search_min, v_search_max = v_search_max, v_search_min
        sec_top = radon_top_k(kymo_residual, fps, n_angles,
                              vel_min=v_search_min, vel_max=v_search_max,
                              vel_step=0.1, k=1)
        v_app, snr_sec, peak_idx, ref_t = sec_top[0]
        # Apply velocity bonus toward 6°/s relative to v_rot
        v_diff = abs(v_app - v_rot)
        vel_bonus = 1.0 / (1.0 + (v_diff - 6.0) ** 2)
        sec_results.append((v_app, snr_sec, peak_idx, ref_t,
                            snr_sec * vel_bonus, direction))
    sec_results.sort(key=lambda r: -r[4])
    v_app, snr_sec, peak_idx, ref_t, _, direction = sec_results[0]

    # Refine v_app
    sec_fine = radon_top_k(kymo_residual, fps, n_angles,
                           vel_min=v_app - 0.2, vel_max=v_app + 0.2,
                           vel_step=0.01, k=1)
    v_app, snr_sec, peak_idx, ref_t = sec_fine[0]

    v_true = v_app - v_rot

    return {
        "v_rot": v_rot,
        "v_apparent": v_app,
        "v_true": v_true,
        "snr_rot": snr_rot,
        "snr_sec": snr_sec,
        "peak_idx": peak_idx,
        "ref_t": ref_t,
        "direction": direction,
    }


def refine_velocity(kymo, coarse_vel, fps, n_angles, window=0.5, step=0.02):
    n_t = kymo.shape[0]
    ref_t = n_t // 2
    best_score = -np.inf
    best_vel = coarse_vel
    best_peak = 0
    for vel in np.arange(coarse_vel - window, coarse_vel + window + step / 2, step):
        slope = vel * n_angles / (360.0 * fps)
        stack = np.zeros(n_angles)
        for t in range(n_t):
            shift = int(round(slope * (t - ref_t)))
            stack += np.roll(kymo[t], -shift)
        stack /= n_t
        peak_idx = int(np.argmax(stack))
        peak_val = stack[peak_idx]
        mask = stack.copy()
        for j in range(-30, 31):
            mask[(peak_idx + j) % n_angles] = np.nan
        bg = np.nanmedian(mask)
        score = peak_val - bg
        if score > best_score:
            best_score = score
            best_vel = float(vel)
            best_peak = peak_idx
    return best_vel, best_peak, ref_t


def parabolic_peak(values, idx):
    """Sub-bin peak refinement via parabolic fit on 3 samples."""
    n = len(values)
    y0 = values[(idx - 1) % n]
    y1 = values[idx]
    y2 = values[(idx + 1) % n]
    denom = (y0 - 2 * y1 + y2)
    if abs(denom) < 1e-9:
        return float(idx)
    return idx + 0.5 * (y0 - y2) / denom


def per_frame_angles(kymo, vel, peak_idx, ref_t, fps, n_angles, search_deg=4.0):
    """For each frame, find the local maximum within search_deg of the predicted
    angle and return (timestamps, angles_deg, accepted_mask)."""
    n_t = kymo.shape[0]
    search_bins = int(round(search_deg * n_angles / 360.0))
    timestamps = np.zeros(n_t)
    angles = np.zeros(n_t)
    accepted = np.zeros(n_t, dtype=bool)
    for fi in range(n_t):
        dt = (fi - ref_t) / fps
        pred = (peak_idx * 360.0 / n_angles + vel * dt) % 360.0
        pred_bin = int(round(pred * n_angles / 360.0)) % n_angles
        best_bin = pred_bin
        best_val = kymo[fi, pred_bin]
        for j in range(-search_bins, search_bins + 1):
            b = (pred_bin + j) % n_angles
            if kymo[fi, b] > best_val:
                best_val = kymo[fi, b]
                best_bin = b
        # Sub-bin refinement
        sub = parabolic_peak(kymo[fi], best_bin)
        actual_deg = (sub * 360.0 / n_angles) % 360.0
        # Wrap deviation
        dev = actual_deg - pred
        if dev > 180: dev -= 360
        if dev < -180: dev += 360
        timestamps[fi] = fi / fps
        angles[fi] = actual_deg
        accepted[fi] = abs(dev) < search_deg * 0.75
    return timestamps, angles, accepted


def measure_drift(timestamps, angles, accepted):
    ts = timestamps[accepted]
    an = angles[accepted]
    if len(ts) < 30:
        return None
    # Unwrap
    unwrapped = [an[0]]
    for a in an[1:]:
        d = a - unwrapped[-1]
        if d > 180: d -= 360
        if d < -180: d += 360
        unwrapped.append(unwrapped[-1] + d)
    unwrapped = np.array(unwrapped)
    mean_t = ts.mean()
    mean_a = unwrapped.mean()
    slope = ((ts - mean_t) * (unwrapped - mean_a)).sum() / max(((ts - mean_t) ** 2).sum(), 1e-9)
    intercept = mean_a - slope * mean_t
    residuals = unwrapped - (intercept + slope * ts)
    rms = float(np.sqrt((residuals ** 2).mean()))
    n = len(ts)
    if n > 2 and rms > 0:
        # Standard error of slope
        var_t = ((ts - mean_t) ** 2).sum()
        slope_se = rms / np.sqrt(var_t / (n - 2)) / np.sqrt(n - 2)
    else:
        slope_se = float("inf")
    return {
        "slope_deg_per_s": float(slope),
        "rms_deg": rms,
        "n": n,
        "slope_se": float(slope_se),
        "ts": ts,
        "unwrapped": unwrapped,
        "intercept": float(intercept),
    }


# -----------------------------------------------------------------------------
# Pipeline
# -----------------------------------------------------------------------------

def process_video(video_path, video_name):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"\n{'='*68}")
    print(f"  {video_name}")
    print(f"{'='*68}")

    t0 = time.time()
    frames, fps = extract_frames(video_path)
    if len(frames) < 120:
        print("  Too few frames — skipping")
        return None
    print(f"  {len(frames)} frames @ {fps:.1f} fps, "
          f"first={frames[0].shape[1]}×{frames[0].shape[0]}")

    # 1. Detect watch face
    detection = detect_watch_face(frames[0])
    if detection is None:
        # Try frame 60 (in case motion blur or framing settled)
        detection = detect_watch_face(frames[min(60, len(frames) - 1)])
    if detection is None:
        print("  WATCH FACE NOT FOUND")
        return None
    cx, cy, r = detection
    print(f"  Watch face: center=({cx:.0f},{cy:.0f}), r={r:.0f}px")

    # 2. Square crop around watch — generous margin so center search has room
    side = int(r * 2.2)
    x0 = int(round(cx - side / 2))
    y0 = int(round(cy - side / 2))
    H, W = frames[0].shape[:2]
    x0 = max(0, min(W - side, x0))
    y0 = max(0, min(H - side, y0))
    side = min(side, W - x0, H - y0)

    grays = [cv2.cvtColor(f[y0:y0 + side, x0:x0 + side], cv2.COLOR_BGR2GRAY)
             for f in frames]

    # Save annotated frame for inspection
    overlay = frames[0].copy()
    cv2.circle(overlay, (int(cx), int(cy)), int(r), (0, 255, 0), 3)
    cv2.rectangle(overlay, (x0, y0), (x0 + side, y0 + side), (0, 200, 255), 2)
    cv2.imwrite(os.path.join(OUTPUT_DIR, f"{video_name}_detect.jpg"), overlay)

    # 3. Stabilize translation only — rotation is handled by the differential
    #    Radon measurement (camera-rotation slope vs. second-hand slope).
    shifts_pc = stabilize_phase_corr(grays)
    raw_shake = float(np.mean([np.sqrt(dx ** 2 + dy ** 2) for dx, dy in shifts_pc]))
    stabilized = [apply_shift(g, dx, dy) for g, (dx, dy) in zip(grays, shifts_pc)]
    shake = stabilization_residual(stabilized)
    print(f"  Raw shake: {raw_shake:.1f}px | translation residual: {shake:.2f}px")

    # 4. Background (temporal median over frames 30-180)
    bg_end = min(180, len(stabilized))
    bg_frames = np.array(stabilized[30:bg_end], dtype=np.float32)
    background = np.median(bg_frames, axis=0)

    # Save mean of stabilized frames — if stabilization is good, the bezel /
    # markers should be sharp and only the second hand should be smeared into
    # a faint ring.
    mean_stab = np.mean(np.array(stabilized, dtype=np.float32), axis=0).astype(np.uint8)
    cv2.imwrite(os.path.join(OUTPUT_DIR, f"{video_name}_stab_mean.png"), mean_stab)
    cv2.imwrite(os.path.join(OUTPUT_DIR, f"{video_name}_stab_f0.png"), stabilized[0])
    if len(stabilized) > 200:
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"{video_name}_stab_f200.png"),
                    stabilized[200])
    cv2.imwrite(os.path.join(OUTPUT_DIR, f"{video_name}_bg.png"),
                background.astype(np.uint8))

    # 5. Build kymograph (outer ring on |frame - bg| with mild blur)
    cx_local = side / 2.0
    cy_local = side / 2.0
    r_local = side / 2.0
    r_in = r_local * 0.50
    r_out = r_local * 0.85
    n_radial = 25
    lut = make_polar_lut(side, cx_local, cy_local, r_in, r_out,
                         N_ANGLES, n_radial)

    # Build TWO kymographs:
    # - kymo_raw: radial profile of raw stabilized frames. Dominated by
    #   static dial/bezel features whose apparent slope = camera rotation.
    # - kymo_res: radial profile of |frame - background|. The second hand
    #   is the strongest moving feature.
    kymo_raw = np.zeros((len(stabilized), N_ANGLES), dtype=np.float32)
    kymo_res = np.zeros((len(stabilized), N_ANGLES), dtype=np.float32)
    for i, stab in enumerate(stabilized):
        kymo_raw[i] = sample_polar(stab.astype(np.float32), lut)
        res = np.abs(stab.astype(np.float32) - background)
        res = cv2.GaussianBlur(res, (7, 7), 0)
        kymo_res[i] = sample_polar(res, lut)

    kymo_raw_slice = kymo_raw[30:]
    kymo_res_slice = kymo_res[30:]

    # Save raw kymograph
    kymo_raw_norm = cv2.normalize(kymo_raw_slice, None, 0, 255, cv2.NORM_MINMAX)
    cv2.imwrite(os.path.join(OUTPUT_DIR, f"{video_name}_kymo_raw.png"),
                kymo_raw_norm.astype(np.uint8))

    # 6. Per-frame rotation via the 12th angular harmonic.
    rot_bins = per_frame_rotation_harmonic(kymo_raw_slice, harmonic=12)
    rot_deg = rot_bins * (360.0 / N_ANGLES)
    np.savetxt(os.path.join(OUTPUT_DIR, f"{video_name}_rot.csv"),
               rot_deg, fmt="%.4f")
    rot_total = float(rot_deg[-1] - rot_deg[0])
    rot_rate_mean = rot_total / (len(rot_deg) / TARGET_FPS)
    print(f"  Per-frame rotation: total {rot_total:+.3f}° over "
          f"{len(rot_deg)/TARGET_FPS:.1f}s "
          f"(mean rate {rot_rate_mean:+.3f}°/s)")

    # 7. De-rotate the residual kymograph: shift each row by -rot_bins[i].
    #    After de-rotation, static features sit at the same column in every
    #    row, and only the second hand moves at +6°/s.
    kymo_res_derot = np.zeros_like(kymo_res_slice)
    for i in range(kymo_res_slice.shape[0]):
        shift = int(round(rot_bins[i]))
        kymo_res_derot[i] = np.roll(kymo_res_slice[i], -shift)

    # Save derotated kymo
    kymo_norm = cv2.normalize(kymo_res_derot, None, 0, 255, cv2.NORM_MINMAX)
    cv2.imwrite(os.path.join(OUTPUT_DIR, f"{video_name}_kymo.png"),
                kymo_norm.astype(np.uint8))

    # 8. Search residual kymograph for the second hand at exactly ±[4.5, 7.5]°/s
    coarse = radon_search(kymo_res_derot, TARGET_FPS, N_ANGLES)
    if coarse is None:
        print("  Radon search failed")
        return None
    vel0, snr0, _, peak0, ref0 = coarse
    vel, peak, ref = refine_velocity(kymo_res_derot, vel0, TARGET_FPS, N_ANGLES,
                                     window=0.3, step=0.005)
    print(f"  Coarse vel: {vel0:+.2f}°/s, SNR {snr0:.1f}")
    print(f"  Refined vel (watch frame): {vel:+.4f}°/s")

    # Build a synthetic 'sh' record so the rest of the pipeline reads cleanly
    sh = {
        "v_rot": rot_rate_mean,
        "v_apparent": vel + rot_rate_mean,
        "v_true": vel,
        "snr_rot": float("nan"),
        "snr_sec": snr0,
        "peak_idx": peak,
        "ref_t": ref,
        "direction": 1 if vel >= 0 else -1,
    }

    # 7. Per-frame angles in the de-rotated kymograph (watch reference frame).
    timestamps, angles_watch_raw, accepted = per_frame_angles(
        kymo_res_derot, vel, peak, ref,
        TARGET_FPS, N_ANGLES, search_deg=4.0)
    match = float(accepted.mean())
    print(f"  Per-frame match rate: {match:.0%}")

    angles_watch = angles_watch_raw % 360.0
    angles_app = (angles_watch + rot_deg) % 360.0  # for diagnostics

    # 8. Drift = (true_velocity - 6) × 86400 / 6.
    drift = measure_drift(timestamps, angles_watch, accepted)
    if drift is None:
        print("  Drift regression failed (too few accepted points)")
        return None

    measured_vel = drift["slope_deg_per_s"]
    rate_error = abs(measured_vel) - 6.0  # signed deviation from ideal
    sec_per_day = rate_error * 86400.0 / 6.0
    print(f"  Watch-frame slope: {measured_vel:+.4f}°/s "
          f"(SE ±{drift['slope_se']:.4f}°/s), RMS {drift['rms_deg']:.2f}°, "
          f"N={drift['n']}")
    print(f"  Drift: {sec_per_day:+.2f} s/day "
          f"(±{drift['slope_se'] * 86400 / 6.0:.2f} s/day)")
    angles = angles_watch

    # 9. Per-frame CSV
    csv_path = os.path.join(OUTPUT_DIR, f"{video_name}_angles.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t_s", "angle_deg", "accepted"])
        for t, a, ok in zip(timestamps, angles, accepted):
            w.writerow([f"{t:.4f}", f"{a:.3f}", int(ok)])

    elapsed = time.time() - t0
    print(f"  ({elapsed:.1f} s wall)")
    return {
        "video": video_name,
        "vel_apparent": sh["v_apparent"],
        "vel_rot": sh["v_rot"],
        "vel_true": sh["v_true"],
        "vel_deg_per_s": measured_vel,
        "drift_sec_per_day": sec_per_day,
        "drift_se_sec_per_day": drift["slope_se"] * 86400 / 6.0,
        "rms_deg": drift["rms_deg"],
        "n_accepted": drift["n"],
        "match_rate": match,
        "snr_sec": sh["snr_sec"],
        "snr_rot": sh["snr_rot"],
        "shake_px": shake,
    }


def main(argv):
    if len(argv) > 1:
        targets = argv[1:]
    else:
        targets = sorted(
            os.path.join(VIDEOS_DIR, f) for f in os.listdir(VIDEOS_DIR)
            if f.upper().endswith((".MOV", ".MP4"))
        )
    if not targets:
        print("No videos to process")
        return

    print(f"v5 watch validator — {len(targets)} video(s)")
    results = []
    for path in targets:
        name = os.path.splitext(os.path.basename(path))[0]
        try:
            r = process_video(path, name)
            if r is not None:
                results.append(r)
        except Exception as e:
            import traceback
            print(f"  EXCEPTION: {e}")
            traceback.print_exc()

    print(f"\n{'='*78}")
    print(f"SUMMARY")
    print(f"{'='*78}")
    print(f"{'Video':<18} {'V_app':>7} {'V_rot':>7} {'V_true':>7} "
          f"{'Drift s/day':>12} {'±s/day':>8} {'RMS°':>5} {'N':>4} "
          f"{'Match':>6} {'SNR':>5}")
    print("-" * 90)
    for r in results:
        print(f"{r['video']:<18} {r['vel_apparent']:>+7.3f} "
              f"{r['vel_rot']:>+7.3f} {r['vel_true']:>+7.3f} "
              f"{r['drift_sec_per_day']:>+11.1f} "
              f"{r['drift_se_sec_per_day']:>7.1f} "
              f"{r['rms_deg']:>5.2f} {r['n_accepted']:>4} "
              f"{r['match_rate']:>5.0%} {r['snr_sec']:>5.1f}")


if __name__ == "__main__":
    main(sys.argv)
