#!/usr/bin/env python3
"""Watch second-hand drift validator v6 (handheld).

Key change from v5: the static-feature suppression happens in *kymograph space*
via a per-angle temporal high-pass, applied AFTER de-rotation. Once the watch is
de-rotated (camera rotation removed via the 12th angular harmonic), every static
feature sits in a fixed column, so subtracting a per-angle moving-average along
time annihilates it. The second hand sweeps ~9° through any angle bin in 1.5s,
so it survives as a clean diagonal ridge. This raised the second-hand velocity
peak SNR ~3-5x over v5 (see exp_highpass.py).

Pipeline:
  frames -> detect face -> square crop -> translation-stabilize (phase corr)
         -> raw kymograph (outer ring) -> 12th-harmonic per-frame rotation
         -> de-rotate kymograph -> per-angle temporal high-pass (abs)
         -> Radon velocity lock (+/-[4.5,7.5] deg/s) -> per-frame angle
         -> unwrap + linear fit -> drift (s/day) with standard error.

Drift sign: watch second hand sweeps clockwise = +6.000 deg/s. A measured slope
> 6 means the watch runs fast (gains s/day); < 6 means slow.

Usage:
  validate_v6.py [video1.MOV ...]      # default: ml/videos/*.MOV
"""

import os
import sys
import time
import csv

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from validate_v5 import (
    extract_frames, detect_watch_face, stabilize_phase_corr, apply_shift,
    make_polar_lut, sample_polar, per_frame_rotation_harmonic,
    radon_search, refine_velocity, per_frame_angles, measure_drift,
    N_ANGLES, TARGET_FPS,
)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "diagnostics_v6")
VIDEOS_DIR = os.path.join(os.path.dirname(__file__), "videos")

HP_WIN = 45      # temporal high-pass half-window in frames (~1.5s @ 30fps)
WARMUP = 30      # frames to drop at the start


def temporal_highpass(kymo, win=HP_WIN):
    """Subtract a per-angle moving average along the TIME axis. Removes any
    static/slow column; keeps the fast-moving second-hand ridge."""
    n_t, n_a = kymo.shape
    kpad = np.pad(kymo, ((win, win), (0, 0)), mode="edge")
    ker = np.ones(2 * win + 1, dtype=np.float32) / (2 * win + 1)
    lp = np.empty_like(kymo)
    for a in range(n_a):
        lp[:, a] = np.convolve(kpad[:, a], ker, mode="valid")
    return kymo - lp


def draw_fit_overlay(kymo_hp, ts, angles, accepted, slope, intercept, ref_t,
                     path):
    """Render the de-rotated high-pass kymograph with the fitted second-hand
    line drawn on top (green = accepted per-frame picks, red = the linear fit)."""
    img = cv2.normalize(kymo_hp, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    n_t, n_a = kymo_hp.shape
    # per-frame picks
    for i in range(n_t):
        col = int(round(angles[i] * n_a / 360.0)) % n_a
        c = (0, 255, 0) if accepted[i] else (0, 90, 90)
        cv2.circle(img, (col, i), 0, c, -1)
    # fitted line (red), wrapped
    for i in range(n_t):
        t = i / TARGET_FPS
        a = (intercept + slope * t) % 360.0
        col = int(round(a * n_a / 360.0)) % n_a
        img[i, col] = (0, 0, 255)
    cv2.imwrite(path, img)


def process_video(video_path, video_name):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"\n{'='*68}\n  {video_name}\n{'='*68}")
    t0 = time.time()

    frames, fps = extract_frames(video_path)
    if len(frames) < 120:
        print("  too few frames"); return None
    print(f"  {len(frames)} frames @ {fps:.1f} fps, "
          f"{frames[0].shape[1]}x{frames[0].shape[0]}")

    det = detect_watch_face(frames[0]) or \
        detect_watch_face(frames[min(60, len(frames) - 1)])
    if det is None:
        print("  WATCH FACE NOT FOUND"); return None
    cx, cy, r = det
    print(f"  face: ({cx:.0f},{cy:.0f}) r={r:.0f}px")

    side = int(r * 2.2)
    H, W = frames[0].shape[:2]
    x0 = max(0, min(W - side, int(round(cx - side / 2))))
    y0 = max(0, min(H - side, int(round(cy - side / 2))))
    side = min(side, W - x0, H - y0)
    grays = [cv2.cvtColor(f[y0:y0+side, x0:x0+side], cv2.COLOR_BGR2GRAY)
             for f in frames]

    # 1. Translation stabilization only (rotation handled below).
    shifts = stabilize_phase_corr(grays)
    raw_shake = float(np.mean([np.hypot(dx, dy) for dx, dy in shifts]))
    stab = [apply_shift(g, dx, dy) for g, (dx, dy) in zip(grays, shifts)]
    print(f"  raw shake {raw_shake:.1f}px")

    # 2. Raw kymograph on the outer ring.
    cl = side / 2.0
    lut = make_polar_lut(side, cl, cl, cl * 0.50, cl * 0.85, N_ANGLES, 25)
    kymo_raw = np.stack([sample_polar(s.astype(np.float32), lut)
                         for s in stab])[WARMUP:]

    # 3. Per-frame watch orientation from the 12th angular harmonic.
    rot_bins = per_frame_rotation_harmonic(kymo_raw, harmonic=12)
    rot_deg = rot_bins * (360.0 / N_ANGLES)
    rot_total = float(rot_deg[-1] - rot_deg[0])
    dur = len(rot_deg) / TARGET_FPS
    print(f"  rotation: {rot_total:+.2f} deg over {dur:.1f}s "
          f"(mean {rot_total/dur:+.3f} deg/s)")

    # 4. De-rotate, THEN per-angle temporal high-pass.
    kymo_derot = np.stack([np.roll(kymo_raw[i], -int(round(rot_bins[i])))
                           for i in range(kymo_raw.shape[0])])
    kymo_hp = np.abs(temporal_highpass(kymo_derot, HP_WIN))

    cv2.imwrite(os.path.join(OUTPUT_DIR, f"{video_name}_kymo_hp.png"),
                cv2.normalize(kymo_hp, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8))

    # 5. Velocity lock in the watch frame. After de-rotation the true second
    #    hand sits near +6 deg/s.
    kz = kymo_hp - kymo_hp.mean(axis=1, keepdims=True)
    coarse = radon_search(kz, TARGET_FPS, N_ANGLES, vel_min=4.5, vel_max=7.5)
    if coarse is None:
        print("  radon failed"); return None
    vel0, snr0, _, peak0, ref0 = coarse
    vel, peak, ref = refine_velocity(kz, vel0, TARGET_FPS, N_ANGLES,
                                     window=0.3, step=0.005)
    print(f"  velocity lock: {vel:+.4f} deg/s (coarse {vel0:+.2f}, SNR {snr0:.1f})")

    # 6. Per-frame angle along the locked ridge, unwrap, linear fit.
    ts, angles, accepted = per_frame_angles(kz, vel, peak, ref,
                                            TARGET_FPS, N_ANGLES, search_deg=3.0)
    match = float(accepted.mean())
    drift = measure_drift(ts, angles, accepted)
    if drift is None:
        print(f"  drift fit failed (match {match:.0%})"); return None

    slope = drift["slope_deg_per_s"]
    sec_day = (abs(slope) - 6.0) * 86400.0 / 6.0
    sec_day_se = drift["slope_se"] * 86400.0 / 6.0
    print(f"  match {match:.0%} | slope {slope:+.4f} deg/s "
          f"(SE {drift['slope_se']:.4f}) | RMS {drift['rms_deg']:.2f} deg | "
          f"N={drift['n']}")
    print(f"  >>> DRIFT {sec_day:+.1f} +/- {sec_day_se:.1f} s/day "
          f"({dur:.0f}s of video)")

    draw_fit_overlay(kymo_hp, ts, angles, accepted, slope, drift["intercept"],
                     ref, os.path.join(OUTPUT_DIR, f"{video_name}_fit.png"))

    with open(os.path.join(OUTPUT_DIR, f"{video_name}_angles.csv"), "w",
              newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["t_s", "angle_deg", "accepted"])
        for t, a, ok in zip(ts, angles, accepted):
            wr.writerow([f"{t:.4f}", f"{a:.3f}", int(ok)])

    print(f"  ({time.time()-t0:.1f}s)")
    return {
        "video": video_name, "slope": slope, "drift": sec_day,
        "drift_se": sec_day_se, "rms": drift["rms_deg"], "n": drift["n"],
        "match": match, "snr": snr0, "rot_total": rot_total, "dur": dur,
    }


def main(argv):
    targets = argv[1:] or sorted(
        os.path.join(VIDEOS_DIR, f) for f in os.listdir(VIDEOS_DIR)
        if f.upper().endswith((".MOV", ".MP4")))
    if not targets:
        print("no videos"); return
    print(f"v6 handheld validator — {len(targets)} video(s)")
    results = []
    for path in targets:
        name = os.path.splitext(os.path.basename(path))[0]
        try:
            r = process_video(path, name)
            if r:
                results.append(r)
        except Exception as e:
            import traceback
            print(f"  EXCEPTION: {e}"); traceback.print_exc()

    print(f"\n{'='*78}\nSUMMARY\n{'='*78}")
    print(f"{'Video':<14}{'Drift s/day':>14}{'+/-':>8}{'slope':>9}"
          f"{'RMS':>6}{'N':>6}{'Match':>7}{'Rot':>8}{'SNR':>6}")
    print("-" * 78)
    for r in results:
        print(f"{r['video']:<14}{r['drift']:>+13.1f}{r['drift_se']:>8.1f}"
              f"{r['slope']:>+9.4f}{r['rms']:>6.2f}{r['n']:>6}"
              f"{r['match']:>6.0%}{r['rot_total']:>+7.1f}d{r['snr']:>6.1f}")


if __name__ == "__main__":
    main(sys.argv)
