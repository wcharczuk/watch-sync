#!/usr/bin/env python3
"""Run the single-frame second-hand reader on every Nth frame of a video and
plot detected angle vs frame index. A clean detector should produce a steady
+6°/s ramp; noise/random results indicate the detector is broken.

Usage:
  validate_singleframe.py videos/IMG_xxxx.MOV [stride]
"""

import os
import sys

import cv2
import numpy as np

from read_seconds import read_second_hand


def main(argv):
    if len(argv) < 2:
        print("usage: validate_singleframe.py <video> [stride]")
        return
    video_path = argv[1]
    stride = int(argv[2]) if len(argv) > 2 else 1

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"video: {n_total} frames @ {fps:.1f} fps, processing every {stride}th")

    name = os.path.splitext(os.path.basename(video_path))[0]
    out_dir = os.path.join(os.path.dirname(__file__), "diagnostics_singleframe")
    os.makedirs(out_dir, exist_ok=True)

    # Per-frame: collect TOP-5 candidates so we can study which one is the
    # second hand. The "winner" alone may be wrong; the second hand is
    # whichever candidate moves at +6°/s.
    timestamps = []
    angles = []
    confs = []
    fwhms = []
    candidates_per_frame = []  # list of [(angle, height, fwhm), ...]
    failures = 0

    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % stride == 0:
            h, w = frame.shape[:2]
            if h > 1500:
                s = 1440 / h
                frame = cv2.resize(frame, (int(w * s), int(h * s)),
                                   interpolation=cv2.INTER_AREA)
            r = read_second_hand(frame)
            t = idx / fps
            if r is None:
                failures += 1
                candidates_per_frame.append([])
                timestamps.append(t)
                angles.append(np.nan)
                confs.append(0)
                fwhms.append(0)
            else:
                timestamps.append(t)
                angles.append(r.angle_deg)
                confs.append(r.confidence)
                fwhms.append(r.fwhm_deg)
                candidates_per_frame.append(r.debug["top_candidates"])
        idx += 1
    cap.release()

    if not angles:
        print("All frames failed.")
        return

    timestamps = np.array(timestamps)
    angles = np.array(angles)
    confs = np.array(confs)
    fwhms = np.array(fwhms)

    # ---------------------------------------------------------------------
    # Sparse Radon search: find the +6°/s line that hits the most candidates.
    # Each frame contributes up to 5 candidate angles; the second hand is the
    # trajectory that's a straight line at +6°/s.
    # ---------------------------------------------------------------------
    def hits_for_line(velocity_deg_per_s, intercept_deg, tol_deg=2.0):
        hit_angles = []
        hit_times = []
        for t, cands in zip(timestamps, candidates_per_frame):
            pred = (intercept_deg + velocity_deg_per_s * t) % 360.0
            best_dev = None
            best_ang = None
            for ang, _, _ in cands:
                d = ((ang - pred + 540) % 360) - 180
                if best_dev is None or abs(d) < abs(best_dev):
                    best_dev = d
                    best_ang = ang
            if best_dev is not None and abs(best_dev) <= tol_deg:
                hit_angles.append(best_ang)
                hit_times.append(t)
        return hit_times, hit_angles

    best_score = -1
    best_v = 6.0
    best_intercept = 0.0
    # Two-stage search: coarse over velocity ±3°/s of nominal, intercept 0..360
    for v in np.arange(5.7, 6.31, 0.05):  # 13 velocities
        for i0 in np.arange(0, 360, 5):  # 72 intercepts
            ts, angs = hits_for_line(v, i0, tol_deg=3.0)
            if len(ts) > best_score:
                best_score = len(ts)
                best_v = v
                best_intercept = i0
    # Refine
    for v in np.arange(best_v - 0.05, best_v + 0.051, 0.005):
        for i0 in np.arange(best_intercept - 5, best_intercept + 5.01, 0.5):
            ts, angs = hits_for_line(v, i0 % 360, tol_deg=3.0)
            if len(ts) > best_score:
                best_score = len(ts)
                best_v = v
                best_intercept = i0 % 360

    hit_times, hit_angles = hits_for_line(best_v, best_intercept, tol_deg=3.0)
    print(f"  Radon: best_v={best_v:.4f}°/s, intercept={best_intercept:.2f}°, "
          f"hits={best_score}/{len(timestamps)}")

    # Linear regression on the hit angles (with unwrap relative to the line)
    if len(hit_times) >= 30:
        ht = np.array(hit_times)
        ha = np.array(hit_angles)
        # Unwrap: align each hit angle to the predicted line so the regression
        # is on a continuous (nearly-linear) signal.
        pred = best_intercept + best_v * ht
        ha_unwrapped = []
        for t, a in zip(ht, ha):
            p = best_intercept + best_v * t
            d = ((a - p + 540) % 360) - 180
            ha_unwrapped.append(p + d)
        ha_unwrapped = np.array(ha_unwrapped)
        mt = ht.mean()
        ma = ha_unwrapped.mean()
        slope_fit = ((ht - mt) * (ha_unwrapped - ma)).sum() / max(((ht - mt) ** 2).sum(), 1e-9)
        intercept_fit = ma - slope_fit * mt
        residuals_fit = ha_unwrapped - (intercept_fit + slope_fit * ht)
        rms_fit = float(np.sqrt((residuals_fit ** 2).mean()))
        n_fit = len(ht)
        slope_se = rms_fit / np.sqrt(((ht - mt) ** 2).sum())
        drift_radon = (slope_fit - 6.0) * 86400 / 6.0
        drift_se = slope_se * 86400 / 6.0
        print(f"  Radon-fit slope={slope_fit:.5f}°/s ±{slope_se:.5f}, RMS={rms_fit:.2f}°")
        print(f"  Drift = {drift_radon:+.1f} ± {drift_se:.1f} s/day  (N={n_fit})")
    else:
        print(f"  Too few hits ({len(hit_times)}) for regression.")
        slope_fit = best_v
        rms_fit = 0
        n_fit = 0

    # Plot ALL candidates from ALL frames in a (time, angle) scatter — a
    # +6°/s second-hand trajectory should jump out as a sloped line.
    plot_w = 1400
    plot_h = 720
    scatter = np.zeros((plot_h, plot_w, 3), dtype=np.uint8)
    pad = 30
    t_max = timestamps.max() if len(timestamps) > 0 else 1
    for ti, cands in zip(timestamps, candidates_per_frame):
        x = int(pad + (ti / max(t_max, 1e-3)) * (plot_w - 2 * pad))
        for k, (ang, height, fw) in enumerate(cands):
            y = int(pad + (ang / 360.0) * (plot_h - 2 * pad))
            # The k-th best candidate gets dimmer
            shade = max(60, 230 - k * 40)
            cv2.circle(scatter, (x, y), 1, (0, shade, shade), -1)
    # Mark the WINNER per frame brighter
    for ti, ang, c in zip(timestamps, angles, confs):
        if np.isnan(ang):
            continue
        x = int(pad + (ti / max(t_max, 1e-3)) * (plot_w - 2 * pad))
        y = int(pad + (ang / 360.0) * (plot_h - 2 * pad))
        cv2.circle(scatter, (x, y), 2, (0, 0, 255), -1)
    cv2.putText(scatter, f"{name}  cyan = top-5 candidates per frame, "
                f"red = winner. Y axis = angle (0-360°), X axis = time.",
                (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)
    cv2.imwrite(os.path.join(out_dir, f"{name}_candidates.png"), scatter)

    # Unwrap angles (assume clockwise +6°/s)
    unw = [angles[0]]
    for a in angles[1:]:
        d = a - (unw[-1] % 360)
        if d > 180: d -= 360
        if d < -180: d += 360
        unw.append(unw[-1] + d)
    unw = np.array(unw)

    # Linear fit
    mt = timestamps.mean()
    ma = unw.mean()
    slope = ((timestamps - mt) * (unw - ma)).sum() / max(((timestamps - mt) ** 2).sum(), 1e-9)
    intercept = ma - slope * mt
    residuals = unw - (intercept + slope * timestamps)
    rms = float(np.sqrt((residuals ** 2).mean()))
    drift_sec_per_day = (slope - 6.0) * 86400 / 6.0

    # Plot
    plot_w = 1400
    plot_h = 600
    plot = np.zeros((plot_h, plot_w, 3), dtype=np.uint8)

    # Predicted line: 6°/s starting from intercept of fit at t=0
    # Compress residuals scaled around fit
    res_max = max(np.abs(residuals).max(), 5)
    pad = 30
    for i, (t, r) in enumerate(zip(timestamps, residuals)):
        x = int(pad + (t / max(timestamps.max(), 1e-3)) * (plot_w - 2 * pad))
        y = int(plot_h / 2 - r / res_max * (plot_h / 2 - pad))
        col = (0, 200, 255) if confs[i] < 1.5 else (0, 255, 0)
        cv2.circle(plot, (x, y), 2, col, -1)
    # Zero line
    cv2.line(plot, (pad, plot_h // 2), (plot_w - pad, plot_h // 2),
             (80, 80, 80), 1)
    cv2.putText(plot,
                f"{name}  fit slope={slope:.4f} deg/s  drift={drift_sec_per_day:+.0f} s/day  "
                f"RMS={rms:.2f}deg  N={len(angles)}/{n_total // stride}  fail={failures}",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(plot, f"y axis: residual from fit, ±{res_max:.1f} deg",
                (10, plot_h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
    cv2.imwrite(os.path.join(out_dir, f"{name}_residuals.png"), plot)

    # Also save the unwrapped angle plot for quick eyeballing
    plot2 = np.zeros((plot_h, plot_w, 3), dtype=np.uint8)
    a_min = unw.min()
    a_max = unw.max()
    for i, (t, a) in enumerate(zip(timestamps, unw)):
        x = int(pad + (t / max(timestamps.max(), 1e-3)) * (plot_w - 2 * pad))
        y = int(plot_h - pad - (a - a_min) / max(a_max - a_min, 1e-3) * (plot_h - 2 * pad))
        col = (0, 200, 255) if confs[i] < 1.5 else (0, 255, 0)
        cv2.circle(plot2, (x, y), 2, col, -1)
    # Predicted line
    for t in np.linspace(timestamps.min(), timestamps.max(), 200):
        a_pred = intercept + slope * t
        x = int(pad + (t / max(timestamps.max(), 1e-3)) * (plot_w - 2 * pad))
        y = int(plot_h - pad - (a_pred - a_min) / max(a_max - a_min, 1e-3) * (plot_h - 2 * pad))
        cv2.circle(plot2, (x, y), 1, (255, 100, 100), -1)
    cv2.putText(plot2, f"{name}  unwrapped angle vs time  slope={slope:.3f} deg/s",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.imwrite(os.path.join(out_dir, f"{name}_angles.png"), plot2)

    print(f"\n{name}:")
    print(f"  N={len(angles)}, failures={failures}")
    print(f"  slope = {slope:.4f}°/s   (target +6.000)")
    print(f"  RMS residuals = {rms:.2f}°")
    print(f"  drift = {drift_sec_per_day:+.0f} s/day")
    print(f"  conf:  mean={confs.mean():.2f} median={np.median(confs):.2f}")
    print(f"  fwhm:  mean={fwhms.mean():.2f}° median={np.median(fwhms):.2f}°")


if __name__ == "__main__":
    main(sys.argv)
