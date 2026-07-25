#!/usr/bin/env python3
"""Experiment: does a temporal high-pass per angle bin clean up the kymograph?

Hypothesis: the second hand is clearly present in the kymograph as a diagonal,
but static features (markers, GMT hand, text, bezel) create vertical banding
that image-space background subtraction doesn't remove. Subtracting a per-angle
moving average ALONG TIME should kill anything static/slow and leave only the
fast-moving second hand.

Outputs comparison kymographs to ml/diagnostics_exp/.
"""
import os
import sys
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(__file__))
from validate_v5 import (
    extract_frames, detect_watch_face, stabilize_phase_corr, apply_shift,
    make_polar_lut, sample_polar, _stack_at_velocity, _peak_snr,
    N_ANGLES, TARGET_FPS,
)

OUT = os.path.join(os.path.dirname(__file__), "diagnostics_exp")


def norm_img(a):
    return cv2.normalize(a, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def velocity_snr(kymo, vmin, vmax, step=0.05):
    """Best peak SNR over the velocity range; returns (vel, snr)."""
    best = (0.0, -1.0)
    for v in np.arange(vmin, vmax + step / 2, step):
        stack, _ = _stack_at_velocity(kymo.astype(np.float64), v, TARGET_FPS, N_ANGLES)
        _, snr, _, _, _ = _peak_snr(stack)
        if snr > best[1]:
            best = (float(v), float(snr))
    return best


def process(path, max_frames=1200):
    name = os.path.splitext(os.path.basename(path))[0]
    os.makedirs(OUT, exist_ok=True)
    print(f"\n=== {name} ===")
    frames, fps = extract_frames(path, max_frames=max_frames)
    print(f"  {len(frames)} frames @ {fps:.1f}fps")
    det = detect_watch_face(frames[0]) or detect_watch_face(frames[min(60, len(frames)-1)])
    if det is None:
        print("  no face"); return
    cx, cy, r = det
    side = int(r * 2.2)
    H, W = frames[0].shape[:2]
    x0 = max(0, min(W - side, int(cx - side / 2)))
    y0 = max(0, min(H - side, int(cy - side / 2)))
    side = min(side, W - x0, H - y0)
    grays = [cv2.cvtColor(f[y0:y0+side, x0:x0+side], cv2.COLOR_BGR2GRAY) for f in frames]
    shifts = stabilize_phase_corr(grays)
    stab = [apply_shift(g, dx, dy) for g, (dx, dy) in zip(grays, shifts)]

    cl = side / 2.0
    lut = make_polar_lut(side, cl, cl, side/2*0.50, side/2*0.85, N_ANGLES, 25)
    kymo_raw = np.stack([sample_polar(s.astype(np.float32), lut) for s in stab])  # (T, A)
    kymo_raw = kymo_raw[30:]
    T = kymo_raw.shape[0]

    # (a) raw
    # (b) image-space residual (v5 style)
    bg = np.median(np.array(stab[30:min(180, len(stab))], np.float32), axis=0)
    kymo_res = np.stack([
        sample_polar(cv2.GaussianBlur(np.abs(s.astype(np.float32) - bg), (7, 7), 0), lut)
        for s in stab])[30:]

    # (c) temporal high-pass: subtract a moving average along TIME for each angle.
    #     window ~ 45 frames (1.5s). Second hand sweeps 9deg in that time (fast);
    #     static/slow features are removed.
    def temporal_highpass(k, win=45):
        kpad = np.pad(k, ((win, win), (0, 0)), mode="edge")
        ker = np.ones(2 * win + 1) / (2 * win + 1)
        lp = np.stack([np.convolve(kpad[:, a], ker, mode="valid") for a in range(k.shape[1])], axis=1)
        return k - lp
    kymo_hp = temporal_highpass(kymo_raw, win=45)
    kymo_hp_abs = np.abs(kymo_hp)

    # (d) temporal high-pass of the residual image kymograph too
    kymo_res_hp = np.abs(temporal_highpass(kymo_res, win=45))

    cv2.imwrite(os.path.join(OUT, f"{name}_a_raw.png"), norm_img(kymo_raw))
    cv2.imwrite(os.path.join(OUT, f"{name}_b_imgres.png"), norm_img(kymo_res))
    cv2.imwrite(os.path.join(OUT, f"{name}_c_hp.png"), norm_img(kymo_hp_abs))
    cv2.imwrite(os.path.join(OUT, f"{name}_d_reshp.png"), norm_img(kymo_res_hp))

    # SNR of the second-hand velocity peak. Search a wide range that includes
    # camera rotation (the second hand is somewhere in +/-[3, 10] deg/s).
    for label, k in [("raw", kymo_raw), ("imgres", kymo_res),
                     ("hp", kymo_hp_abs), ("reshp", kymo_res_hp)]:
        # subtract per-row mean so DC doesn't dominate the stack
        kz = k - k.mean(axis=1, keepdims=True)
        v_pos, snr_pos = velocity_snr(kz, 3.0, 10.0)
        v_neg, snr_neg = velocity_snr(kz, -10.0, -3.0)
        v0, snr0 = velocity_snr(kz, -2.5, 2.5)  # static/rotation band
        best_v, best_snr = (v_pos, snr_pos) if snr_pos >= snr_neg else (v_neg, snr_neg)
        print(f"  {label:7s}: sec-hand v={best_v:+.2f} SNR={best_snr:5.1f} | "
              f"static-band peak SNR={snr0:5.1f} (v={v0:+.2f})  "
              f"[ratio sec/static = {best_snr/max(snr0,1e-3):.2f}]")


if __name__ == "__main__":
    targets = sys.argv[1:] or [
        os.path.join(os.path.dirname(__file__), "videos", "IMG_7720.MOV"),
    ]
    for t in targets:
        process(t)
