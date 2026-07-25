#!/usr/bin/env python3
"""Render kymographs for the handheld clip to SEE what's happening (uses cached
crops; no re-decode). Shows: raw kymograph (static features slope by the true
total rotation), and high-pass + tracked second hand for raw / 12h / LK."""
import os, sys
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(__file__))
from validate_v5 import stabilize_phase_corr, apply_shift, make_polar_lut, sample_polar
from validate_v7 import temporal_highpass, track, fit_harmonic, fit_plain, derotate_12h, N_A, WARMUP
from exp_handheld import get_crops, track_rotation_lk, fourier_shift_rows

OUT = os.path.join(os.path.dirname(__file__), "diagnostics_v7")


def hp_track_img(kymo_derot, ts, tag, name):
    hp = np.abs(temporal_highpass(kymo_derot))
    kz = hp - hp.mean(axis=1, keepdims=True)
    img = cv2.normalize(kz, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    t, y, pred, snr, vel = track(kymo_derot, ts)
    mh, seh, rmsh, nh, amp1 = fit_harmonic(t, y, pred)
    dh = (mh-6)/6*86400
    cv2.putText(img, f"{name} [{tag}] snr={snr:.0f} drift={dh:+.0f}s/day rms={rmsh:.2f}",
                (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    cv2.imwrite(os.path.join(OUT, f"{name}_HH_{tag}.png"), img)
    print(f"  {tag}: snr={snr:.0f} drift={dh:+.0f} s/day")


def main(video):
    name = os.path.splitext(os.path.basename(video))[0]
    grays, ts, side, r = get_crops(video, name)
    cl = side/2.0
    glist = [np.asarray(grays[i]) for i in range(len(grays))]
    shifts = stabilize_phase_corr(glist)
    stab = [apply_shift(g, dx, dy) for g, (dx, dy) in zip(glist, shifts)]
    lut = make_polar_lut(side, cl, cl, cl*0.50, cl*0.85, N_A, 25)
    kymo = np.stack([sample_polar(s.astype(np.float32), lut) for s in stab])[WARMUP:]
    ts = ts[WARMUP:]

    # raw kymograph (NOT high-passed) to read off total rotation from static slopes
    raw_img = cv2.normalize(kymo, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    cv2.imwrite(os.path.join(OUT, f"{name}_HH_rawkymo.png"), raw_img)
    print(f"  wrote raw kymograph ({kymo.shape[0]} rows x {N_A})")

    hp_track_img(kymo, ts, "raw", name)
    d12, rb = derotate_12h(kymo)
    print(f"  12h total rot = {(rb[-1]-rb[0])*360/N_A:+.1f} deg")
    hp_track_img(d12, ts, "12h", name)
    rot_deg, n_kf = track_rotation_lk(stab[WARMUP:], cl, cl, r, side)
    print(f"  LK total rot = {rot_deg[-1]-rot_deg[0]:+.1f} deg ({n_kf} kf)")
    hp_track_img(fourier_shift_rows(kymo, rot_deg*N_A/360.0), ts, "LKpos", name)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "videos/IMG_7854.MOV")
