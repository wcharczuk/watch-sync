#!/usr/bin/env python3
"""Robust handheld de-rotation via 2D feature tracking (the real attack).

The 12th-harmonic de-rotation breaks at large wrist rotation (-120deg in
IMG_7854 -> +868 s/day garbage). Here we track the dial's 2D texture with
Lucas-Kanade optical flow against keyframes, fit per-frame rotation about the
watch center with a RANSAC partial-affine (which rejects the moving second hand
as outliers), and re-anchor keyframes as rotation accumulates. This handles
arbitrary total rotation and uses the dial's asymmetric text to stay locked.

Goal: bring IMG_7854 (handheld) into agreement with the stable clip (~+32-65).

Caches the decoded grayscale crops to ml/cache/<name>_crops.npy so the 4K decode
(~3min) happens once.
"""
import os, sys, time
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(__file__))
from validate_v5 import stabilize_phase_corr, apply_shift, make_polar_lut, sample_polar
from validate_v7 import (load_crops, temporal_highpass, track, fit_plain,
                         fit_harmonic, derotate_12h, N_A, HP_WIN, WARMUP)

CACHE = os.path.join(os.path.dirname(__file__), "cache")
OUT = os.path.join(os.path.dirname(__file__), "diagnostics_v7")


def get_crops(video_path, name):
    os.makedirs(CACHE, exist_ok=True)
    gp = os.path.join(CACHE, f"{name}_crops.npy")
    mp = os.path.join(CACHE, f"{name}_crops_meta.npz")
    if os.path.exists(gp) and os.path.exists(mp):
        m = np.load(mp)
        grays = np.load(gp, mmap_mode="r")
        return grays, m["ts"], int(m["side"]), float(m["r"])
    print(f"  decoding {name} (one-time) ...")
    grays, ts, side, r = load_crops(video_path)
    grays = np.stack(grays).astype(np.uint8)
    np.save(gp, grays)
    np.savez(mp, ts=ts, side=side, r=r)
    return grays, ts, side, r


def track_rotation_lk(stab, cx, cy, r, side,
                      re_kf_deg=18.0, min_inliers=40):
    """Per-frame absolute rotation (deg, relative to frame 0) via LK feature
    tracking + RANSAC partial-affine, with keyframe re-anchoring."""
    n = len(stab)
    # annulus mask: dial markers + text + bezel, exclude center hub
    yy, xx = np.mgrid[0:side, 0:side]
    rr = np.hypot(xx-cx, yy-cy)
    mask = ((rr > 0.18*r) & (rr < 1.02*r)).astype(np.uint8)*255

    def detect(g):
        return cv2.goodFeaturesToTrack(g, maxCorners=500, qualityLevel=0.01,
                                       minDistance=6, mask=mask, blockSize=7)
    lk = dict(winSize=(21, 21), maxLevel=3,
              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

    rot = np.zeros(n)
    kf_gray = np.asarray(stab[0]); kf_rot = 0.0
    p_kf = detect(kf_gray)
    n_kf = 1
    for i in range(1, n):
        cur = np.asarray(stab[i])
        p_i, stt, _ = cv2.calcOpticalFlowPyrLK(kf_gray, cur, p_kf, None, **lk)
        ok = (stt.ravel() == 1)
        gk = p_kf[ok].reshape(-1, 2); gi = p_i[ok].reshape(-1, 2)
        theta = 0.0; ninl = 0
        if len(gi) >= 12:
            M, inl = cv2.estimateAffinePartial2D(gk, gi, method=cv2.RANSAC,
                                                 ransacReprojThreshold=2.0)
            if M is not None and inl is not None:
                theta = np.degrees(np.arctan2(M[1, 0], M[0, 0]))
                ninl = int(inl.sum())
        rot[i] = kf_rot + theta
        # re-anchor keyframe if rotation from kf is large or tracking weak
        if abs(theta) > re_kf_deg or ninl < min_inliers:
            kf_gray = cur; kf_rot = rot[i]; p_kf = detect(cur); n_kf += 1
    return rot, n_kf


def fourier_shift_rows(kymo, shifts):
    n_t, n_a = kymo.shape
    k = np.fft.rfftfreq(n_a)
    F = np.fft.rfft(kymo, axis=1)
    return np.fft.irfft(F*np.exp(2j*np.pi*np.outer(shifts, k)), n=n_a, axis=1)


def measure(kymo_derot, ts, tag):
    t, y, pred, snr, vel = track(kymo_derot, ts)
    if len(t) < 60:
        print(f"  {tag:10s}: too few tracked points"); return None
    mp, sep, rmsp, _ = fit_plain(t, y)
    mh, seh, rmsh, nh, amp1 = fit_harmonic(t, y, pred)
    dp = (mp-6)/6*86400; dh = (mh-6)/6*86400
    print(f"  {tag:10s} snr={snr:4.0f}  plain {dp:+8.1f}+/-{sep/6*86400:4.1f}  "
          f"harmonic {dh:+8.1f}+/-{seh/6*86400:4.1f} s/day (rms {rmsh:.2f}, center {amp1:.2f}d)")
    return dh


def process(video_path):
    name = os.path.splitext(os.path.basename(video_path))[0]
    print(f"\n{'='*70}\n  {name}\n{'='*70}")
    t0 = time.time()
    grays, ts, side, r = get_crops(video_path, name)
    cl = side/2.0
    # translation stabilize (lock center) before rotation tracking
    glist = [np.asarray(grays[i]) for i in range(len(grays))]
    shifts = stabilize_phase_corr(glist)
    stab = [apply_shift(g, dx, dy) for g, (dx, dy) in zip(glist, shifts)]
    lut = make_polar_lut(side, cl, cl, cl*0.50, cl*0.85, N_A, 25)
    kymo = np.stack([sample_polar(s.astype(np.float32), lut) for s in stab])[WARMUP:]
    ts = ts[WARMUP:]
    dur = ts[-1]-ts[0]
    print(f"  {len(stab)} frames, dur={dur:.1f}s ({dur/60:.2f} rev), r={r:.0f}px ({time.time()-t0:.0f}s load)")

    # baseline: 12h
    d12, rb = derotate_12h(kymo)
    print(f"  12h rot={ (rb[-1]-rb[0])*360/N_A:+.1f}deg")
    measure(d12, ts, "12h")

    # LK feature-tracking de-rotation
    tlk = time.time()
    rot_deg, n_kf = track_rotation_lk(stab[WARMUP:], cl, cl, r, side)
    print(f"  LK rot={rot_deg[-1]-rot_deg[0]:+.1f}deg ({n_kf} keyframes, {time.time()-tlk:.0f}s)")
    rot_bins = rot_deg * N_A/360.0
    # try both signs (convention) — report whichever locks (higher snr handled in measure)
    d_lk = fourier_shift_rows(kymo, rot_bins)
    measure(d_lk, ts, "LK +")
    d_lk2 = fourier_shift_rows(kymo, -rot_bins)
    measure(d_lk2, ts, "LK -")
    print(f"  (total {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    targets = sys.argv[1:] or ["videos/IMG_7854.MOV"]
    for t in targets:
        process(t)
