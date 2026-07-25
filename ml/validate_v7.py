#!/usr/bin/env python3
"""Watch drift validator v7 — consolidates everything that works.

Built for the NEW footage (multi-revolution; a stable/propped clip and a
handheld clip). Incorporates the validated findings:

  * REAL per-frame timestamps (CAP_PROP_POS_MSEC), never assumed fps
    (iPhone shoots 59.94fps VFR with dropped frames).
  * Temporal high-pass per angle bin to isolate the second-hand ridge
    (validated 3-5x SNR gain).
  * Sub-bin intensity-weighted centroid tracking + robust (trimmed) fit.
  * Optional de-rotation (12th-harmonic) for handheld; OFF for stable.
  * Harmonic center-calibration folded into the slope fit (needs >=1 rev) so a
    center offset is absorbed instead of biasing the rate.

Synthetic validation (exp_synth.py) shows this chain recovers a known
6.000 deg/s to +/-5 s/day over ONE revolution when rotation is smooth.

Usage:
  validate_v7.py video.MOV [more.MOV ...]
  validate_v7.py --stable video.MOV     # skip de-rotation (propped/tripod)
"""
import os, sys, time
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(__file__))
from validate_v5 import (detect_watch_face, stabilize_phase_corr, apply_shift,
                         make_polar_lut, sample_polar, per_frame_rotation_harmonic,
                         radon_search, refine_velocity)

N_A = 720
HP_WIN = 45
WARMUP = 30
OUT = os.path.join(os.path.dirname(__file__), "diagnostics_v7")


def load_crops(video_path, target_fps=30, max_frames=8000, max_h=1080):
    """Memory-efficient: detect the watch on the first usable frame, then for
    every kept frame downsample + crop + grayscale on the fly, keeping ONLY the
    small watch crop (not the full 4K frame). Returns (grays, ts, side, r)."""
    cap = cv2.VideoCapture(video_path)
    native = cap.get(cv2.CAP_PROP_FPS) or 60.0
    skip = max(1, int(round(native/target_fps)))
    grays, ts = [], []
    crop = None
    idx = 0
    while len(grays) < max_frames:
        ok, fr = cap.read()
        if not ok:
            break
        t = cap.get(cv2.CAP_PROP_POS_MSEC)/1000.0
        if idx % skip == 0:
            h, w = fr.shape[:2]
            if h > max_h:
                s = max_h/h
                fr = cv2.resize(fr, (int(w*s), int(h*s)), interpolation=cv2.INTER_AREA)
            if crop is None:
                det = detect_watch_face(fr)
                if det is None:
                    idx += 1
                    continue  # try the next kept frame for detection
                cx, cy, r = det
                side = int(r*2.2); H, W = fr.shape[:2]
                x0 = max(0, min(W-side, int(round(cx-side/2))))
                y0 = max(0, min(H-side, int(round(cy-side/2))))
                side = min(side, W-x0, H-y0)
                crop = (x0, y0, side, r)
            x0, y0, side, r = crop
            g = cv2.cvtColor(fr[y0:y0+side, x0:x0+side], cv2.COLOR_BGR2GRAY)
            grays.append(g); ts.append(t)
        idx += 1
    cap.release()
    if crop is None:
        return [], np.array([]), 0, 0.0
    return grays, np.array(ts), crop[2], crop[3]


def temporal_highpass(kymo, win=HP_WIN):
    n_t, n_a = kymo.shape
    kpad = np.pad(kymo, ((win, win), (0, 0)), mode="edge")
    ker = np.ones(2*win+1, np.float32)/(2*win+1)
    lp = np.empty_like(kymo)
    for a in range(n_a):
        lp[:, a] = np.convolve(kpad[:, a], ker, mode="valid")
    return kymo - lp


def derotate_12h(kymo_raw):
    rb = per_frame_rotation_harmonic(kymo_raw, harmonic=12)
    return np.stack([np.roll(kymo_raw[i], -int(round(rb[i]))) for i in range(len(rb))]), rb


def unwrap_to_line(angles_deg, ts, vel, phase):
    out = []
    for t, a in zip(ts, angles_deg):
        p = phase + vel*t
        out.append(p + (((a-p+540) % 360)-180))
    return np.array(out)


def track(kymo_derot, ts):
    hp = np.abs(temporal_highpass(kymo_derot))
    kz = hp - hp.mean(axis=1, keepdims=True)
    n_t, n_a = kz.shape; bpd = n_a/360.0
    coarse = radon_search(kz, 30.0, n_a, vel_min=4.5, vel_max=7.5)
    vel, peak, ref = refine_velocity(kz, coarse[0], 30.0, n_a, window=0.3, step=0.005)
    phase = (peak*360.0/n_a) - vel*ts[ref]
    win = int(round(2.5*bpd))
    ang = np.zeros(n_t); ok = np.zeros(n_t, bool); pred = np.zeros(n_t)
    for i in range(n_t):
        p = phase+vel*ts[i]; pred[i] = p
        pb = int(round((p % 360.0)*bpd)) % n_a
        js = np.arange(pb-win, pb+win+1)
        vals = np.clip(kz[i, js % n_a], 0, None); wsum = vals.sum()
        if wsum > 1e-6:
            ang[i] = ((js*vals).sum()/wsum*360.0/n_a) % 360.0
            am = js[int(np.argmax(vals))] % n_a
            ok[i] = abs(((am*360.0/n_a-(p % 360.0)+540) % 360)-180) < 2.5*0.9
        else:
            ang[i] = p % 360.0
    y = unwrap_to_line(ang[ok], ts[ok], vel, phase)
    return ts[ok], y, pred[ok], coarse[1], vel


def fit_harmonic(t, y, pred_deg, nharm=2, iters=6, k=2.5):
    pr = np.radians(pred_deg)
    cols = [np.ones_like(t), t]
    for h in range(1, nharm+1):
        cols += [np.cos(h*pr), np.sin(h*pr)]
    A = np.column_stack(cols)
    mask = np.ones(len(t), bool); coef = None; rms = 0
    for _ in range(iters):
        coef, *_ = np.linalg.lstsq(A[mask], y[mask], rcond=None)
        res = y - A@coef; rms = np.sqrt((res[mask]**2).mean())
        nm = np.abs(res) <= k*max(rms, 1e-6)
        if nm.sum() == mask.sum():
            break
        mask = nm
    AtA_inv = np.linalg.pinv(A[mask].T@A[mask])
    se = np.sqrt(max(AtA_inv[1, 1], 0))*rms
    amp1 = np.hypot(coef[2], coef[3])
    return coef[1], se, rms, int(mask.sum()), amp1


def fit_plain(t, y, iters=6, k=2.5):
    mask = np.ones(len(t), bool); coef = None; rms = 0
    A = np.c_[np.ones_like(t), t]
    for _ in range(iters):
        coef, *_ = np.linalg.lstsq(A[mask], y[mask], rcond=None)
        res = y - A@coef; rms = np.sqrt((res[mask]**2).mean())
        nm = np.abs(res) <= k*max(rms, 1e-6)
        if nm.sum() == mask.sum():
            break
        mask = nm
    se = rms/np.sqrt(((t[mask]-t[mask].mean())**2).sum())
    return coef[1], se, rms, int(mask.sum())


def process(video_path, stable=False):
    os.makedirs(OUT, exist_ok=True)
    name = os.path.splitext(os.path.basename(video_path))[0]
    print(f"\n{'='*70}\n  {name}  {'(stable)' if stable else '(handheld)'}\n{'='*70}")
    t0 = time.time()
    grays, ts, side, r = load_crops(video_path)
    if len(grays) < 120:
        print("  too few frames / watch not found"); return
    shifts = stabilize_phase_corr(grays)
    stab = [apply_shift(g, dx, dy) for g, (dx, dy) in zip(grays, shifts)]
    cl = side/2.0
    lut = make_polar_lut(side, cl, cl, cl*0.50, cl*0.85, N_A, 25)
    kymo = np.stack([sample_polar(s.astype(np.float32), lut) for s in stab])[WARMUP:]
    ts = ts[WARMUP:]
    dur = ts[-1]-ts[0]
    print(f"  {len(grays)} frames, dur={dur:.1f}s ({dur/60:.2f} rev), r={r:.0f}px")

    variants = [("raw", kymo)]
    if not stable:
        d12, rb = derotate_12h(kymo)
        rot = (rb[-1]-rb[0])*360.0/N_A
        print(f"  12h rotation estimate: {rot:+.2f} deg")
        variants.append(("12h-derot", d12))

    for tag, kd in variants:
        t, y, pred, snr, vel = track(kd, ts)
        if len(t) < 60:
            print(f"  {tag}: too few tracked points"); continue
        mp, sep, rmsp, np_ = fit_plain(t, y)
        mh, seh, rmsh, nh, amp1 = fit_harmonic(t, y, pred)
        dp, dpse = (mp-6)/6*86400, sep/6*86400
        dh, dhse = (mh-6)/6*86400, seh/6*86400
        warn = "" if dur >= 55 else "  [<1 rev: harmonic fit unreliable]"
        print(f"  {tag:10s} snr={snr:3.0f} N={nh}")
        print(f"     plain    : {dp:+8.1f} +/- {dpse:5.1f} s/day  (slope {mp:+.5f}, rms {rmsp:.2f})")
        print(f"     harmonic : {dh:+8.1f} +/- {dhse:5.1f} s/day  (slope {mh:+.5f}, rms {rmsh:.2f}, "
              f"center {amp1:.2f}deg){warn}")
    print(f"  ({time.time()-t0:.1f}s)")


def main(argv):
    args = [a for a in argv[1:] if not a.startswith("--")]
    stable = "--stable" in argv
    if not args:
        print("usage: validate_v7.py [--stable] video.MOV ..."); return
    for v in args:
        process(v, stable=stable)


if __name__ == "__main__":
    main(sys.argv)
