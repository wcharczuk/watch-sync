#!/usr/bin/env python3
"""Iterate on the *measurement* stage with real timestamps.

The expensive part (decode + stabilize + kymograph + de-rotate) is cached to
ml/cache/<name>.npz so we can iterate the slope-estimation quickly.

Compares slope estimators:
  - argmax  : greedy per-frame local max (v6 original)
  - centroid: sub-bin intensity-weighted centroid in a window
  - robust  : centroid + Theil-Sen-ish trimmed regression
all on REAL per-frame timestamps (POS_MSEC), not assumed fps.
"""
import os, sys, time
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(__file__))
from validate_v5 import (
    detect_watch_face, stabilize_phase_corr, apply_shift,
    make_polar_lut, sample_polar, per_frame_rotation_harmonic,
    radon_search, refine_velocity, N_ANGLES,
)
from validate_v6 import temporal_highpass, HP_WIN, WARMUP

CACHE = os.path.join(os.path.dirname(__file__), "cache")


def extract_with_ts(video_path, target_fps=30, max_frames=4000, max_h=1440):
    """Return (frames, timestamps_sec) using real POS_MSEC, subsampled toward
    target_fps. Timestamps are the genuine presentation times of kept frames."""
    cap = cv2.VideoCapture(video_path)
    native = cap.get(cv2.CAP_PROP_FPS) or 60.0
    skip = max(1, int(round(native / target_fps)))
    frames, ts = [], []
    idx = 0
    while len(frames) < max_frames:
        ok, fr = cap.read()
        if not ok:
            break
        t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        if idx % skip == 0:
            h, w = fr.shape[:2]
            if h > max_h:
                s = max_h / h
                fr = cv2.resize(fr, (int(w*s), int(h*s)), interpolation=cv2.INTER_AREA)
            frames.append(fr)
            ts.append(t)
        idx += 1
    cap.release()
    return frames, np.array(ts)


CACHE_VER = 2  # bump to invalidate old caches


def build_cache(video_path, name):
    """Cache the RAW (pre-derotation) kymograph + real timestamps + crop info,
    so de-rotation and measurement can be iterated without re-decoding."""
    os.makedirs(CACHE, exist_ok=True)
    p = os.path.join(CACHE, f"{name}_v{CACHE_VER}.npz")
    if os.path.exists(p):
        d = np.load(p)
        return d["kymo_raw"], d["ts"], dict(side=int(d["side"]))
    print(f"  building cache for {name} ...")
    frames, ts = extract_with_ts(video_path)
    det = detect_watch_face(frames[0]) or detect_watch_face(frames[min(60, len(frames)-1)])
    cx, cy, r = det
    side = int(r*2.2); H, W = frames[0].shape[:2]
    x0 = max(0, min(W-side, int(round(cx-side/2)))); y0 = max(0, min(H-side, int(round(cy-side/2))))
    side = min(side, W-x0, H-y0)
    grays = [cv2.cvtColor(f[y0:y0+side, x0:x0+side], cv2.COLOR_BGR2GRAY) for f in frames]
    shifts = stabilize_phase_corr(grays)
    stab = [apply_shift(g, dx, dy) for g, (dx, dy) in zip(grays, shifts)]
    cl = side/2.0
    lut = make_polar_lut(side, cl, cl, cl*0.50, cl*0.85, N_ANGLES, 25)
    kymo_raw = np.stack([sample_polar(s.astype(np.float32), lut) for s in stab])[WARMUP:]
    ts = ts[WARMUP:]
    np.savez_compressed(p, kymo_raw=kymo_raw.astype(np.float32), ts=ts, side=side)
    return kymo_raw, ts, dict(side=side)


def derotate_12h(kymo_raw):
    rot_bins = per_frame_rotation_harmonic(kymo_raw, harmonic=12)
    derot = np.stack([np.roll(kymo_raw[i], -int(round(rot_bins[i])))
                      for i in range(kymo_raw.shape[0])])
    return derot, rot_bins


def unwrap_to_line(angles_deg, ts, vel, phase):
    """Unwrap each angle to the prediction (phase + vel*t) so the series is
    continuous for regression."""
    out = []
    for t, a in zip(ts, angles_deg):
        p = phase + vel*t
        d = ((a - p + 540) % 360) - 180
        out.append(p + d)
    return np.array(out)


def lsq_slope(ts, y):
    mt, my = ts.mean(), y.mean()
    s = ((ts-mt)*(y-my)).sum() / max(((ts-mt)**2).sum(), 1e-12)
    b = my - s*mt
    res = y - (s*ts + b)
    rms = float(np.sqrt((res**2).mean()))
    se = rms/np.sqrt(((ts-mt)**2).sum()) if len(ts) > 2 else float('inf')
    return s, b, rms, se, res


def robust_slope(ts, y, iters=5, k=2.5):
    """Iteratively reweighted (trimmed) least squares: drop points >k*RMS."""
    mask = np.ones(len(ts), bool)
    s = b = rms = se = 0.0
    for _ in range(iters):
        s, b, rms, se, res = lsq_slope(ts[mask], y[mask])
        full_res = y - (s*ts + b)
        newmask = np.abs(full_res) <= k*max(rms, 1e-6)
        if newmask.sum() == mask.sum() and (newmask == mask).all():
            break
        if newmask.sum() < 30:
            break
        mask = newmask
    return s, b, rms, se, mask


def measure(name, kymo_derot, ts, rot_total, win_deg=2.5):
    hp = np.abs(temporal_highpass(kymo_derot, HP_WIN))
    kz = hp - hp.mean(axis=1, keepdims=True)
    n_t, n_a = kz.shape
    bpd = n_a/360.0
    dur = ts[-1]-ts[0]

    # init lock via radon (uses uniform-row assumption; fine for init)
    coarse = radon_search(kz, 30.0, n_a, vel_min=4.5, vel_max=7.5)
    vel0, snr0, _, peak0, ref0 = coarse
    vel, peak, ref = refine_velocity(kz, vel0, 30.0, n_a, window=0.3, step=0.005)
    phase = peak*360.0/n_a - vel*(ref/30.0)  # deg at t=0 (approx, refined below)
    # better phase anchor: use real ts[ref]
    phase = (peak*360.0/n_a) - vel*ts[ref]

    win = int(round(win_deg*bpd))
    argmax_ang = np.zeros(n_t); cent_ang = np.zeros(n_t); ok = np.zeros(n_t, bool)
    for i in range(n_t):
        pred = (phase + vel*ts[i]) % 360.0
        pb = int(round(pred*bpd)) % n_a
        # window
        js = np.arange(pb-win, pb+win+1)
        vals = kz[i, js % n_a].copy()
        vals = np.clip(vals, 0, None)  # ridge is positive after abs/mean-sub
        # argmax
        am = js[int(np.argmax(vals))] % n_a
        # sub-bin parabolic around am
        a0 = kz[i, (am-1) % n_a]; a1 = kz[i, am]; a2 = kz[i, (am+1) % n_a]
        den = a0-2*a1+a2
        amf = am + (0.5*(a0-a2)/den if abs(den) > 1e-9 else 0.0)
        argmax_ang[i] = (amf*360.0/n_a) % 360.0
        # centroid (intensity-weighted) over window
        wsum = vals.sum()
        if wsum > 1e-6:
            c = (js*vals).sum()/wsum
            cent_ang[i] = (c*360.0/n_a) % 360.0
            ok[i] = True
        else:
            cent_ang[i] = pred
        # accept if argmax near prediction
        dev = ((argmax_ang[i]-pred+540) % 360)-180
        ok[i] = ok[i] and abs(dev) < win_deg*0.9

    results = {}
    for label, ang in [("argmax", argmax_ang), ("centroid", cent_ang)]:
        m = ok
        y = unwrap_to_line(ang[m], ts[m], vel, phase)
        s, b, rms, se, res = lsq_slope(ts[m], y)
        results[label] = (s, rms, se, int(m.sum()))
    # robust on centroid
    m = ok
    y = unwrap_to_line(cent_ang[m], ts[m], vel, phase)
    s, b, rms, se, rmask = robust_slope(ts[m], y)
    results["robust"] = (s, rms, se, int(rmask.sum()))

    print(f"\n=== {name} === dur={dur:.1f}s rot={rot_total:+.1f}deg "
          f"lock={vel:+.4f} snr={snr0:.1f} match={ok.mean():.0%}")
    for label in ["argmax", "centroid", "robust"]:
        s, rms, se, n = results[label]
        drift = (s-6.0)/6.0*86400
        drift_se = se/6.0*86400
        print(f"  {label:9s}: slope={s:+.5f} deg/s  drift={drift:+8.1f} +/- {drift_se:5.1f} s/day  RMS={rms:.3f} N={n}")
    return results


if __name__ == "__main__":
    targets = sys.argv[1:] or ["videos/IMG_7720.MOV"]
    for t in targets:
        name = os.path.splitext(os.path.basename(t))[0]
        kymo_derot, ts, rot_total = build_cache(t, name)
        measure(name, kymo_derot, ts, rot_total)
