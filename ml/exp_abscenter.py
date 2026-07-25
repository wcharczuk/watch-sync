#!/usr/bin/env python3
"""Non-accumulating absolute center estimation.

Sequential phase-corr center tracking ACCUMULATES drift (invented 66px on the
propped stable clip -> swung it +65 to -56). Fix: a GENEROUS fixed crop (so the
watch never exits even with real drift) + per-frame ABSOLUTE center via Hough,
temporally SMOOTHED (real center motion is slow; Hough noise is high-freq). No
accumulation. Sample the kymograph about each frame's smoothed center.

Goal: both clips of the same watch finally AGREE.
"""
import os, sys, time
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(__file__))
from validate_v5 import detect_watch_face, make_polar_lut, sample_polar, radon_search, refine_velocity
from validate_v7 import temporal_highpass, unwrap_to_line, fit_plain, N_A, WARMUP

CACHE = os.path.join(os.path.dirname(__file__), "cache")


def load_generous(video_path, target_fps=30, max_frames=8000, max_h=1080, margin=2.8):
    cap = cv2.VideoCapture(video_path)
    native = cap.get(cv2.CAP_PROP_FPS) or 60.0
    skip = max(1, int(round(native/target_fps)))
    crops, ts = [], []
    box = None; r = None
    idx = 0
    while len(crops) < max_frames:
        ok, fr = cap.read()
        if not ok:
            break
        t = cap.get(cv2.CAP_PROP_POS_MSEC)/1000.0
        if idx % skip == 0:
            h, w = fr.shape[:2]
            if h > max_h:
                s = max_h/h; fr = cv2.resize(fr, (int(w*s), int(h*s)), interpolation=cv2.INTER_AREA)
            g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
            H, W = g.shape
            if box is None:
                det = detect_watch_face(fr)
                if det is None:
                    idx += 1; continue
                cx, cy, r = det
                side = int(r*margin)
                x0 = max(0, min(W-side, int(round(cx-side/2))))
                y0 = max(0, min(H-side, int(round(cy-side/2))))
                side = min(side, W-x0, H-y0)
                box = (x0, y0, side)
            x0, y0, side = box
            crops.append(g[y0:y0+side, x0:x0+side].copy()); ts.append(t)
        idx += 1
    cap.release()
    return np.stack(crops), np.array(ts), side, float(r)


def abs_centers(crops, r, side):
    """Per-frame absolute center via Hough on each crop; nan-fill + smooth."""
    n = len(crops)
    cx = np.full(n, np.nan); cy = np.full(n, np.nan)
    for i in range(n):
        det = detect_watch_face(cv2.cvtColor(np.asarray(crops[i]), cv2.COLOR_GRAY2BGR))
        if det is not None:
            dcx, dcy, dr = det
            # accept only plausible radius + center near crop middle
            if 0.7*r < dr < 1.3*r and abs(dcx-side/2) < 0.4*side and abs(dcy-side/2) < 0.4*side:
                cx[i], cy[i] = dcx, dcy
    # interpolate gaps
    good = ~np.isnan(cx)
    if good.sum() < n*0.3:
        # fallback: constant center
        cx[:] = np.nanmedian(cx) if good.any() else side/2
        cy[:] = np.nanmedian(cy) if good.any() else side/2
    else:
        xi = np.arange(n)
        cx = np.interp(xi, xi[good], cx[good])
        cy = np.interp(xi, xi[good], cy[good])
    # smooth: median (despike) then moving average (real motion is slow)
    def smooth(a, med=15, avg=31):
        from scipy.ndimage import median_filter
        a = median_filter(a, med, mode="nearest")
        k = np.ones(avg)/avg
        return np.convolve(np.pad(a, avg//2, mode="edge"), k, mode="valid")[:len(a)]
    try:
        cx, cy = smooth(cx), smooth(cy)
    except Exception:
        kx = np.ones(31)/31
        cx = np.convolve(np.pad(cx, 15, mode="edge"), kx, mode="valid")[:n]
        cy = np.convolve(np.pad(cy, 15, mode="edge"), kx, mode="valid")[:n]
    return cx, cy, int(good.sum())


def measure(video):
    name = os.path.splitext(os.path.basename(video))[0]
    print(f"\n=== {name} ===")
    t0 = time.time()
    kc = os.path.join(CACHE, f"{name}_abs_kymo.npz")
    if os.path.exists(kc):
        d = np.load(kc); kymo = d["kymo"]; ts = d["ts"]; cmove = float(d["cmove"]); ngood = int(d["ngood"])
    else:
        crops, ts, side, r = load_generous(video)
        cx, cy, ngood = abs_centers(crops, r, side)
        cmove = float(np.hypot(cx-cx.mean(), cy-cy.mean()).max())
        kymo = np.zeros((len(crops), N_A), np.float32)
        for i in range(len(crops)):
            lut = make_polar_lut(side, cx[i], cy[i], r*0.50, r*0.85, N_A, 25)
            kymo[i] = sample_polar(np.asarray(crops[i]).astype(np.float32), lut)
        del crops
        np.savez_compressed(kc, kymo=kymo, ts=ts, cmove=cmove, ngood=ngood)
    kymo = kymo[WARMUP:]; ts = ts[WARMUP:]
    dur = ts[-1]-ts[0]
    hp = np.abs(temporal_highpass(kymo)); kz = hp - hp.mean(axis=1, keepdims=True)
    n_t, n_a = kz.shape; bpd = n_a/360.0
    coarse = radon_search(kz, 30.0, n_a, vel_min=4.5, vel_max=7.5)
    vel, peak, ref = refine_velocity(kz, coarse[0], 30.0, n_a, window=0.3, step=0.005)
    phase = (peak*360.0/n_a) - vel*ts[ref]
    win = int(round(2.5*bpd))
    ang = np.zeros(n_t); conf = np.zeros(n_t)
    for i in range(n_t):
        p = phase+vel*ts[i]
        pb = int(round((p % 360.0)*bpd)) % n_a
        js = np.arange(pb-win, pb+win+1)
        vals = np.clip(kz[i, js % n_a], 0, None); wsum = vals.sum()
        row = kz[i]; mad = np.median(np.abs(row-np.median(row)))+1e-6
        conf[i] = vals.max()/mad
        ang[i] = ((js*vals).sum()/wsum*360.0/n_a) % 360.0 if wsum > 1e-6 else (p % 360.0)
    print(f"  dur={dur:.1f}s SNR={coarse[1]:.0f} center-move(abs,smoothed)={cmove:.0f}px good-detect={ngood} ({time.time()-t0:.0f}s)")
    res = {}
    for pct in [0, 50, 70]:
        m = conf >= np.percentile(conf, pct)
        if m.sum() < 60:
            continue
        y = unwrap_to_line(ang[m], ts[m], vel, phase)
        slope, se, rms, nfit = fit_plain(ts[m], y)
        d = (slope-6)/6*86400
        res[pct] = d
        print(f"  conf>=p{pct:<2d} N={m.sum():4d}  DRIFT {d:+8.1f} +/- {se/6*86400:4.1f} s/day (rms {rms:.2f})")
    return res.get(50)


if __name__ == "__main__":
    targets = sys.argv[1:] or ["videos/IMG_7854.MOV", "videos/IMG_7855.MOV"]
    out = []
    for v in targets:
        out.append((os.path.basename(v), measure(v)))
    print("\nSUMMARY (conf>=p50):")
    for nm, d in out:
        print(f"  {nm:<20} {('%+.1f s/day'%d) if d is not None else 'n/a'}")
