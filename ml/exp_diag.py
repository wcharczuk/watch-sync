#!/usr/bin/env python3
"""Diagnose the per-frame angle residual structure (uses cached kymographs).

Plots, per video:
  - residual (centroid angle - linear fit) vs TIME
  - residual vs ABSOLUTE second-hand angle  (reveals position-dependent bias:
    a feature-crossing bias shows as a repeatable pattern vs angle)
Saves to diagnostics_exp/<name>_resid.png and prints correlation diagnostics.
"""
import os, sys
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(__file__))
from validate_v5 import radon_search, refine_velocity, N_ANGLES
from validate_v6 import temporal_highpass, HP_WIN
from exp_measure import build_cache, unwrap_to_line, lsq_slope, robust_slope

OUT = os.path.join(os.path.dirname(__file__), "diagnostics_exp")


def centroids(kz, ts, vel, phase, win_deg=2.5):
    n_t, n_a = kz.shape
    bpd = n_a/360.0
    win = int(round(win_deg*bpd))
    ang = np.zeros(n_t); ok = np.zeros(n_t, bool)
    for i in range(n_t):
        pred = (phase + vel*ts[i]) % 360.0
        pb = int(round(pred*bpd)) % n_a
        js = np.arange(pb-win, pb+win+1)
        vals = np.clip(kz[i, js % n_a], 0, None)
        wsum = vals.sum()
        if wsum > 1e-6:
            c = (js*vals).sum()/wsum
            ang[i] = (c*360.0/n_a) % 360.0
            am = js[int(np.argmax(vals))] % n_a
            dev = ((am*360.0/n_a - pred + 540) % 360)-180
            ok[i] = abs(dev) < win_deg*0.9
        else:
            ang[i] = pred
    return ang, ok


def diag(name, kymo_derot, ts, rot_total):
    os.makedirs(OUT, exist_ok=True)
    hp = np.abs(temporal_highpass(kymo_derot, HP_WIN))
    kz = hp - hp.mean(axis=1, keepdims=True)
    n_t, n_a = kz.shape
    coarse = radon_search(kz, 30.0, n_a, vel_min=4.5, vel_max=7.5)
    vel0 = coarse[0]
    vel, peak, ref = refine_velocity(kz, vel0, 30.0, n_a, window=0.3, step=0.005)
    phase = (peak*360.0/n_a) - vel*ts[ref]

    ang, ok = centroids(kz, ts, vel, phase)
    y = unwrap_to_line(ang[ok], ts[ok], vel, phase)
    s, b, rms, se, res = lsq_slope(ts[ok], y)
    abs_angle = (ang[ok]) % 360.0
    tt = ts[ok]

    # residual vs angle: bin into 36 bins of 10deg, mean residual per bin
    bins = (abs_angle // 10).astype(int)
    binmean = np.array([res[bins == k].mean() if (bins == k).any() else 0 for k in range(36)])
    binstd = np.array([res[bins == k].std() if (bins == k).any() else 0 for k in range(36)])
    print(f"\n=== {name} === slope={s:+.5f} drift={(s-6)/6*86400:+.1f} s/day RMS={rms:.3f} rot={rot_total:+.1f}")
    print(f"  residual-vs-angle: peak-to-peak of binned mean = {binmean.max()-binmean.min():.3f} deg "
          f"(std of binned means {binmean.std():.3f})")
    print(f"  residual-vs-time: corr with t = {np.corrcoef(tt, res)[0,1]:+.3f}")

    # plot
    W, Hh = 1400, 760
    img = np.zeros((Hh, W, 3), np.uint8)
    pad = 40
    # top half: residual vs time
    def yscale(r, rmax, y0, h):
        return int(y0 + h/2 - r/rmax*(h/2-10))
    rmax = max(np.abs(res).max(), 2.0)
    h2 = Hh//2 - pad
    cv2.line(img, (pad, pad+h2//2), (W-pad, pad+h2//2), (60,60,60), 1)
    for t, r in zip(tt, res):
        x = int(pad + (t-tt.min())/max(tt.max()-tt.min(),1e-3)*(W-2*pad))
        cv2.circle(img, (x, yscale(r, rmax, pad, h2)), 1, (0,255,0), -1)
    cv2.putText(img, f"{name}: residual(green) vs TIME   slope={s:+.5f} drift={(s-6)/6*86400:+.0f}s/day RMS={rms:.2f} +-{rmax:.1f}deg",
                (pad, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220,220,220), 1)
    # bottom half: residual vs angle (scatter + binned mean)
    y0b = Hh//2 + 10
    cv2.line(img, (pad, y0b+h2//2), (W-pad, y0b+h2//2), (60,60,60), 1)
    for a, r in zip(abs_angle, res):
        x = int(pad + a/360.0*(W-2*pad))
        cv2.circle(img, (x, yscale(r, rmax, y0b, h2)), 1, (0,180,180), -1)
    for k in range(36):
        x = int(pad + (k*10+5)/360.0*(W-2*pad))
        cv2.circle(img, (x, yscale(binmean[k], rmax, y0b, h2)), 3, (0,0,255), -1)
    cv2.putText(img, f"residual vs ABSOLUTE ANGLE (0-360). red=10deg-bin mean. p2p of bin-mean={binmean.max()-binmean.min():.2f}deg",
                (pad, y0b-2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
    cv2.imwrite(os.path.join(OUT, f"{name}_resid.png"), img)


if __name__ == "__main__":
    targets = sys.argv[1:] or ["videos/IMG_7720.MOV", "videos/IMG_7844.MOV"]
    for t in targets:
        name = os.path.splitext(os.path.basename(t))[0]
        kymo_derot, ts, rot_total = build_cache(t, name)
        diag(name, kymo_derot, ts, rot_total)
