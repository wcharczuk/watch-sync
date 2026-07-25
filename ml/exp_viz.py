#!/usr/bin/env python3
"""Render de-rotated high-pass kymographs (12h vs logpolar) with the tracked
second hand, so we can SEE whether the dial is static and the ridge is straight.
Uses the _lp cache built by exp_logpolar.py."""
import os, sys
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(__file__))
from validate_v6 import temporal_highpass, HP_WIN
from validate_v5 import radon_search, refine_velocity
from exp_measure import unwrap_to_line, lsq_slope, robust_slope, derotate_12h
from exp_derot import fourier_shift_rows

CACHE = os.path.join(os.path.dirname(__file__), "cache")
OUT = os.path.join(os.path.dirname(__file__), "diagnostics_exp")
N_A = 720


def render(name, kymo_derot, ts, tag):
    hp = np.abs(temporal_highpass(kymo_derot, HP_WIN))
    kz = hp - hp.mean(axis=1, keepdims=True)
    n_t, n_a = kz.shape; bpd = n_a/360.0
    coarse = radon_search(kz, 30.0, n_a, vel_min=4.5, vel_max=7.5)
    vel, peak, ref = refine_velocity(kz, coarse[0], 30.0, n_a, window=0.3, step=0.005)
    phase = (peak*360.0/n_a) - vel*ts[ref]
    win = int(round(2.5*bpd))
    ang = np.zeros(n_t); ok = np.zeros(n_t, bool)
    for i in range(n_t):
        pred = (phase+vel*ts[i]) % 360.0
        pb = int(round(pred*bpd)) % n_a
        js = np.arange(pb-win, pb+win+1)
        vals = np.clip(kz[i, js % n_a], 0, None); wsum = vals.sum()
        if wsum > 1e-6:
            ang[i] = ((js*vals).sum()/wsum*360.0/n_a) % 360.0
            am = js[int(np.argmax(vals))] % n_a
            ok[i] = abs(((am*360.0/n_a-pred+540) % 360)-180) < 2.5*0.9
    y = unwrap_to_line(ang[ok], ts[ok], vel, phase)
    s, b, rms, se, rmask = robust_slope(ts[ok], y)
    drift = (s-6.0)/6.0*86400
    img = cv2.normalize(kz, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    idxs = np.where(ok)[0]
    for j, i in enumerate(idxs):
        col = int(round(ang[i]*bpd)) % n_a
        img[i, col] = (0, 255, 0)
    for i in range(n_t):
        a = (phase + s*ts[i]) % 360.0  # use measured slope line
        img[i, int(round(a*bpd)) % n_a] = (0, 0, 255)
    cv2.putText(img, f"{name} [{tag}] drift={drift:+.0f}s/day slope={s:+.4f} snr={coarse[1]:.0f} rms={rms:.2f}",
                (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    path = os.path.join(OUT, f"{name}_{tag}_track.png")
    cv2.imwrite(path, img)
    print(f"  wrote {path}  drift={drift:+.0f} slope={s:+.4f}")


if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else "IMG_7844"
    d = np.load(os.path.join(CACHE, f"{name}_lp.npz"))
    kymo_raw, ts, rot_lp = d["kymo_raw"], d["ts"], d["rot_lp"]
    d12, _ = derotate_12h(kymo_raw)
    render(name, d12, ts, "12h")
    render(name, fourier_shift_rows(kymo_raw, rot_lp), ts, "logpolar")
    # also: NO derotation (raw) to see the apparent ridge + dial
    render(name, kymo_raw, ts, "raw")
