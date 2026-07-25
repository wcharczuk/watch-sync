#!/usr/bin/env python3
"""Discriminate the -51 s/day systematic: measure the cached abs-center kymograph
over different time sub-windows.
  - CONSTANT across windows -> timebase / true rate (needs external ground truth)
  - VARIES with window      -> angle-periodic effect (center offset / crossings)
"""
import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from validate_v7 import temporal_highpass, unwrap_to_line, fit_plain, N_A, WARMUP
from validate_v5 import radon_search, refine_velocity

CACHE = os.path.join(os.path.dirname(__file__), "cache")


def drift_on(kymo, ts):
    hp = np.abs(temporal_highpass(kymo)); kz = hp - hp.mean(axis=1, keepdims=True)
    n_t, n_a = kz.shape; bpd = n_a/360.0
    coarse = radon_search(kz, 30.0, n_a, vel_min=4.5, vel_max=7.5)
    vel, peak, ref = refine_velocity(kz, coarse[0], 30.0, n_a, window=0.3, step=0.005)
    phase = (peak*360.0/n_a) - vel*ts[ref]
    win = int(round(2.5*bpd))
    ang = np.zeros(n_t); conf = np.zeros(n_t)
    for i in range(n_t):
        p = phase+vel*ts[i]; pb = int(round((p % 360.0)*bpd)) % n_a
        js = np.arange(pb-win, pb+win+1)
        vals = np.clip(kz[i, js % n_a], 0, None); wsum = vals.sum()
        row = kz[i]; mad = np.median(np.abs(row-np.median(row)))+1e-6
        conf[i] = vals.max()/mad
        ang[i] = ((js*vals).sum()/wsum*360.0/n_a) % 360.0 if wsum > 1e-6 else (p % 360.0)
    m = conf >= np.percentile(conf, 50)
    y = unwrap_to_line(ang[m], ts[m], vel, phase)
    slope, se, rms, nfit = fit_plain(ts[m], y)
    return (slope-6.0)/6.0*86400, se/6*86400, rms


def main(name):
    d = np.load(os.path.join(CACHE, f"{name}_abs_kymo.npz"))
    kymo = d["kymo"][WARMUP:]; ts = d["ts"][WARMUP:]
    t0 = ts[0]; dur = ts[-1]-t0
    print(f"\n=== {name} === dur={dur:.1f}s ({dur/60:.2f} rev)")
    # full, halves, and sliding 60s (1-rev) windows
    segments = [("full", t0, ts[-1]),
                ("first half", t0, t0+dur/2),
                ("second half", t0+dur/2, ts[-1]),
                ("rev1 [0-60s]", t0, t0+60),
                ("rev2 [60-120s]", t0+60, t0+120)]
    for lbl, a, b in segments:
        sel = (ts >= a) & (ts <= b)
        if sel.sum() < 200:
            continue
        dr, se, rms = drift_on(kymo[sel], ts[sel])
        print(f"  {lbl:16s} N={sel.sum():4d}  drift={dr:+8.1f} +/- {se:4.1f} s/day (rms {rms:.2f})")


if __name__ == "__main__":
    for name in (sys.argv[1:] or ["IMG_7855", "IMG_7854"]):
        try:
            main(name)
        except FileNotFoundError:
            print(f"  no cache for {name}")
