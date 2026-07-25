#!/usr/bin/env python3
"""Hunt the reproducible ~-50 s/day systematic with synthetic ground truth.

Build a kymograph directly (perfect 6.000 deg/s, no rotation, perfect center) and
run the EXACT v8 measurement. If it returns ~-50 s/day, the bias is in the
estimator. Suspects probed:
  - asymmetric second hand (bright tip + small counterweight on the far side)
    -> the temporal high-pass ridge is asymmetric -> centroid pulled in the
       sweep direction -> a constant rate bias that does NOT cancel over revs.
  - high-pass EDGE effects near clip ends -> slope leverage. (end-trim test)
  - centroid vs parabolic-argmax angle estimator.
"""
import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from validate_v7 import temporal_highpass, unwrap_to_line, fit_plain, N_A, HP_WIN
from validate_v5 import radon_search, refine_velocity

A = np.arange(N_A)*360.0/N_A


def ring(center_deg, amp, width_deg):
    d = ((A - center_deg + 180) % 360) - 180
    return amp*np.exp(-0.5*(d/width_deg)**2)


def synth(true_vel=6.0, dur=120.0, fps=29.97, counterweight=0.0, noise=2.0,
          tip_w=1.6, seed=1):
    rng = np.random.default_rng(seed)
    n = int(dur*fps); ts = np.arange(n)/fps
    markers = sum(ring(k*30.0, 40.0, 3.5) for k in range(12))
    markers = markers + ring(95, 60, 5) + ring(270, 30, 8)
    kymo = np.empty((n, N_A), np.float32)
    phase0 = 33.0
    for i, t in enumerate(ts):
        sec = (phase0 + true_vel*t) % 360.0
        row = 80.0 + markers + ring(sec, 35.0, tip_w)
        if counterweight > 0:
            row = row + ring((sec+180) % 360, counterweight, 2.5)  # tail on far side
        kymo[i] = row + rng.normal(0, noise, N_A)
    return kymo, ts


def measure(kymo, ts, end_trim=0, use_centroid=True):
    hp = np.abs(temporal_highpass(kymo)); kz = hp - hp.mean(axis=1, keepdims=True)
    n_t, n_a = kz.shape; bpd = n_a/360.0
    coarse = radon_search(kz, 30.0, n_a, vel_min=4.5, vel_max=7.5)
    vel, peak, ref = refine_velocity(kz, coarse[0], 30.0, n_a, window=0.3, step=0.005)
    phase = (peak*360.0/n_a) - vel*ts[ref]
    win = int(round(2.5*bpd))
    ang = np.zeros(n_t); ok = np.zeros(n_t, bool)
    for i in range(n_t):
        p = phase+vel*ts[i]; pb = int(round((p % 360.0)*bpd)) % n_a
        js = np.arange(pb-win, pb+win+1)
        vals = np.clip(kz[i, js % n_a], 0, None); wsum = vals.sum()
        if use_centroid:
            ang[i] = ((js*vals).sum()/wsum*360.0/n_a) % 360.0 if wsum > 1e-6 else (p % 360.0)
        else:
            am = js[int(np.argmax(vals))] % n_a
            a0, a1, a2 = kz[i, (am-1) % n_a], kz[i, am], kz[i, (am+1) % n_a]
            den = a0-2*a1+a2
            ang[i] = (((am + (0.5*(a0-a2)/den if abs(den) > 1e-9 else 0))*360.0/n_a) % 360.0)
        ok[i] = wsum > 1e-6
    if end_trim > 0:
        ok[:end_trim] = False; ok[-end_trim:] = False
    y = unwrap_to_line(ang[ok], ts[ok], vel, phase)
    slope, se, rms, nfit = fit_plain(ts[ok], y)
    return (slope-6.0)/6.0*86400, rms


def run(label, **kw):
    meas = {k: kw.pop(k) for k in ["end_trim", "use_centroid"] if k in kw}
    kymo, ts = synth(**kw)
    d, rms = measure(kymo, ts, **meas)
    print(f"  {label:42s} drift={d:+8.1f} s/day  (rms {rms:.3f})")


if __name__ == "__main__":
    print("Synthetic 2-rev (120s), true rate = 0 s/day exactly. Bias = what we read:\n")
    run("symmetric hand, centroid")
    run("symmetric hand, centroid, end-trim", end_trim=HP_WIN)
    run("symmetric hand, parabolic-argmax", use_centroid=False)
    print()
    run("asymmetric hand (cw=12), centroid", counterweight=12.0)
    run("asymmetric hand (cw=12), parabolic-argmax", counterweight=12.0, use_centroid=False)
    run("asymmetric hand (cw=20), centroid", counterweight=20.0)
    run("asymmetric hand (cw=20), parabolic-argmax", counterweight=20.0, use_centroid=False)
    print()
    run("low noise, symmetric, centroid", noise=0.5)
    run("hi noise, symmetric, centroid", noise=5.0)
