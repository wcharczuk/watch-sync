#!/usr/bin/env python3
"""Center self-calibration via harmonic debiasing.

A center offset (second-hand pivot != assumed center) injects a once-per-rev
sinusoid into the measured angle: dtheta ~= -(ex*cos t + ey*sin t)/R. If we fit
the slope WITHOUT modeling it, that sinusoid leaks into the rate (badly so over
<1 rev). Solution: jointly fit angle(t) = b0 + m*t + harmonics(1st,2nd), so the
cyclic part is absorbed by the harmonics and `m` is the debiased rate.

Compares plain robust slope vs harmonic-debiased slope on cached videos.
"""
import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from validate_v6 import temporal_highpass, HP_WIN
from validate_v5 import radon_search, refine_velocity
from exp_measure import build_cache, unwrap_to_line, derotate_12h


def track(kymo_derot, ts):
    hp = np.abs(temporal_highpass(kymo_derot, HP_WIN))
    kz = hp - hp.mean(axis=1, keepdims=True)
    n_t, n_a = kz.shape; bpd = n_a/360.0
    coarse = radon_search(kz, 30.0, n_a, vel_min=4.5, vel_max=7.5)
    vel, peak, ref = refine_velocity(kz, coarse[0], 30.0, n_a, window=0.3, step=0.005)
    phase = (peak*360.0/n_a) - vel*ts[ref]
    win = int(round(2.5*bpd))
    ang = np.zeros(n_t); ok = np.zeros(n_t, bool); pred_arr = np.zeros(n_t)
    for i in range(n_t):
        pred = (phase+vel*ts[i]) % 360.0
        pred_arr[i] = phase+vel*ts[i]
        pb = int(round(pred*bpd)) % n_a
        js = np.arange(pb-win, pb+win+1)
        vals = np.clip(kz[i, js % n_a], 0, None); wsum = vals.sum()
        if wsum > 1e-6:
            ang[i] = ((js*vals).sum()/wsum*360.0/n_a) % 360.0
            am = js[int(np.argmax(vals))] % n_a
            ok[i] = abs(((am*360.0/n_a-pred+540) % 360)-180) < 2.5*0.9
        else:
            ang[i] = pred
    y = unwrap_to_line(ang[ok], ts[ok], vel, phase)
    return ts[ok], y, pred_arr[ok], coarse[1]


def fit_plain(t, y, iters=5, k=2.5):
    mask = np.ones(len(t), bool); m = se = rms = 0
    for _ in range(iters):
        tt, yy = t[mask], y[mask]
        A = np.c_[np.ones_like(tt), tt]
        coef, *_ = np.linalg.lstsq(A, yy, rcond=None)
        m = coef[1]
        res = y - (coef[0]+m*t); rms = np.sqrt((res[mask]**2).mean())
        nm = np.abs(res) <= k*max(rms, 1e-6)
        if nm.sum() == mask.sum():
            break
        mask = nm
    var_t = ((t[mask]-t[mask].mean())**2).sum()
    se = rms/np.sqrt(var_t)
    return m, se, rms, int(mask.sum())


def fit_harmonic(t, y, pred_rad, nharm=2, iters=5, k=2.5):
    """Joint fit: y = b0 + m*t + sum_k [a_k cos(k*pred) + b_k sin(k*pred)]."""
    pr = np.radians(pred_rad)
    cols = [np.ones_like(t), t]
    for h in range(1, nharm+1):
        cols += [np.cos(h*pr), np.sin(h*pr)]
    A = np.column_stack(cols)
    mask = np.ones(len(t), bool); m = se = rms = 0
    for _ in range(iters):
        coef, *_ = np.linalg.lstsq(A[mask], y[mask], rcond=None)
        m = coef[1]
        res = y - A@coef; rms = np.sqrt((res[mask]**2).mean())
        nm = np.abs(res) <= k*max(rms, 1e-6)
        if nm.sum() == mask.sum():
            break
        mask = nm
    # SE of slope from the joint fit
    AtA_inv = np.linalg.pinv(A[mask].T@A[mask])
    se = np.sqrt(max(AtA_inv[1, 1], 0))*rms
    # recovered center offset amplitude (deg) from 1st harmonic
    amp1 = np.hypot(coef[2], coef[3])
    return m, se, rms, int(mask.sum()), amp1


def report(name, kymo_derot, ts, tag):
    t, y, pred, snr = track(kymo_derot, ts)
    m0, se0, rms0, n0 = fit_plain(t, y)
    m1, se1, rms1, n1, amp1 = fit_harmonic(t, y, pred, nharm=2)
    d0 = (m0-6)/6*86400; d0se = se0/6*86400
    d1 = (m1-6)/6*86400; d1se = se1/6*86400
    print(f"  {tag:10s} snr={snr:3.0f}  plain: {d0:+8.1f}+/-{d0se:4.1f} s/day (rms{rms0:.2f})   "
          f"harmonic: {d1:+8.1f}+/-{d1se:4.1f} s/day (rms{rms1:.2f}, center-sinusoid {amp1:.2f}deg)")


if __name__ == "__main__":
    targets = sys.argv[1:] or ["videos/IMG_7844.MOV", "videos/IMG_7704.MOV", "videos/IMG_7720.MOV"]
    for tpath in targets:
        name = os.path.splitext(os.path.basename(tpath))[0]
        kymo_raw, ts, info = build_cache(tpath, name)
        dur = ts[-1]-ts[0]
        print(f"\n=== {name} === dur={dur:.1f}s ({dur/60:.2f} rev)")
        report(name, kymo_raw, ts, "raw")
        d12, _ = derotate_12h(kymo_raw)
        report(name, d12, ts, "12h")
