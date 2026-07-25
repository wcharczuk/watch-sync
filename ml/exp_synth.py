#!/usr/bin/env python3
"""Ground-truth check: synthesize a kymograph with a known 6.000 deg/s second
hand (+ static markers + optional known camera rotation) and verify the
measurement pipeline recovers the rate. Isolates estimator correctness from
real-world de-rotation/centering difficulty.
"""
import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from validate_v5 import radon_search, refine_velocity
from validate_v6 import temporal_highpass, HP_WIN
from exp_measure import unwrap_to_line, lsq_slope, robust_slope
from exp_derot import derotate_xcorr, fourier_shift_rows
from exp_measure import derotate_12h

N_A = 720


def gauss_ring(n_a, center_deg, amp, width_deg):
    a = np.arange(n_a)*360.0/n_a
    d = ((a - center_deg + 180) % 360) - 180
    return amp*np.exp(-0.5*(d/width_deg)**2)


def synth(true_vel=6.000, dur=20.0, fps=30.0, rot_func=None, noise=2.0,
          drop_prob=0.0, seed=0):
    rng = np.random.default_rng(seed)
    n_t = int(dur*fps)
    ts = np.arange(n_t)/fps
    # simulate dropped frames -> nonuniform ts (like VFR)
    if drop_prob > 0:
        keep = rng.random(n_t) > drop_prob
        keep[0] = True
        ts = ts[keep]
        n_t = len(ts)
    markers = [(k*30.0, 40.0, 3.5) for k in range(12)]      # 12 hour markers
    markers += [(95.0, 60.0, 5.0), (270.0, 30.0, 8.0)]      # logo, date window
    kymo = np.zeros((n_t, N_A), np.float32)
    phase0 = 33.0
    for i, t in enumerate(ts):
        rot = rot_func(t) if rot_func else 0.0
        row = np.full(N_A, 80.0, np.float32)                # dial base
        for c, amp, w in markers:
            row += gauss_ring(N_A, c+rot, amp, w)
        sec = (phase0 + true_vel*t + rot) % 360.0           # second hand (camera frame)
        row += gauss_ring(N_A, sec, 35.0, 1.6)
        row += rng.normal(0, noise, N_A)
        kymo[i] = row
    return kymo, ts


def measure(kymo_derot, ts, label):
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
        else:
            ang[i] = pred
    y = unwrap_to_line(ang[ok], ts[ok], vel, phase)
    s, b, rms, se, rmask = robust_slope(ts[ok], y)
    drift = (s-6.0)/6.0*86400
    print(f"  {label:22s}: slope={s:+.5f} (err {s-6.0:+.5f})  drift={drift:+8.1f} s/day  RMS={rms:.3f} N={int(rmask.sum())}")
    return s


if __name__ == "__main__":
    print("TEST 1: perfect 6.000 deg/s, NO rotation, uniform 30fps")
    k, ts = synth(rot_func=None, drop_prob=0.0)
    measure(k, ts, "no-rot")

    print("\nTEST 2: 6.000 deg/s, NO rotation, with VFR dropped frames + noise")
    k, ts = synth(rot_func=None, drop_prob=0.02, noise=3.0)
    measure(k, ts, "no-rot VFR")

    print("\nTEST 3: 6.000 deg/s + slow linear camera rotation 0.4 deg/s, 65s")
    k, ts = synth(dur=65.0, rot_func=lambda t: 0.4*t)
    d12, _ = derotate_12h(k); measure(d12, ts, "linrot 12h")
    sx = derotate_xcorr(k); measure(fourier_shift_rows(k, sx), ts, "linrot xcorr")
    measure(k, ts, "linrot NONE(ctrl)")

    print("\nTEST 4: 6.000 deg/s + handheld wobble rotation, 65s")
    rotf = lambda t: 5*np.sin(2*np.pi*t/11) + 0.2*t
    k, ts = synth(dur=65.0, rot_func=rotf)
    d12, _ = derotate_12h(k); measure(d12, ts, "wobble 12h")
    sx = derotate_xcorr(k); measure(fourier_shift_rows(k, sx), ts, "wobble xcorr")
