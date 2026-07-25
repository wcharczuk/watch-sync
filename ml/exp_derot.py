#!/usr/bin/env python3
"""Compare de-rotation methods and their effect on the drift slope.

  12h    : per-frame 12th-harmonic phase (current v5/v6 method)
  xcorr  : sub-bin circular cross-correlation of each frame's full angular
           profile against the median (static) profile -- uses ALL dial
           structure, iteratively refined. Sub-bin shifts applied via Fourier.

For each, de-rotate -> temporal high-pass -> centroid track -> robust slope on
REAL timestamps. Reports drift and residual-vs-angle / vs-time structure.
"""
import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from validate_v5 import radon_search, refine_velocity
from validate_v6 import temporal_highpass, HP_WIN
from exp_measure import build_cache, derotate_12h, unwrap_to_line, lsq_slope, robust_slope


def fourier_shift_rows(kymo, shifts):
    """Roll each row i by -shifts[i] (sub-bin) via the Fourier shift theorem."""
    n_t, n_a = kymo.shape
    k = np.fft.rfftfreq(n_a)            # cycles/sample, 0..0.5
    F = np.fft.rfft(kymo, axis=1)
    phase = np.exp(2j*np.pi*np.outer(shifts, k))  # +shift => roll by -shift
    return np.fft.irfft(F*phase, n=n_a, axis=1)


def derotate_xcorr(kymo_raw, iters=3):
    n_t, n_a = kymo_raw.shape
    prof = kymo_raw - kymo_raw.mean(axis=1, keepdims=True)
    k = np.fft.rfftfreq(n_a)
    ref = np.median(prof, axis=0)
    shifts = np.zeros(n_t)
    for _ in range(iters):
        F_ref = np.fft.rfft(ref)
        F_all = np.fft.rfft(prof, axis=1)
        xc = np.fft.irfft(F_all*np.conj(F_ref), n=n_a, axis=1)  # (n_t, n_a)
        kpk = np.argmax(xc, axis=1)
        new = np.zeros(n_t)
        for i in range(n_t):
            kk = int(kpk[i])
            a0, a1, a2 = xc[i, (kk-1) % n_a], xc[i, kk], xc[i, (kk+1) % n_a]
            den = a0-2*a1+a2
            sub = kk + (0.5*(a0-a2)/den if abs(den) > 1e-9 else 0.0)
            if sub > n_a/2:
                sub -= n_a
            new[i] = sub
        shifts = new
        derot = fourier_shift_rows(prof, shifts)
        ref = np.median(derot, axis=0)
    return shifts


def measure_drift_struct(label, kymo_derot, ts):
    hp = np.abs(temporal_highpass(kymo_derot, HP_WIN))
    kz = hp - hp.mean(axis=1, keepdims=True)
    n_t, n_a = kz.shape
    bpd = n_a/360.0
    coarse = radon_search(kz, 30.0, n_a, vel_min=4.5, vel_max=7.5)
    vel0 = coarse[0]; snr0 = coarse[1]
    vel, peak, ref = refine_velocity(kz, vel0, 30.0, n_a, window=0.3, step=0.005)
    phase = (peak*360.0/n_a) - vel*ts[ref]
    win = int(round(2.5*bpd))
    ang = np.zeros(n_t); ok = np.zeros(n_t, bool)
    for i in range(n_t):
        pred = (phase + vel*ts[i]) % 360.0
        pb = int(round(pred*bpd)) % n_a
        js = np.arange(pb-win, pb+win+1)
        vals = np.clip(kz[i, js % n_a], 0, None)
        wsum = vals.sum()
        if wsum > 1e-6:
            ang[i] = ((js*vals).sum()/wsum*360.0/n_a) % 360.0
            am = js[int(np.argmax(vals))] % n_a
            dev = ((am*360.0/n_a-pred+540) % 360)-180
            ok[i] = abs(dev) < 2.5*0.9
        else:
            ang[i] = pred
    y = unwrap_to_line(ang[ok], ts[ok], vel, phase)
    s, b, rms, se, rmask = robust_slope(ts[ok], y)
    drift = (s-6.0)/6.0*86400; drift_se = se/6.0*86400
    # residual-vs-angle p2p
    res = y - (s*ts[ok]+b)
    aa = ang[ok] % 360.0
    bins = (aa//10).astype(int)
    bm = np.array([res[bins == j].mean() if (bins == j).any() else np.nan for j in range(36)])
    p2p = np.nanmax(bm)-np.nanmin(bm)
    print(f"  {label:7s}: drift={drift:+8.1f} +/- {drift_se:5.1f} s/day  "
          f"slope={s:+.5f} RMS={rms:.3f} N={int(rmask.sum())} match={ok.mean():.0%} "
          f"snr={snr0:.0f}  resid-vs-angle p2p={p2p:.2f}deg")
    return drift


if __name__ == "__main__":
    targets = sys.argv[1:] or ["videos/IMG_7720.MOV", "videos/IMG_7844.MOV"]
    for t in targets:
        name = os.path.splitext(os.path.basename(t))[0]
        kymo_raw, ts, info = build_cache(t, name)
        dur = ts[-1]-ts[0]
        print(f"\n=== {name} === dur={dur:.1f}s ({dur/60:.2f} rev)")
        d12, rb = derotate_12h(kymo_raw)
        rot12 = (rb[-1]-rb[0])*360.0/kymo_raw.shape[1]
        sx = derotate_xcorr(kymo_raw)
        rotx = (sx[-1]-sx[0])*360.0/kymo_raw.shape[1]
        print(f"  total rotation: 12h={rot12:+.2f}deg  xcorr={rotx:+.2f}deg")
        measure_drift_struct("12h", d12, ts)
        dx = fourier_shift_rows(kymo_raw, sx)
        measure_drift_struct("xcorr", dx, ts)
