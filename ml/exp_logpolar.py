#!/usr/bin/env python3
"""De-rotation via 2D-polar phase correlation.

Rotation = pure shift along the ANGLE axis of the polar image. Correlating the
full 2D polar map (angle x radius) against the median (static, second-hand-free)
reference uses the dial's ASYMMETRIC features (text, date window, logo) to break
the 30-degree marker-symmetry ambiguity, and is robust + sub-bin.

Caches per-frame rotation so re-runs are fast. Compares to 12h and reports drift.
"""
import os, sys
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(__file__))
from validate_v5 import detect_watch_face, stabilize_phase_corr, apply_shift
from validate_v6 import temporal_highpass, HP_WIN, WARMUP
from validate_v5 import radon_search, refine_velocity
from exp_measure import extract_with_ts, unwrap_to_line, lsq_slope, robust_slope, derotate_12h
from exp_derot import fourier_shift_rows

CACHE = os.path.join(os.path.dirname(__file__), "cache")
N_A = 720


def polar2d_lut(side, cx, cy, r_in, r_out, n_a, n_r):
    angles = np.arange(n_a)*(2*np.pi/n_a)
    sin_a, cos_a = np.sin(angles), -np.cos(angles)
    radii = np.linspace(r_in, r_out, n_r)
    px = cx + np.outer(sin_a, radii); py = cy + np.outer(cos_a, radii)
    ix = np.clip(np.floor(px).astype(np.int32), 0, side-2)
    iy = np.clip(np.floor(py).astype(np.int32), 0, side-2)
    fx = (px-ix).astype(np.float32); fy = (py-iy).astype(np.float32)
    return ix, iy, fx, fy


def sample2d(img, lut):
    ix, iy, fx, fy = lut
    f = img.astype(np.float32)
    return (f[iy, ix]*(1-fx)*(1-fy) + f[iy, ix+1]*fx*(1-fy) +
            f[iy+1, ix]*(1-fx)*fy + f[iy+1, ix+1]*fx*fy)   # (n_a, n_r)


def build(video_path, name):
    os.makedirs(CACHE, exist_ok=True)
    p = os.path.join(CACHE, f"{name}_lp.npz")
    if os.path.exists(p):
        d = np.load(p)
        return d["kymo_raw"], d["ts"], d["rot_lp"]
    print(f"  building log-polar cache for {name} ...")
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
    # rotation-tracking polar map: WIDE radial span incl. text/date/logo
    n_r = 36
    lut2d = polar2d_lut(side, cl, cl, cl*0.20, cl*0.85, N_A, n_r)
    P = np.stack([sample2d(s, lut2d) for s in stab])[WARMUP:]   # (n_t, n_a, n_r)
    ts = ts[WARMUP:]
    # second-hand kymograph: outer ring mean (as before)
    from validate_v5 import make_polar_lut, sample_polar
    lut1d = make_polar_lut(side, cl, cl, cl*0.50, cl*0.85, N_A, 25)
    kymo_raw = np.stack([sample_polar(s.astype(np.float32), lut1d) for s in stab])[WARMUP:]

    rot_lp = derotate_polar(P)
    np.savez_compressed(p, kymo_raw=kymo_raw.astype(np.float32), ts=ts,
                        rot_lp=rot_lp.astype(np.float32))
    return kymo_raw, ts, rot_lp


def derotate_polar(P, iters=3):
    """Per-frame rotation (in BINS) via 2D-polar phase correlation vs median ref.
    P: (n_t, n_a, n_r). Returns shift bins to de-rotate (roll by -shift)."""
    n_t, n_a, n_r = P.shape
    Pz = P - P.mean(axis=1, keepdims=True)         # remove per-radius DC
    shifts = np.zeros(n_t)
    ref = np.median(Pz, axis=0)                    # (n_a, n_r) static reference
    for _ in range(iters):
        Fref = np.fft.rfft(ref, axis=0)            # (a, r)
        new = np.zeros(n_t)
        for i in range(n_t):
            Fi = np.fft.rfft(Pz[i], axis=0)
            xc = np.fft.irfft(np.sum(Fi*np.conj(Fref), axis=1), n=n_a)  # (n_a,)
            k = int(np.argmax(xc))
            a0, a1, a2 = xc[(k-1) % n_a], xc[k], xc[(k+1) % n_a]
            den = a0-2*a1+a2
            sub = k + (0.5*(a0-a2)/den if abs(den) > 1e-9 else 0.0)
            if sub > n_a/2:
                sub -= n_a
            new[i] = sub
        shifts = new
        # rebuild ref from de-rotated stack (sub-bin shift each radius column)
        derot = np.stack([fourier_shift_rows(Pz[i].T, np.full(n_r, shifts[i])).T
                          for i in range(n_t)])
        ref = np.median(derot, axis=0)
    return shifts


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
    drift = (s-6.0)/6.0*86400; drift_se = se/6.0*86400
    print(f"  {label:12s}: drift={drift:+8.1f} +/- {drift_se:5.1f} s/day  "
          f"slope={s:+.5f} RMS={rms:.3f} N={int(rmask.sum())} match={ok.mean():.0%} snr={coarse[1]:.0f}")
    return drift


if __name__ == "__main__":
    targets = sys.argv[1:] or ["videos/IMG_7844.MOV", "videos/IMG_7704.MOV"]
    for t in targets:
        name = os.path.splitext(os.path.basename(t))[0]
        kymo_raw, ts, rot_lp = build(t, name)
        dur = ts[-1]-ts[0]
        rot_total = (rot_lp[-1]-rot_lp[0])*360.0/N_A
        print(f"\n=== {name} === dur={dur:.1f}s ({dur/60:.2f} rev)  logpolar rot={rot_total:+.2f}deg")
        d12, rb = derotate_12h(kymo_raw)
        print(f"  (12h rot={ (rb[-1]-rb[0])*360.0/N_A:+.2f}deg)")
        measure(d12, ts, "12h")
        dlp = fourier_shift_rows(kymo_raw, rot_lp)
        measure(dlp, ts, "logpolar")
