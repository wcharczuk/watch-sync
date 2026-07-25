#!/usr/bin/env python3
"""Per-frame confidence gating. The handheld clip has a strong second-hand
ridge in part of the clip and a degraded (blur/focus) section elsewhere. Fit the
rate only on frames where the ridge is genuinely strong. Rotation is small
(hardware-stabilized), so we use the RAW kymograph (no harmful 12h de-rotation).

Reports drift vs confidence threshold for the handheld and stable clips.
"""
import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from validate_v5 import stabilize_phase_corr, apply_shift, make_polar_lut, sample_polar
from validate_v7 import temporal_highpass, fit_harmonic, fit_plain, unwrap_to_line, N_A, WARMUP
from validate_v5 import radon_search, refine_velocity
from exp_handheld import get_crops


def track_conf(kymo, ts):
    """Track second-hand centroid AND a per-frame confidence (ridge peak over
    local MAD within the search window)."""
    hp = np.abs(temporal_highpass(kymo))
    kz = hp - hp.mean(axis=1, keepdims=True)
    n_t, n_a = kz.shape; bpd = n_a/360.0
    coarse = radon_search(kz, 30.0, n_a, vel_min=4.5, vel_max=7.5)
    vel, peak, ref = refine_velocity(kz, coarse[0], 30.0, n_a, window=0.3, step=0.005)
    phase = (peak*360.0/n_a) - vel*ts[ref]
    win = int(round(2.5*bpd))
    ang = np.zeros(n_t); conf = np.zeros(n_t); pred = np.zeros(n_t)
    for i in range(n_t):
        p = phase+vel*ts[i]; pred[i] = p
        pb = int(round((p % 360.0)*bpd)) % n_a
        js = np.arange(pb-win, pb+win+1)
        vals = np.clip(kz[i, js % n_a], 0, None); wsum = vals.sum()
        # ridge strength: window peak vs whole-row MAD
        row = kz[i]; mad = np.median(np.abs(row-np.median(row)))+1e-6
        conf[i] = vals.max()/mad
        ang[i] = ((js*vals).sum()/wsum*360.0/n_a) % 360.0 if wsum > 1e-6 else (p % 360.0)
    return ts, ang, conf, pred, vel, phase, coarse[1]


def main(video, label):
    name = os.path.splitext(os.path.basename(video))[0]
    grays, ts, side, r = get_crops(video, name)
    cl = side/2.0
    glist = [np.asarray(grays[i]) for i in range(len(grays))]
    shifts = stabilize_phase_corr(glist)
    stab = [apply_shift(g, dx, dy) for g, (dx, dy) in zip(glist, shifts)]
    lut = make_polar_lut(side, cl, cl, cl*0.50, cl*0.85, N_A, 25)
    kymo = np.stack([sample_polar(s.astype(np.float32), lut) for s in stab])[WARMUP:]
    ts = ts[WARMUP:]
    tt, ang, conf, pred, vel, phase, snr0 = track_conf(kymo, ts)
    print(f"\n=== {name} ({label}) === snr={snr0:.0f}  conf: med={np.median(conf):.1f} "
          f"p25={np.percentile(conf,25):.1f} p75={np.percentile(conf,75):.1f}")
    for pct in [0, 40, 55, 70, 80, 90]:
        thr = np.percentile(conf, pct)
        m = conf >= thr
        # require >=1 rev of coverage among kept
        ang_span = np.ptp(((ang[m]-pred[m]+540) % 360))  # not exact; just info
        y = unwrap_to_line(ang[m], tt[m], vel, phase)
        if m.sum() < 60:
            continue
        mh, seh, rmsh, nh, amp1 = fit_harmonic(tt[m], y, pred[m])
        mp, sep, rmsp, _ = fit_plain(tt[m], y)
        kept_dur = tt[m].max()-tt[m].min()
        print(f"  conf>=p{pct:<2d}({thr:5.1f}) N={m.sum():4d} dur={kept_dur:5.1f}s  "
              f"plain {(mp-6)/6*86400:+8.1f}  harmonic {(mh-6)/6*86400:+8.1f}+/-{seh/6*86400:4.1f} (rms{rmsh:.2f})")


if __name__ == "__main__":
    main("videos/IMG_7854.MOV", "handheld")
    main("videos/IMG_7855.MOV", "stable")
