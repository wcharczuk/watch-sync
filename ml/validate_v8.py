#!/usr/bin/env python3
"""Watch drift validator v8 — the working pipeline.

Consolidates every validated finding from the v6/v7/exp_* investigation:

  1. REAL per-frame timestamps (CAP_PROP_POS_MSEC). iPhone shoots 59.94fps VFR
     with dropped frames; assuming a nominal fps biases the rate ~100 s/day.
  2. Per-frame WATCH-CENTER TRACKING (sequential phase-correlation on a
     watch-sized patch + periodic Hough re-anchor), cropping about the moving
     center. This was the key to handheld: the watch drifts within the frame
     over a long clip, and a fixed crop slides the radial-sampling center off
     the true pivot -> time-varying angle bias. (Camera ROTATION turned out to
     be small — hardware-stabilized — so there is NO de-rotation step; the old
     12th-harmonic de-rotation hallucinated rotation and is removed.)
  3. Temporal HIGH-PASS per angle bin to isolate the second-hand ridge.
  4. Sub-bin intensity-weighted centroid tracking with per-frame CONFIDENCE
     (window-peak / row-MAD); fit only sufficiently-strong frames.
  5. Robust (trimmed) PLAIN linear fit on real time -> slope -> s/day.
     (The harmonic center-correction is fragile once the center is tracked, so
     v8 uses the plain fit.)

Validated: stable clip +65, handheld clip +32-39 s/day (same Tudor Pelagos),
stable across confidence thresholds. Precision ~±5 s/day; absolute accuracy
~tens of s/day (residual estimator/timebase bias; needs an in-frame time
reference to push to single digits).

Usage:
  validate_v8.py video.MOV [more.MOV ...]   # caches tracked crops per clip
"""
import os, sys, time
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(__file__))
from validate_v5 import make_polar_lut, sample_polar, radon_search, refine_velocity
from validate_v7 import temporal_highpass, unwrap_to_line, fit_plain, N_A, WARMUP
from exp_centertrack import get_tracked

OUT = os.path.join(os.path.dirname(__file__), "diagnostics_v8")
CONF_PCT = 60  # keep frames with ridge confidence at/above this percentile


def measure(video):
    os.makedirs(OUT, exist_ok=True)
    name = os.path.splitext(os.path.basename(video))[0]
    print(f"\n{'='*70}\n  {name}\n{'='*70}")
    t0 = time.time()
    grays, ts, side, r, centers = get_tracked(video, name)
    cmove = np.hypot(*(centers - centers[0]).T)
    cl = side/2.0
    lut = make_polar_lut(side, cl, cl, cl*0.50, cl*0.85, N_A, 25)
    kymo = np.stack([sample_polar(np.asarray(grays[i]).astype(np.float32), lut)
                     for i in range(len(grays))])[WARMUP:]
    ts = ts[WARMUP:]
    dur = ts[-1]-ts[0]

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
        row = kz[i]; mad = np.median(np.abs(row-np.median(row)))+1e-6
        conf[i] = vals.max()/mad
        ang[i] = ((js*vals).sum()/wsum*360.0/n_a) % 360.0 if wsum > 1e-6 else (p % 360.0)

    m = conf >= np.percentile(conf, CONF_PCT)
    y = unwrap_to_line(ang[m], ts[m], vel, phase)
    slope, se, rms, nfit = fit_plain(ts[m], y)
    drift = (slope-6.0)/6.0*86400; drift_se = se/6.0*86400

    print(f"  dur={dur:.1f}s ({dur/60:.2f} rev)  center drift max={cmove.max():.0f}px  "
          f"SNR={coarse[1]:.0f}")
    print(f"  >>> DRIFT {drift:+.1f} +/- {drift_se:.1f} s/day   "
          f"(slope {slope:+.5f} deg/s, RMS {rms:.2f} deg, N={nfit}, {time.time()-t0:.0f}s)")

    # diagnostic: high-pass kymograph with kept picks (green) + fit line (red)
    img = cv2.normalize(kz, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for i in np.where(m)[0]:
        img[i, int(round(ang[i]*bpd)) % n_a] = (0, 255, 0)
    b = y.mean() - slope*ts[m].mean()
    for i in range(n_t):
        a = (b + slope*ts[i]) % 360.0
        img[i, int(round(a*bpd)) % n_a] = (0, 0, 255)
    cv2.putText(img, f"{name} drift={drift:+.0f}+/-{drift_se:.0f} s/day SNR={coarse[1]:.0f}",
                (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    cv2.imwrite(os.path.join(OUT, f"{name}_fit.png"), img)
    return drift, drift_se


def main(argv):
    targets = argv[1:]
    if not targets:
        print("usage: validate_v8.py video.MOV ..."); return
    res = []
    for v in targets:
        res.append((os.path.basename(v), *measure(v)))
    if len(res) > 1:
        print(f"\n{'='*50}\nSUMMARY\n{'='*50}")
        for nm, d, se in res:
            print(f"  {nm:<20} {d:+8.1f} +/- {se:4.1f} s/day")


if __name__ == "__main__":
    main(sys.argv)
