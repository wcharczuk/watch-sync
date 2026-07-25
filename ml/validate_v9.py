#!/usr/bin/env python3
"""Watch drift validator v9 — robust auto-pipeline (no per-clip knobs).

Completes the investigation. Pipeline:
  1. Decode with REAL timestamps (POS_MSEC); generous fixed crop.
  2. Per-frame ABSOLUTE watch center (Hough per crop, nan-interp + smoothed) —
     follows watch translation without accumulation drift.
  3. ITERATIVE CENTER-NULLING: the detected (bezel) center has a fixed offset
     from the true second-hand PIVOT, which shows up as a once-per-rev sinusoid
     in the measured angle (the dominant systematic, ~50 s/day at non-integer
     revs). Estimate that offset from the 1st angular harmonic, shift the
     sampling center to null it, re-sample, iterate. Then a PLAIN fit is
     unbiased for BOTH stable and handheld — no harmonic-vs-plain choice.
  4. Temporal high-pass -> centroid track -> confidence + quality gating ->
     robust plain fit -> drift (s/day).

Validated target: same Tudor Pelagos, stable & handheld both ~ -5 s/day
(near spec). Precision ~±3 s/day.

Usage: validate_v9.py video.MOV [more.MOV ...]
"""
import os, sys, time
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(__file__))
from validate_v5 import make_polar_lut, sample_polar, radon_search, refine_velocity
from validate_v7 import temporal_highpass, unwrap_to_line, fit_plain, fit_harmonic, N_A, WARMUP
from exp_abscenter import load_generous, abs_centers

CACHE = os.path.join(os.path.dirname(__file__), "cache")
OUT = os.path.join(os.path.dirname(__file__), "diagnostics_v9")


def get_crops_centers(video, name):
    os.makedirs(CACHE, exist_ok=True)
    gp = os.path.join(CACHE, f"{name}_gen.npy"); mp = os.path.join(CACHE, f"{name}_gen_meta.npz")
    if os.path.exists(gp) and os.path.exists(mp):
        m = np.load(mp)
        return np.load(gp, mmap_mode="r"), m["ts"], int(m["side"]), float(m["r"]), m["cx"], m["cy"]
    print(f"  decoding {name} (one-time) ...")
    crops, ts, side, r = load_generous(video)
    cx, cy, ngood = abs_centers(crops, r, side)
    np.save(gp, crops)
    np.savez(mp, ts=ts, side=side, r=r, cx=cx, cy=cy)
    print(f"    {len(crops)} frames, {ngood} good center detections")
    return crops, ts, side, r, cx, cy


def build_kymo(crops, cx, cy, r, side, ox, oy):
    n = len(crops)
    kymo = np.empty((n, N_A), np.float32)
    for i in range(n):
        lut = make_polar_lut(side, cx[i]+ox, cy[i]+oy, r*0.50, r*0.85, N_A, 25)
        kymo[i] = sample_polar(np.asarray(crops[i]).astype(np.float32), lut)
    return kymo


def track_angles(kymo, ts):
    hp = np.abs(temporal_highpass(kymo)); kz = hp - hp.mean(axis=1, keepdims=True)
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
    return ang, conf, pred, vel, phase, coarse[1], kz


def measure(video):
    os.makedirs(OUT, exist_ok=True)
    name = os.path.splitext(os.path.basename(video))[0]
    print(f"\n{'='*68}\n  {name}\n{'='*68}")
    t0 = time.time()
    crops, ts, side, r, cx, cy = get_crops_centers(video, name)
    ts = ts[:len(crops)]
    R_eff = r*0.675  # mid-ring radius where the second hand lives (px)

    ox = oy = 0.0
    sign = 1.0
    prev_amp = None
    for it in range(5):
        kymo = build_kymo(crops, cx, cy, r, side, ox, oy)[WARMUP:]
        tt = ts[WARMUP:]
        ang, conf, pred, vel, phase, snr, kz = track_angles(kymo, tt)
        m = conf >= np.percentile(conf, 50)
        y = unwrap_to_line(ang[m], tt[m], vel, phase)
        # harmonic fit to read the once-per-rev (center-offset) term
        slope, se, rms, nfit, amp1 = fit_harmonic(tt[m], y, pred[m], nharm=1)
        mp, sep, rmsp, _ = fit_plain(tt[m], y)
        # 1st-harmonic coefficients -> center offset (deg = -(ex cos + ey sin)/R)
        pr = np.radians(pred[m]); A = np.column_stack([np.ones_like(tt[m]), tt[m], np.cos(pr), np.sin(pr)])
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        a1, b1 = coef[2], coef[3]
        amp = np.hypot(a1, b1)
        print(f"   iter{it}: offset=({ox:+.1f},{oy:+.1f})px  harm-amp={amp:.3f}deg  "
              f"plain={(mp-6)/6*86400:+7.1f}  harmonic={(slope-6)/6*86400:+7.1f} s/day  snr={snr:.0f}")
        if prev_amp is not None and amp > prev_amp + 0.02:
            sign = -sign  # wrong direction last step; flip
        prev_amp = amp
        if amp < 0.03:
            break
        # center shift to null: dtheta(deg) = -(ex cos + ey sin)/R*(180/pi)
        # so ex = -a1*R*pi/180, ey = -b1*R*pi/180; move center by that.
        dx = sign * a1 * R_eff * np.pi/180.0
        dy = sign * b1 * R_eff * np.pi/180.0
        ox += dx; oy += dy

    # final measurement with nulled center: PLAIN fit (now unbiased), gated
    kymo = build_kymo(crops, cx, cy, r, side, ox, oy)[WARMUP:]
    tt = ts[WARMUP:]
    ang, conf, pred, vel, phase, snr, kz = track_angles(kymo, tt)
    # quality gating: confidence floor + percentile
    m = (conf >= np.percentile(conf, 50)) & (conf >= 4.0)
    if m.sum() < 120:
        m = conf >= np.percentile(conf, 40)
    y = unwrap_to_line(ang[m], tt[m], vel, phase)
    slope, se, rms, nfit = fit_plain(tt[m], y)
    drift = (slope-6.0)/6.0*86400; drift_se = se/6.0*86400
    dur = tt[-1]-tt[0]
    cdrift = float(np.hypot(cx-np.median(cx), cy-np.median(cy)).max())
    print(f"  --> nulled center offset ({ox:+.1f},{oy:+.1f})px, residual harm done")
    print(f"  >>> DRIFT {drift:+.1f} +/- {drift_se:.1f} s/day  "
          f"(dur {dur:.0f}s, center-track {cdrift:.0f}px, SNR {snr:.0f}, N={nfit}, rms {rms:.2f}, {time.time()-t0:.0f}s)")

    img = cv2.normalize(kz, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR); bpd = N_A/360.0
    for i in np.where(m)[0]:
        img[i, int(round(ang[i]*bpd)) % N_A] = (0, 255, 0)
    cv2.putText(img, f"{name} drift={drift:+.1f}+/-{drift_se:.1f} s/day SNR={snr:.0f}",
                (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    cv2.imwrite(os.path.join(OUT, f"{name}_v9.png"), img)
    return name, drift, drift_se


def main(argv):
    targets = argv[1:]
    if not targets:
        print("usage: validate_v9.py video.MOV ..."); return
    res = [measure(v) for v in targets]
    print(f"\n{'='*44}\nSUMMARY\n{'='*44}")
    for nm, d, se in res:
        print(f"  {nm:<16} {d:+7.1f} +/- {se:4.1f} s/day")


if __name__ == "__main__":
    main(sys.argv)
