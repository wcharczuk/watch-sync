#!/usr/bin/env python3
"""Time-varying center correction: refine the per-frame center by sub-pixel
phase-correlation to a sharp median reference (absolute, lag-free, no
accumulation), replacing the noisy/laggy per-frame Hough. Then static
pivot-offset nulling on top. Runs from the cached generous crops (fast, no
re-decode).
"""
import os, sys, time
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(__file__))
from validate_v5 import make_polar_lut, sample_polar, radon_search, refine_velocity
from validate_v7 import temporal_highpass, unwrap_to_line, fit_plain, N_A, WARMUP

CACHE = os.path.join(os.path.dirname(__file__), "cache")


def load_gen(name):
    g = np.load(os.path.join(CACHE, f"{name}_gen.npy"), mmap_mode="r")
    m = np.load(os.path.join(CACHE, f"{name}_gen_meta.npz"))
    return g, m["ts"], int(m["side"]), float(m["r"]), m["cx"], m["cy"]


def shift_to_center(img, cx, cy, side):
    M = np.float32([[1, 0, side/2.0-cx], [0, 1, side/2.0-cy]])
    return cv2.warpAffine(img, M, (side, side), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REFLECT)


def refine_centers(crops, cx, cy, side, r, iters=2):
    """Per-frame center via sub-pixel phase-corr to a median reference."""
    n = len(crops)
    cx = cx.astype(np.float64).copy(); cy = cy.astype(np.float64).copy()
    # correlation window: central ~2r region (the dial), Hanning-weighted
    w = int(min(side, 2.1*r)); w -= w % 2
    o = (side-w)//2
    han = cv2.createHanningWindow((w, w), cv2.CV_32F)
    for _ in range(iters):
        # build reference: median of crops aligned to current centers
        acc = np.zeros((side, side), np.float32)
        # median is heavy on 3600 frames; use a strided subset for the ref
        idxs = np.arange(0, n, max(1, n//400))
        stack = np.stack([shift_to_center(np.asarray(crops[i]).astype(np.float32),
                                          cx[i], cy[i], side) for i in idxs])
        ref = np.median(stack, axis=0)
        ref_win = (ref[o:o+w, o:o+w]*han).astype(np.float32)
        for i in range(n):
            al = shift_to_center(np.asarray(crops[i]).astype(np.float32),
                                 cx[i], cy[i], side)
            cur = (al[o:o+w, o:o+w]*han).astype(np.float32)
            (dx, dy), resp = cv2.phaseCorrelate(ref_win, cur)
            # `al` has the watch at center+? ; shift tells how cur moved vs ref.
            # the true center in the ORIGINAL crop = current center minus this.
            cx[i] -= dx; cy[i] -= dy
    return cx, cy


def measure(name, refined=True):
    crops, ts, side, r, cxH, cyH = load_gen(name)
    ts = ts[:len(crops)]
    t0 = time.time()
    if refined:
        cx, cy = refine_centers(crops, cxH, cyH, side, r)
        tag = "phasecorr-center"
    else:
        cx, cy = cxH.astype(float), cyH.astype(float)
        tag = "hough-center"
    # how time-varying is the center?
    cmove = float(np.hypot(cx-np.median(cx), cy-np.median(cy)).max())

    def build(ox, oy):
        k = np.empty((len(crops), N_A), np.float32)
        for i in range(len(crops)):
            lut = make_polar_lut(side, cx[i]+ox, cy[i]+oy, r*0.50, r*0.85, N_A, 25)
            k[i] = sample_polar(np.asarray(crops[i]).astype(np.float32), lut)
        return k

    def track(k, tt):
        hp = np.abs(temporal_highpass(k)); kz = hp-hp.mean(axis=1, keepdims=True)
        n_t, n_a = kz.shape; bpd = n_a/360.0
        c = radon_search(kz, 30.0, n_a, vel_min=4.5, vel_max=7.5)
        vel, peak, ref = refine_velocity(kz, c[0], 30.0, n_a, window=0.3, step=0.005)
        phase = (peak*360.0/n_a)-vel*tt[ref]; win = int(round(2.5*bpd))
        ang = np.zeros(n_t); conf = np.zeros(n_t); pred = np.zeros(n_t)
        for i in range(n_t):
            p = phase+vel*tt[i]; pred[i] = p; pb = int(round((p % 360.0)*bpd)) % n_a
            js = np.arange(pb-win, pb+win+1); vals = np.clip(kz[i, js % n_a], 0, None); ws = vals.sum()
            row = kz[i]; mad = np.median(np.abs(row-np.median(row)))+1e-6; conf[i] = vals.max()/mad
            ang[i] = ((js*vals).sum()/ws*360.0/n_a) % 360.0 if ws > 1e-6 else p % 360.0
        return ang, conf, pred, vel, phase, c[1]

    # iterative static-offset nulling
    R_eff = r*0.675; ox = oy = 0.0; sign = 1.0; prev = None
    for it in range(5):
        k = build(ox, oy)[WARMUP:]; tt = ts[WARMUP:]
        ang, conf, pred, vel, phase, snr = track(k, tt)
        m = conf >= np.percentile(conf, 50)
        y = unwrap_to_line(ang[m], tt[m], vel, phase)
        pr = np.radians(pred[m]); A = np.column_stack([np.ones_like(tt[m]), tt[m], np.cos(pr), np.sin(pr)])
        coef, *_ = np.linalg.lstsq(A, y, rcond=None); a1, b1 = coef[2], coef[3]; amp = np.hypot(a1, b1)
        mp, sep, rms, _ = fit_plain(tt[m], y)
        if prev is not None and amp > prev+0.02:
            sign = -sign
        prev = amp
        if amp < 0.03:
            break
        ox += sign*a1*R_eff*np.pi/180.0; oy += sign*b1*R_eff*np.pi/180.0
    k = build(ox, oy)[WARMUP:]; tt = ts[WARMUP:]
    ang, conf, pred, vel, phase, snr = track(k, tt)
    m = (conf >= np.percentile(conf, 50)) & (conf >= 4.0)
    if m.sum() < 120:
        m = conf >= np.percentile(conf, 40)
    y = unwrap_to_line(ang[m], tt[m], vel, phase)
    slope, se, rms, nfit = fit_plain(tt[m], y)
    print(f"  {name} [{tag}]: center-move={cmove:.0f}px offset=({ox:+.1f},{oy:+.1f}) "
          f"SNR={snr:.0f}  DRIFT {(slope-6)/6*86400:+.1f} +/- {se/6*86400:.1f} s/day "
          f"(rms {rms:.2f}, N={nfit}, {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    for name in (sys.argv[1:] or ["IMG_7854", "IMG_7855"]):
        measure(name, refined=False)
        measure(name, refined=True)
