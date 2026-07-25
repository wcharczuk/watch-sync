#!/usr/bin/env python3
"""Per-frame watch-center tracking, then measure.

Diagnosis: over a 2-min handheld clip the watch DRIFTS within the frame; the
fixed frame-0 crop + global phase-corr lose it, so the radial sampling center
slides off the true pivot -> time-varying angle bias -> slope bias. Rotation is
small (hardware-stabilized), so the fix is to follow the watch CENTER, not to
de-rotate.

Here: track the center per kept frame via sequential phase-correlation on a
watch-sized patch, with a periodic Hough re-anchor to stop accumulation drift,
and crop tightly about the moving center. Then raw kymograph -> high-pass ->
confidence-gated robust fit. Caches tracked crops.
"""
import os, sys, time
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(__file__))
from validate_v5 import make_polar_lut, sample_polar, detect_watch_face
from validate_v7 import temporal_highpass, fit_harmonic, fit_plain, unwrap_to_line, N_A, WARMUP
from validate_v5 import radon_search, refine_velocity

CACHE = os.path.join(os.path.dirname(__file__), "cache")


def load_crops_tracked(video_path, target_fps=30, max_frames=8000, max_h=1080,
                       reanchor=150):
    cap = cv2.VideoCapture(video_path)
    native = cap.get(cv2.CAP_PROP_FPS) or 60.0
    skip = max(1, int(round(native/target_fps)))
    grays, ts, centers = [], [], []
    side = r = None
    cx = cy = None
    prev_patch = None
    han = None
    idx = 0; kept = 0
    while kept < max_frames:
        ok, fr = cap.read()
        if not ok:
            break
        t = cap.get(cv2.CAP_PROP_POS_MSEC)/1000.0
        if idx % skip == 0:
            h, w = fr.shape[:2]
            if h > max_h:
                s = max_h/h
                fr = cv2.resize(fr, (int(w*s), int(h*s)), interpolation=cv2.INTER_AREA)
            g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
            H, W = g.shape
            if side is None:
                det = detect_watch_face(fr)
                if det is None:
                    idx += 1; continue
                cx, cy, r = det
                side = int(r*2.2)
                han = cv2.createHanningWindow((side, side), cv2.CV_32F)
            else:
                # sequential phase-corr: how did the watch patch move?
                x0p = int(round(cx-side/2)); y0p = int(round(cy-side/2))
                x0p = max(0, min(W-side, x0p)); y0p = max(0, min(H-side, y0p))
                cur_patch = g[y0p:y0p+side, x0p:x0p+side].astype(np.float32)
                if prev_patch is not None and cur_patch.shape == prev_patch.shape:
                    (dx, dy), _ = cv2.phaseCorrelate(prev_patch, cur_patch, han)
                    cx += dx; cy += dy
                # periodic Hough re-anchor to kill accumulation drift
                if kept % reanchor == 0:
                    det = detect_watch_face(fr)
                    if det is not None:
                        ncx, ncy, nr = det
                        if abs(ncx-cx) < 0.3*r and abs(ncy-cy) < 0.3*r:
                            cx, cy = ncx, ncy  # snap if plausible
            cx = float(np.clip(cx, side/2, W-side/2))
            cy = float(np.clip(cy, side/2, H-side/2))
            x0 = int(round(cx-side/2)); y0 = int(round(cy-side/2))
            x0 = max(0, min(W-side, x0)); y0 = max(0, min(H-side, y0))
            crop = g[y0:y0+side, x0:x0+side]
            grays.append(crop.copy()); ts.append(t); centers.append((cx, cy))
            prev_patch = crop.astype(np.float32)
            kept += 1
        idx += 1
    cap.release()
    return np.stack(grays), np.array(ts), side, float(r), np.array(centers)


def get_tracked(video_path, name):
    os.makedirs(CACHE, exist_ok=True)
    gp = os.path.join(CACHE, f"{name}_ct.npy"); mp = os.path.join(CACHE, f"{name}_ct_meta.npz")
    if os.path.exists(gp) and os.path.exists(mp):
        m = np.load(mp)
        return np.load(gp, mmap_mode="r"), m["ts"], int(m["side"]), float(m["r"]), m["centers"]
    print(f"  decoding+tracking {name} (one-time) ...")
    g, ts, side, r, centers = load_crops_tracked(video_path)
    np.save(gp, g); np.savez(mp, ts=ts, side=side, r=r, centers=centers)
    return g, ts, side, r, centers


def measure(video, label):
    name = os.path.splitext(os.path.basename(video))[0]
    t0 = time.time()
    grays, ts, side, r, centers = get_tracked(video, name)
    cmove = np.hypot(*(centers - centers[0]).T)
    cl = side/2.0
    lut = make_polar_lut(side, cl, cl, cl*0.50, cl*0.85, N_A, 25)
    kymo = np.stack([sample_polar(np.asarray(grays[i]).astype(np.float32), lut)
                     for i in range(len(grays))])[WARMUP:]
    ts = ts[WARMUP:]
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
    print(f"\n=== {name} ({label}) === snr={coarse[1]:.0f}  center drift max={cmove.max():.0f}px "
          f"end={cmove[-1]:.0f}px  ({time.time()-t0:.0f}s)")
    for pct in [0, 50, 70, 85]:
        thr = np.percentile(conf, pct); m = conf >= thr
        if m.sum() < 60:
            continue
        y = unwrap_to_line(ang[m], ts[m], vel, phase)
        mh, seh, rmsh, nh, amp1 = fit_harmonic(ts[m], y, pred[m])
        mp, sep, rmsp, _ = fit_plain(ts[m], y)
        print(f"  conf>=p{pct:<2d} N={m.sum():4d} dur={ts[m].max()-ts[m].min():5.1f}s  "
              f"plain {(mp-6)/6*86400:+8.1f}  harmonic {(mh-6)/6*86400:+8.1f}+/-{seh/6*86400:4.1f} (rms{rmsh:.2f})")


if __name__ == "__main__":
    targets = sys.argv[1:] or ["videos/IMG_7854.MOV"]
    for v in targets:
        measure(v, "handheld")
