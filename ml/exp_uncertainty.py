#!/usr/bin/env python3
"""
How wrong is the reading at time T, and what predicts it?

The matched-filter estimator's residual-based SE is 10-20x too optimistic
(residuals are correlated: the contact and the watch itself wander). So measure
the real thing: |rate(T) - rate(final)| across the corpus, and compare it to
candidate predictors so we can publish an honest +/-.

  ./venv/bin/python exp_uncertainty.py recordings/*.wav
"""
import glob
import sys

import numpy as np
import soundfile as sf

from tg_lab import run_beats, select_band


def load(path):
    x, fs = sf.read(path, always_2d=False)
    if x.ndim == 2:
        x = x.mean(axis=1)
    return x.astype(np.float64), float(fs)


def subwindow_rates(res, tau_s):
    """Rate fitted in each non-overlapping tau-second block of beat events."""
    k, t, keep = res["k"], res["t"], res["keep"]
    k, t = k[keep], t[keep]
    if len(k) < 8:
        return np.array([])
    out = []
    t0 = t[0]
    while True:
        m = (t >= t0) & (t < t0 + tau_s)
        if m.sum() < 8:
            break
        A = np.vstack([np.ones(m.sum()), k[m]]).T
        coef, *_ = np.linalg.lstsq(A, t[m], rcond=None)
        nominal = 7200.0 / res["bph"]
        out.append((nominal / coef[1] - 1) * 86400)
        t0 += tau_s
        if t0 > t[-1] - tau_s:
            break
    return np.array(out)


def main(paths):
    rows = []
    for p in paths:
        x, fs = load(p)
        dur = len(x) / fs
        band = select_band(x, fs)
        final = run_beats(x, fs, band=band)
        if final is None:
            continue
        print(f"\n=== {p}  {dur:.0f}s  bph={final['bph']}  final={final['rate']:+.1f} ===")
        print(f"{'T':>5} {'rate':>8} {'err':>7} {'SEnaive':>8} {'jit_ms':>7} "
              f"{'match':>6} {'sd10':>7} {'sd10*sc':>8}")
        for T in [8, 10, 12, 16, 20, 25, 30, 40, 50, 60]:
            if T > dur - 1:
                break
            r = run_beats(x, fs, up_to=T, band=band)
            if r is None:
                continue
            err = r["rate"] - final["rate"]
            sub = subwindow_rates(r, 5.0)
            sd5 = sub.std(ddof=1) if len(sub) > 1 else np.nan
            # white-phase-noise scaling: SE(T) = sd(tau) * (tau/T)^1.5 / sqrt(n)
            n_blocks = max(1, len(sub))
            scaled = sd5 * (5.0 / T) ** 1.5 * np.sqrt(n_blocks) if len(sub) > 1 else np.nan
            match = r["n_beats"] / max(1, r["n_total"])
            rows.append(dict(T=T, err=abs(err), jit=r["jitter_ms"], match=match,
                             se=r["se_rate"], sd5=sd5, scaled=scaled, dur=dur))
            print(f"{T:5.0f} {r['rate']:+8.1f} {err:+7.2f} {r['se_rate']:8.3f} "
                  f"{r['jitter_ms']:7.3f} {match:6.2f} {sd5:7.2f} {scaled:7.3f}")

    if not rows:
        return
    print("\n=== predictor quality (ratio err/predictor; want ~1, never <<1) ===")
    for name in ["se", "sd5", "scaled"]:
        v = np.array([r["err"] / max(r[name], 1e-6) for r in rows
                      if np.isfinite(r[name]) and r["T"] < r["dur"] - 5])
        v = v[np.isfinite(v)]
        if len(v):
            print(f"{name:>8}: median {np.median(v):7.2f}  p90 {np.percentile(v, 90):8.2f}"
                  f"  max {v.max():8.2f}")

    print("\n=== err vs T (all files, T < dur-5) ===")
    for T in [8, 10, 12, 16, 20, 25, 30, 40]:
        e = [r["err"] for r in rows if r["T"] == T and r["T"] < r["dur"] - 5]
        if e:
            print(f"T={T:3.0f}  n={len(e):2d}  median {np.median(e):5.2f}  "
                  f"p90 {np.percentile(e, 90):5.2f}  max {max(e):5.2f}")


if __name__ == "__main__":
    ps = []
    for a in sys.argv[1:]:
        ps.extend(sorted(glob.glob(a)))
    main(ps)
