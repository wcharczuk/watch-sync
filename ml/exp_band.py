#!/usr/bin/env python3
"""
Which band should we listen in — and does it matter?

Today the app picks the band with the strongest spectral beat line. But what we
actually need is the band in which each beat can be *timed* most precisely. So
score every candidate band by beat-timing jitter and 5 s block-rate scatter, and
see whether the spectral ranking picks the same winner.

Also asks the question behind "can we boost certain frequencies": is a single
band the right model at all, or should several be combined?

  ./venv/bin/python exp_band.py recordings/*.wav
"""
import glob
import sys

import numpy as np
import soundfile as sf

from tg_lab import dft_mag, envelope, select_band
from exp_shippable import analyze

BANDS = [(1000, 4000), (2000, 6000), (4000, 9000), (5000, 11000), (7000, 13000),
         (9000, 15000), (11000, 18000), (13000, 21000), (15000, 23000),
         (3000, 20000), (1000, 20000), (6000, 20000)]


def prominence(x, fs, lo, hi, bph, scan_s=12.0):
    env, er = envelope(x[: int(scan_s * fs)], fs, lo, hi)
    f0 = bph / 3600
    nb = np.mean([dft_mag(env, er, f0 * k) for k in (0.6, 0.75, 1.2, 1.35, 1.6)])
    return dft_mag(env, er, f0) / (nb + 1e-12)


def main(paths):
    agree = []
    for p in paths:
        x, fs = sf.read(p, always_2d=False)
        if x.ndim == 2:
            x = x.mean(axis=1)
        x = x.astype(np.float64)
        fs = float(fs)
        auto = select_band(x, fs)
        ref = analyze(x, fs, band=auto)
        print(f"\n=== {p}  bph {auto['bph']}  auto {auto['lo']/1000:.0f}-"
              f"{auto['hi']/1000:.0f}k  ->{ref['rate'] if ref else float('nan'):+.2f} ===")
        print(f"{'band':>12} {'prom':>7} {'rate':>8} {'jit_ms':>7} {'detect':>7} "
              f"{'score':>6} {'sd5':>7}")
        rows = []
        for lo, hi in BANDS:
            prom = prominence(x, fs, lo, hi, auto["bph"])
            r = analyze(x, fs, band=dict(lo=lo, hi=hi, bph=auto["bph"]))
            if r is None:
                print(f"{lo/1000:5.0f}-{hi/1000:<4.0f}k {prom:7.1f}    (no lock)")
                continue
            sd5 = r["blocks"].std(ddof=1) if len(r["blocks"]) > 1 else np.nan
            mark = "  <- auto" if (lo, hi) == (auto["lo"], auto["hi"]) else ""
            print(f"{lo/1000:5.0f}-{hi/1000:<4.0f}k {prom:7.1f} {r['rate']:+8.2f} "
                  f"{r['jitter_ms']:7.3f} {r['detect']:7.2f} {r['score']:6.2f} "
                  f"{sd5:7.2f}{mark}")
            rows.append((prom, r["jitter_ms"], sd5, r["rate"], (lo, hi)))
        if rows:
            by_prom = max(rows, key=lambda r: r[0])
            by_jit = min(rows, key=lambda r: r[1])
            by_sd = min((r for r in rows if np.isfinite(r[2])), key=lambda r: r[2],
                        default=None)
            print(f"  best prominence {by_prom[4]} -> {by_prom[3]:+.2f} "
                  f"(jit {by_prom[1]:.3f})")
            print(f"  best jitter     {by_jit[4]} -> {by_jit[3]:+.2f} "
                  f"(jit {by_jit[1]:.3f})")
            if by_sd:
                print(f"  best block sd   {by_sd[4]} -> {by_sd[3]:+.2f} "
                      f"(sd5 {by_sd[2]:.2f})")
            agree.append(abs(by_prom[3] - by_jit[3]))
    if agree:
        a = np.array(agree)
        print(f"\n=== rate difference between prominence-picked and jitter-picked "
              f"band: median {np.median(a):.2f} max {a.max():.2f} s/day ===")


if __name__ == "__main__":
    ps = []
    for a in sys.argv[1:]:
        ps.extend(sorted(glob.glob(a)))
    main(ps)
