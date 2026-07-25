#!/usr/bin/env python3
"""
How fast can we honestly put a number on screen?

Two knobs decide it: how long the band probe listens before locking, and how
long the sub-window blocks are (the +/- needs at least two of them). Shortening
either gets a reading up sooner but risks a +/- that no longer covers the real
error — so sweep both and measure, for each setting:

  - time to the first published number, and to reaching +/-2 s/day
  - coverage: how often |rate(t) - final rate| stays inside the published +/-

  ./venv/bin/python exp_timing.py recordings/*.wav
"""
import glob
import sys

import numpy as np
import soundfile as sf

import tune_timegrapher as tg


def replay_collect(path, probe_s, block_s, interval=0.5):
    """Replay one file, returning [(t, rate, unc, stage)] and the final rate."""
    x, fs = sf.read(path, always_2d=False)
    if x.ndim == 2:
        x = x.mean(axis=1)
    x = x.astype(np.float64)
    fs = float(fs)

    tg.PROBE_SECONDS = probe_s
    tg.BLOCK_SECONDS = block_s
    g = tg.Timegrapher(fs)
    chunk = int(interval * fs)
    rows, pos = [], 0
    while pos < len(x):
        g.process(x[pos:pos + chunk])
        pos += chunk
        r = g.analyze()
        rows.append((r["elapsed"], r["rate"], r["unc"], r["stage"]))
    final = rows[-1][1] if rows else None
    return rows, final


def evaluate(paths, probe_s, block_s):
    first, to_target, ratios, failures = [], [], [], 0
    for p in paths:
        rows, final = replay_collect(p, probe_s, block_s)
        if final is None:
            failures += 1
            continue
        got_first = got_target = None
        for t, rate, unc, stage in rows:
            if rate is None or unc is None:
                continue
            if got_first is None:
                got_first = t
            if got_target is None and unc <= tg.TARGET_PRECISION:
                got_target = t
            # Only score readings with enough recording left that `final` is a
            # meaningfully better answer than the reading being scored.
            if t < rows[-1][0] - 5:
                ratios.append(abs(rate - final) / max(unc, 0.5))
        if got_first:
            first.append(got_first)
        if got_target:
            to_target.append(got_target)
    r = np.array(ratios) if ratios else np.array([np.nan])
    return dict(
        probe=probe_s, block=block_s,
        first=np.median(first) if first else np.nan,
        first_max=max(first) if first else np.nan,
        target=np.median(to_target) if to_target else np.nan,
        n_target=len(to_target),
        covered=100 * np.mean(r <= 1),
        median_ratio=np.median(r),
        max_ratio=np.nanmax(r),
        failures=failures,
    )


def main(paths):
    print(f"{'probe':>6} {'block':>6} | {'1st#':>6} {'worst':>6} {'±2 at':>6} "
          f"{'n':>3} | {'covered':>8} {'med':>6} {'max':>6} {'fail':>5}")
    for probe_s in (3.0, 4.0, 5.0, 6.0):
        for block_s in (2.5, 3.0, 4.0, 5.0):
            e = evaluate(paths, probe_s, block_s)
            print(f"{e['probe']:6.1f} {e['block']:6.1f} | {e['first']:6.1f} "
                  f"{e['first_max']:6.1f} {e['target']:6.1f} {e['n_target']:3d} | "
                  f"{e['covered']:7.0f}% {e['median_ratio']:6.2f} "
                  f"{e['max_ratio']:6.2f} {e['failures']:5d}")


if __name__ == "__main__":
    ps = []
    for a in sys.argv[1:]:
        ps.extend(sorted(glob.glob(a)))
    main(ps)
