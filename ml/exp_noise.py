#!/usr/bin/env python3
"""
Where does the beat tracker break under noise, and what would fix it?

Anecdotes about "a noisy room" aren't measurable, so inject controlled
interference into clean recordings and score the result against that recording's
own clean reading. Four kinds, because they break things differently:

  hiss    broadband white noise           — raises the floor everywhere
  tone    narrowband whine (fan, coil)    — one loud in-band line
  knocks  sparse impulsive transients     — table taps, the watch shifting
  rustle  bursty bandpass noise           — fabric, hand movement, paper

Levels are in dB relative to the recording's own in-band RMS, so 0 dB means the
interference is as loud as everything the microphone already hears.

  ./venv/bin/python exp_noise.py                 # baseline sweep
  ./venv/bin/python exp_noise.py --mitigations   # compare fixes
"""
import argparse
import glob

import numpy as np
import soundfile as sf
from scipy.signal import butter, lfilter

import tune_timegrapher as tg

RNG_SEED = 12345
BAND = (4000, 20000)


# ---------------------------------------------------------------------------
# noise generation
# ---------------------------------------------------------------------------

def inband_rms(x, fs, lo=BAND[0], hi=BAND[1]):
    b, a = butter(2, [lo / (fs / 2), min(hi, fs / 2 - 500) / (fs / 2)], btype="band")
    return float(np.sqrt(np.mean(lfilter(b, a, x) ** 2)))


def make_noise(kind, n, fs, rng):
    """Unit-RMS interference of the requested kind."""
    t = np.arange(n) / fs
    if kind == "hiss":
        v = rng.standard_normal(n)
    elif kind == "tone":
        # Slight wobble so it isn't a perfectly stationary line.
        f0 = 9000 + 40 * np.sin(2 * np.pi * 0.3 * t)
        v = np.sin(2 * np.pi * np.cumsum(f0) / fs)
    elif kind == "knocks":
        v = np.zeros(n)
        # ~1.5 knocks a second, each a 6 ms decaying burst.
        for start in rng.integers(0, n, size=max(1, int(1.5 * n / fs))):
            L = int(0.006 * fs)
            if start + L >= n:
                continue
            env = np.exp(-np.arange(L) / (0.0012 * fs))
            v[start:start + L] += rng.standard_normal(L) * env
    elif kind == "rustle":
        # Bursty broadband: noise gated by a slow random envelope.
        env = np.abs(rng.standard_normal(n // 512 + 1))
        env = np.repeat(env, 512)[:n]
        env = np.convolve(env, np.ones(2048) / 2048, mode="same")
        b, a = butter(2, [5000 / (fs / 2), 18000 / (fs / 2)], btype="band")
        v = lfilter(b, a, rng.standard_normal(n)) * env
    else:
        raise ValueError(kind)
    r = np.sqrt(np.mean(v ** 2))
    return v / (r + 1e-12)


def contaminate(x, fs, kind, db, rng):
    if kind == "clean":
        return x
    target = inband_rms(x, fs) * (10 ** (db / 20))
    return x + make_noise(kind, len(x), fs, rng) * target


# ---------------------------------------------------------------------------
# running the pipeline
# ---------------------------------------------------------------------------

def run(x, fs, tracker_class=None):
    """Replay a signal through the analyzer; return the final result dict."""
    original = tg.BeatTracker
    if tracker_class is not None:
        tg.BeatTracker = tracker_class
    try:
        g = tg.Timegrapher(fs)
        chunk = int(0.5 * fs)
        pos, last = 0, None
        while pos < len(x):
            g.process(x[pos:pos + chunk])
            pos += chunk
            last = g.analyze()
        return last
    finally:
        tg.BeatTracker = original


def load(path):
    x, fs = sf.read(path, always_2d=False)
    if x.ndim == 2:
        x = x.mean(axis=1)
    return x.astype(np.float64), float(fs)


def score(res, truth):
    """(error, printable) for one run."""
    if res is None or res["rate"] is None:
        return None, f"{'—':>7} {'lost':>6}"
    err = res["rate"] - truth
    return abs(err), (f"{err:+7.1f} {res['unc']:6.1f} {res['jitter']:6.3f} "
                      f"{res['detect']:5.2f} {res['stage']:>9}")


# ---------------------------------------------------------------------------
# sweeps
# ---------------------------------------------------------------------------

KINDS = ["hiss", "tone", "knocks", "rustle"]
LEVELS = [6, 12, 18, 24, 30]


def baseline_sweep(paths):
    rng = np.random.default_rng(RNG_SEED)
    print(f"{'file':>14} {'noise':>7} {'dB':>4} | {'err':>7} {'±':>6} "
          f"{'jit':>6} {'det':>5} {'stage':>9}")
    fails = {}
    for p in paths:
        x, fs = load(p)
        clean = run(x, fs)
        if clean is None or clean["rate"] is None:
            print(f"{p:>14}  clean run failed, skipping")
            continue
        truth = clean["rate"]
        name = p.split("/")[-1].replace(".wav", "")
        print(f"{name:>14} {'clean':>7} {'—':>4} | {0.0:+7.1f} {clean['unc']:6.1f} "
              f"{clean['jitter']:6.3f} {clean['detect']:5.2f} {clean['stage']:>9}")
        for kind in KINDS:
            for db in LEVELS:
                y = contaminate(x, fs, kind, db, rng)
                err, text = score(run(y, fs), truth)
                print(f"{'':>14} {kind:>7} {db:4d} | {text}")
                key = (kind, db)
                fails.setdefault(key, []).append(err)
    print("\n=== breakdown by interference (median |err|, lost-lock count) ===")
    print(f"{'noise':>7} {'dB':>4} | {'median err':>10} {'p90':>7} {'lost':>5} {'n':>3}")
    for kind in KINDS:
        for db in LEVELS:
            v = fails.get((kind, db), [])
            lost = sum(1 for e in v if e is None)
            ok = [e for e in v if e is not None]
            med = np.median(ok) if ok else float("nan")
            p90 = np.percentile(ok, 90) if ok else float("nan")
            print(f"{kind:>7} {db:4d} | {med:10.2f} {p90:7.2f} {lost:5d} {len(v):3d}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("wavs", nargs="*", default=None)
    ap.add_argument("--mitigations", action="store_true")
    args = ap.parse_args()
    paths = args.wavs or sorted(glob.glob("recordings/*.wav"))
    # The degrading recording has no stable truth to score against.
    paths = [p for p in paths if "new3" not in p]
    if args.mitigations:
        from exp_noise_fixes import mitigation_sweep
        mitigation_sweep(paths)
    else:
        baseline_sweep(paths)


if __name__ == "__main__":
    main()
