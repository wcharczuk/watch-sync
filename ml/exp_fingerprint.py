#!/usr/bin/env python3
"""
Can a watch be recognised by the sound of its escapement?

The idea: once beats are being tracked we know exactly where every tick is, so we
can average hundreds of them into a clean acoustic signature and compare it
against previously labelled ones. If signatures from the same watch match each
other more closely than they match other watches, the app could offer "this looks
like your Tudor" and let the user confirm.

The fingerprint deliberately avoids anything the *measurement* chose: the
listening band varies run to run, so comparing templates directly would compare
band choices. Instead each tick is cut from the raw audio and reduced to a
log-power spectrum over a fixed 1–23 kHz range, then median-averaged over every
beat — a description of the tick itself.

Beat rate is kept separate: it's a hard constraint (a 21600 watch is never a
28800 one), not part of the similarity score.

  ./venv/bin/python exp_fingerprint.py
"""
import glob
import os

import numpy as np
import soundfile as sf

import tune_timegrapher as tg

BANDS = 64
FLO, FHI = 1000.0, 23000.0
WINDOW_MS = 24.0


def load(path):
    x, fs = sf.read(path, always_2d=False)
    if x.ndim == 2:
        x = x.mean(axis=1)
    return x.astype(np.float64), float(fs)


def track(x, fs):
    """Run the shipped pipeline and hand back the tracker with its beat times."""
    g = tg.Timegrapher(fs)
    chunk = int(0.5 * fs)
    pos, last = 0, None
    while pos < len(x):
        g.process(x[pos:pos + chunk])
        pos += chunk
        last = g.analyze()
    return g.tracker, last


def fingerprint(x, fs, tracker):
    """Median log-power spectrum of the tick, over a fixed frequency range."""
    if tracker is None or len(tracker.t) == 0:
        return None
    n = int(WINDOW_MS * fs / 1000)
    win = np.hanning(n)
    edges = np.linspace(FLO, FHI, BANDS + 1)
    freqs = np.fft.rfftfreq(n, 1 / fs)
    idx = [np.where((freqs >= edges[i]) & (freqs < edges[i + 1]))[0]
           for i in range(BANDS)]

    rows = []
    for t, keep in zip(tracker.t, tracker.keep):
        if not keep:
            continue
        s = int(t * fs)
        if s < 0 or s + n >= len(x):
            continue
        seg = x[s:s + n] * win
        spec = np.abs(np.fft.rfft(seg)) ** 2
        rows.append([spec[i].mean() if len(i) else 0.0 for i in idx])
    if len(rows) < 20:
        return None
    v = np.log10(np.median(np.array(rows), axis=0) + 1e-20)
    # Remove overall loudness and contrast: coupling changes both, and neither
    # says anything about which watch this is.
    v = v - v.mean()
    return v / (np.linalg.norm(v) + 1e-12)


def discriminative(prints):
    """Subtract the component every recording shares.

    Raw tick spectra all correlate 0.85-0.99, because what dominates them is the
    phone's own response and the general character of "a click" — not the watch.
    Removing the corpus mean leaves only what differs, which is where any
    identification has to come from. On this corpus it takes the one pair known
    to be the same watch (rolex1/rolex2) from indistinguishable to a mutual
    nearest-neighbour match with a wide margin.
    """
    names = list(prints)
    M = np.array([prints[n] for n in names])
    M = M - M.mean(axis=0)
    M = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-12)
    return names, M


def main():
    paths = sorted(glob.glob("recordings/*.wav"))
    prints, meta = {}, {}
    for p in paths:
        x, fs = load(p)
        tracker, res = track(x, fs)
        fp = fingerprint(x, fs, tracker)
        name = os.path.basename(p).replace(".wav", "")
        if fp is None:
            print(f"{name:>22}  no lock — cannot fingerprint")
            continue
        prints[name] = fp
        meta[name] = (res["bph"], res["rate"], res["beat_error"])
        be = f"{res['beat_error']:.2f}" if res["beat_error"] is not None else "—"
        print(f"{name:>22}  bph {res['bph']}  rate {res['rate']:+6.1f}  beat err {be}")

    names = list(prints)
    print(f"\n=== pairwise similarity (cosine, 1.0 = identical) ===")
    print(f"{'':>22} " + " ".join(f"{n[:7]:>7}" for n in names))
    sims = np.zeros((len(names), len(names)))
    for i, a in enumerate(names):
        row = []
        for j, b in enumerate(names):
            s = float(np.dot(prints[a], prints[b]))
            sims[i, j] = s
            row.append(f"{s:7.2f}")
        print(f"{a:>22} " + " ".join(row))

    print("\n=== nearest neighbour for each recording ===")
    for i, a in enumerate(names):
        order = np.argsort(-sims[i])
        best = [j for j in order if j != i][0]
        second = [j for j in order if j != i][1]
        gap = sims[i, best] - sims[i, second]
        same_rate = "same-bph" if meta[a][0] == meta[names[best]][0] else "DIFF-bph"
        print(f"{a:>22} -> {names[best]:>22}  sim {sims[i, best]:.2f} "
              f"(next {sims[i, second]:.2f}, gap {gap:.2f}) {same_rate}")


if __name__ == "__main__":
    main()
