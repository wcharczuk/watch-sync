#!/usr/bin/env python3
"""
Offline tuning harness for the acoustic timegrapher.

This is a faithful port of WatchSync/WatchSync/Timegrapher.swift. Feed it a raw
WAV exported from the app and it reproduces the app's numbers, so we can sweep
DSP parameters against real recordings and port the winners back to Swift.

Pull a recording off the device:
  - In the app, take a measurement, Stop, tap the Export (share) button, and
    AirDrop / save the `watch-<ts>.wav` (+ `.json` sidecar) to this ml/ folder.

Usage:
  ./venv/bin/python tune_timegrapher.py watch-1721880000.wav
  ./venv/bin/python tune_timegrapher.py watch-*.wav --reference -3   # known s/day
  ./venv/bin/python tune_timegrapher.py watch-*.wav --bph 28800      # force rate
  ./venv/bin/python tune_timegrapher.py watch-*.wav --sweep
  ./venv/bin/python tune_timegrapher.py watch-*.wav --plot out.png

The `.json` sidecar (if present next to the WAV) is compared against this port
to catch Swift-vs-Python drift.
"""
import argparse
import json
import os
import sys
from dataclasses import dataclass, replace

import numpy as np
import soundfile as sf

# Standard mechanical beat rates (bph, beats/second) — mirrors Timegrapher.swift.
STANDARD_RATES = [
    (18000, 5.0), (19800, 5.5), (21600, 6.0),
    (25200, 7.0), (28800, 8.0), (36000, 10.0),
]


@dataclass
class Params:
    """DSP knobs. Defaults mirror the current Swift constants."""
    highpass_cutoff: float = 1000.0
    highpass_q: float = 0.707
    env_target_rate: float = 8000.0
    threshold_frac: float = 0.30     # threshold = median + frac*(p95-median)
    autocorr_fmin: float = 4.0       # beat-freq search window (Hz)
    autocorr_fmax: float = 11.0
    fundamental_frac: float = 0.40   # first autocorr peak >= frac*global max wins
    search_frac: float = 0.40        # tracking window = ±frac*period
    anchor_frac: float = 1.20        # anchor peak search span (periods)
    jitter_denom: float = 0.15       # fitQuality = 1 - (jitter/period)/denom


# ----------------------------------------------------------------------------
# Signal pipeline
# ----------------------------------------------------------------------------

def read_wav(path):
    """Return (sample_rate, mono float64 samples)."""
    data, fs = sf.read(path, always_2d=False)
    if data.ndim == 2:
        data = data.mean(axis=1)
    return float(fs), data.astype(np.float64)


def rbj_highpass(fs, cutoff, q):
    """RBJ cookbook high-pass biquad, normalized. Matches Swift configureHighPass."""
    w0 = 2 * np.pi * cutoff / fs
    cosw, sinw = np.cos(w0), np.sin(w0)
    alpha = sinw / (2 * q)
    a0 = 1 + alpha
    b = np.array([(1 + cosw) / 2, -(1 + cosw), (1 + cosw) / 2]) / a0
    a = np.array([1.0, (-2 * cosw) / a0, (1 - alpha) / a0])
    return b, a


def build_envelope(samples, fs, p: Params):
    """High-pass -> square -> decimate to ~env_target_rate. Returns (env, env_rate)."""
    from scipy.signal import lfilter
    b, a = rbj_highpass(fs, p.highpass_cutoff, p.highpass_q)
    y = lfilter(b, a, samples)
    decim = max(1, int(round(fs / p.env_target_rate)))
    energy = y * y
    n = (len(energy) // decim) * decim
    env = energy[:n].reshape(-1, decim).mean(axis=1)
    return env.astype(np.float64), fs / decim


def estimate_period_samples(env, env_rate, p: Params, manual_bph=None):
    """Autocorrelation beat-period estimate (env samples), with subharmonic check."""
    if manual_bph is not None:
        return env_rate * 3600.0 / manual_bph
    n = min(len(env), int(4 * env_rate))
    lag_min = int(env_rate / p.autocorr_fmax)
    lag_max = min(int(env_rate / p.autocorr_fmin), n - 1)
    if lag_max <= lag_min:
        return None
    e = env[:n] - env[:n].mean()
    corr = np.zeros(lag_max + 1)
    for lag in range(lag_min, lag_max + 1):
        m = n - lag
        corr[lag] = np.dot(e[:m], e[lag:lag + m]) / m
    best_lag = int(np.argmax(corr[lag_min:lag_max + 1])) + lag_min
    best_val = corr[best_lag]
    if best_val <= 0:
        return None
    # The watch makes a sound every beat, so the true fundamental is the
    # SMALLEST period with a strong autocorrelation peak. Beat error or
    # tick/tock amplitude differences can make the full-oscillation period (2×)
    # the global max, so we scan upward for the first strong local peak instead
    # of taking the global argmax.
    thr = p.fundamental_frac * best_val
    chosen = best_lag
    for lag in range(lag_min + 1, lag_max):
        if corr[lag] >= thr and corr[lag] >= corr[lag - 1] and corr[lag] >= corr[lag + 1]:
            chosen = lag
            break
    return _parabolic(corr, chosen)


def _parabolic(y, k):
    if k < 1 or k + 1 >= len(y):
        return float(k)
    ym1, y0, yp1 = y[k - 1], y[k], y[k + 1]
    denom = ym1 - 2 * y0 + yp1
    if abs(denom) < 1e-12:
        return float(k)
    return k + max(-1.0, min(1.0, 0.5 * (ym1 - yp1) / denom))


def track_beats(env, period, threshold, env_rate, p: Params):
    """Period-locked peak tracking. Returns list of (index, time_seconds)."""
    n = len(env)
    if period <= 2 or n <= 2 * period:
        return []
    beats = []
    first_hi = min(n - 2, int(p.anchor_frac * period))
    anchor = 1 + int(np.argmax(env[1:first_hi + 1]))
    ref = _parabolic(env, anchor)
    beats.append((0, ref / env_rate))
    search_half = p.search_frac * period
    slot = 0
    while True:
        slot += 1
        predicted = ref + period
        if predicted > n - 2:
            break
        lo = max(1, int(predicted - search_half))
        hi = min(n - 2, int(predicted + search_half))
        if lo >= hi:
            break
        k = lo + int(np.argmax(env[lo:hi + 1]))
        if env[k] > threshold:
            refined = _parabolic(env, k)
            beats.append((slot, refined / env_rate))
            ref = refined
        else:
            ref = predicted
    return beats


def snap_octave(detected_period, manual_bph):
    """Given the detected-train period (s), find (bph, f) where the movement's
    single beat is 3600/bph and we detected every f-th beat (f in {1,2}).

    This makes the rate correct even when the tracker locks onto the full
    oscillation (2×) — common when tick and tock differ in loudness. Returns
    (bph, f, nominal_detected_period).
    """
    if manual_bph is not None:
        # For a manual rate, pick the f whose predicted period is closest.
        cands = [(f, f * 3600.0 / manual_bph) for f in (1, 2)]
        f, nd = min(cands, key=lambda c: abs(detected_period - c[1]))
        return manual_bph, f, nd
    best = None
    for bph, _ in STANDARD_RATES:
        for f in (1, 2):
            nominal_detected = f * 3600.0 / bph
            relerr = abs(detected_period - nominal_detected) / nominal_detected
            if best is None or relerr < best[0]:
                best = (relerr, bph, f, nominal_detected)
    return best[1], best[2], best[3]


def _robust_fit(idx, t):
    """OLS of t vs idx with one MAD-based outlier rejection pass.
    Returns (slope, intercept, residual_std, sii, keep_mask)."""
    def fit(ii, tt):
        mi, mt = ii.mean(), tt.mean()
        sii = float(np.sum((ii - mi) ** 2))
        sit = float(np.sum((ii - mi) * (tt - mt)))
        slope = sit / sii
        intercept = mt - slope * mi
        return slope, intercept, sii
    slope, intercept, sii = fit(idx, t)
    resid = t - (intercept + slope * idx)
    mad = np.median(np.abs(resid - np.median(resid)))
    keep = np.abs(resid - np.median(resid)) <= max(5 * mad, 1e-6)
    if keep.sum() >= max(8, int(0.6 * len(idx))) and keep.sum() < len(idx):
        slope, intercept, sii = fit(idx[keep], t[keep])
    else:
        keep = np.ones(len(idx), bool)
    resid = (t - (intercept + slope * idx))[keep]
    residual_std = float(np.sqrt(np.sum(resid ** 2) / max(keep.sum() - 2, 1)))
    return slope, intercept, residual_std, sii, keep


def analyze(samples, fs, p: Params, manual_bph=None):
    env, env_rate = build_envelope(samples, fs, p)
    if len(env) < env_rate * 1.5:
        return None
    median = float(np.median(env))
    p95 = float(np.percentile(env, 95))
    threshold = median + p.threshold_frac * max(p95 - median, 1e-12)

    period = estimate_period_samples(env, env_rate, p, manual_bph)
    if period is None:
        return None
    beats = track_beats(env, period, threshold, env_rate, p)
    if len(beats) < 8:
        return None

    idx = np.array([b[0] for b in beats], float)
    t = np.array([b[1] for b in beats], float)
    slope, intercept, residual_std, sii, keep = _robust_fit(idx, t)
    if sii <= 0 or slope <= 0:
        return None

    # Resolve the octave: rate is scale-invariant, so a 2× (full-oscillation)
    # lock still yields the correct bph and rate.
    nominal_bph, octave_f, nominal_detected = snap_octave(slope, manual_bph)
    measured_bps = octave_f / slope          # true single-beat rate
    rate = 86400.0 * (nominal_detected - slope) / slope
    se_slope = residual_std / np.sqrt(sii)
    rate_unc = 86400.0 * nominal_detected / (slope * slope) * se_slope

    # Beat error only makes sense when we detect every beat (f == 1).
    beat_error_ms = 0.0
    if octave_f == 1:
        ev, od = [], []
        for j in range(1, len(beats)):
            if beats[j][0] - beats[j - 1][0] == 1:
                d = beats[j][1] - beats[j - 1][1]
                (ev if beats[j - 1][0] % 2 == 0 else od).append(d)
        if ev and od:
            beat_error_ms = abs(np.mean(ev) - np.mean(od)) * 1000

    span = (beats[-1][0] - beats[0][0]) + 1
    detection_rate = min(1.0, len(beats) / max(span, 1))
    jitter = residual_std / slope
    fit_quality = max(0.0, min(1.0, 1.0 - jitter / p.jitter_denom))
    confidence = detection_rate * fit_quality

    return {
        "beatsPerHour": nominal_bph,
        "measuredBeatsPerSecond": measured_bps,
        "rateSecondsPerDay": rate,
        "rateUncertainty": rate_unc,
        "beatErrorMs": beat_error_ms,
        "confidence": confidence,
        "detectionRate": detection_rate,
        "residualJitterMs": residual_std * 1000,
        "beatCount": len(beats),
        "elapsedSeconds": len(env) / env_rate,
        "_env": env, "_env_rate": env_rate, "_beats": beats,
        "_threshold": threshold, "_period": period,
        "_slope": slope, "_intercept": intercept, "_nominal_period": nominal_detected,
    }


# ----------------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------------

def print_result(r, label=""):
    if r is None:
        print(f"{label}  no lock (too little / too noisy audio)")
        return
    print(f"{label}")
    print(f"  rate         {r['rateSecondsPerDay']:+.2f} s/day  "
          f"(± {r['rateUncertainty']:.2f})")
    print(f"  beat rate    {r['beatsPerHour']} bph   "
          f"(measured {r['measuredBeatsPerSecond']:.4f}/s)")
    print(f"  beat error   {r['beatErrorMs']:.2f} ms")
    print(f"  beats        {r['beatCount']}  detected {r['detectionRate']*100:.0f}%  "
          f"jitter {r['residualJitterMs']:.2f} ms")
    print(f"  confidence   {r['confidence']*100:.0f}%   "
          f"elapsed {r['elapsedSeconds']:.1f}s")


def compare_sidecar(r, wav_path):
    side = os.path.splitext(wav_path)[0] + ".json"
    if not os.path.exists(side) or r is None:
        return
    with open(side) as f:
        app = json.load(f)
    print("  --- Swift vs Python parity ---")
    for k in ["beatsPerHour", "measuredBeatsPerSecond", "rateSecondsPerDay",
              "beatErrorMs", "beatCount", "detectionRate", "residualJitterMs"]:
        if k in app:
            a, py = app[k], r[k]
            try:
                d = f"Δ {abs(float(a)-float(py)):.4g}"
            except (TypeError, ValueError):
                d = ""
            print(f"  {k:24s} swift={a}   py={py:.6g}  {d}" if isinstance(py, float)
                  else f"  {k:24s} swift={a}   py={py}  {d}")


def sweep(samples, fs, base: Params, manual_bph, reference):
    grid_cut = [500, 800, 1000, 1500, 2000]
    grid_thr = [0.15, 0.20, 0.30, 0.40]
    grid_sf = [0.30, 0.40]
    rows = []
    for c in grid_cut:
        for th in grid_thr:
            for sf_ in grid_sf:
                p = replace(base, highpass_cutoff=c, threshold_frac=th, search_frac=sf_)
                r = analyze(samples, fs, p, manual_bph)
                if r is None:
                    continue
                err = (abs(r["rateSecondsPerDay"] - reference)
                       if reference is not None else None)
                rows.append((c, th, sf_, r, err))
    if not rows:
        print("sweep: no configuration produced a lock")
        return
    # Sort by reference error if given, else by (low jitter, high detection).
    if reference is not None:
        rows.sort(key=lambda x: x[4])
    else:
        rows.sort(key=lambda x: (x[3]["residualJitterMs"], -x[3]["detectionRate"]))
    print(f"\n{'cutoff':>7} {'thr':>5} {'srch':>5} {'rate':>9} {'err':>7} "
          f"{'bph':>6} {'jit':>6} {'det':>5} {'conf':>5}")
    for c, th, sf_, r, err in rows[:15]:
        es = f"{err:6.2f}" if err is not None else "   -  "
        print(f"{c:7.0f} {th:5.2f} {sf_:5.2f} {r['rateSecondsPerDay']:+9.2f} {es} "
              f"{r['beatsPerHour']:6d} {r['residualJitterMs']:6.2f} "
              f"{r['detectionRate']*100:4.0f}% {r['confidence']*100:4.0f}%")


def plot(r, out_path):
    if r is None:
        print("plot: nothing to draw")
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    env, env_rate = r["_env"], r["_env_rate"]
    beats = r["_beats"]
    t_env = np.arange(len(env)) / env_rate
    fig, ax = plt.subplots(3, 1, figsize=(11, 9))

    # 1) Envelope + detected beats + threshold (first 6 s)
    lim = int(min(len(env), 6 * env_rate))
    ax[0].plot(t_env[:lim], env[:lim], lw=0.5)
    ax[0].axhline(r["_threshold"], color="r", ls="--", lw=0.7, label="threshold")
    for i, tt in beats:
        if tt <= 6:
            ax[0].axvline(tt, color="g", alpha=0.3, lw=0.5)
    ax[0].set_title("Energy envelope + detected beats (first 6 s)")
    ax[0].legend(loc="upper right")

    # 2) Beat-strip: residual vs nominal period (ms) over time -> the two lines
    idx = np.array([b[0] for b in beats])
    tt = np.array([b[1] for b in beats])
    resid_ms = (tt - r["_intercept"] - idx * r["_nominal_period"]) * 1000
    half = r["_nominal_period"] * 1000
    resid_ms = ((resid_ms + half / 2) % half) - half / 2
    colors = ["g" if i % 2 == 0 else "c" for i in idx]
    ax[1].scatter(tt, resid_ms, s=6, c=colors)
    ax[1].set_title("Beat strip (residual vs nominal period, ms)")
    ax[1].set_xlabel("time (s)")

    # 3) Inter-beat interval histogram (ms)
    d = np.diff(tt) * 1000
    ax[2].hist(d, bins=60)
    ax[2].axvline(r["_slope"] * 1000, color="r", ls="--",
                  label=f"period {r['_slope']*1000:.2f} ms")
    ax[2].set_title("Inter-beat intervals (ms)")
    ax[2].legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    print(f"plot saved -> {out_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("wav", nargs="+", help="raw WAV file(s) exported from the app")
    ap.add_argument("--bph", type=int, default=None, help="force nominal beat rate")
    ap.add_argument("--reference", type=float, default=None,
                    help="known true rate (s/day) for sweep error ranking")
    ap.add_argument("--sweep", action="store_true", help="grid-search DSP params")
    ap.add_argument("--plot", metavar="OUT.png", default=None, help="write diagnostic plot")
    args = ap.parse_args()

    base = Params()
    for path in args.wav:
        if not os.path.exists(path):
            print(f"skip (missing): {path}", file=sys.stderr)
            continue
        fs, samples = read_wav(path)
        dur = len(samples) / fs
        print(f"\n=== {os.path.basename(path)}  ({dur:.1f}s @ {fs:.0f} Hz) ===")
        r = analyze(samples, fs, base, args.bph)
        print_result(r, "default params:")
        compare_sidecar(r, path)
        if args.sweep:
            sweep(samples, fs, base, args.bph, args.reference)
        if args.plot:
            plot(r, args.plot)


if __name__ == "__main__":
    main()
