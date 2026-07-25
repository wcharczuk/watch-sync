#!/usr/bin/env python3
"""
Timegrapher lab: reproduce the app's live behaviour offline and try better DSP.

Two things we're chasing:

  1. The reading *collapses* — a Hamilton read -30 s/day early on and settled at
     ~0 after 60 s. Is that estimator variance, a transient, or a bias?
  2. Confidence is hard to come by. What actually predicts the error?

`current`  — faithful port of Timegrapher.swift (band scan + lock-in at f0).
`beats`    — tg-style: envelope -> autocorrelation period -> folded template ->
             matched-filter per-beat timings -> robust fit. Gives per-beat
             residuals, so the uncertainty is measurable rather than guessed.

Usage:
  ./venv/bin/python tg_lab.py recordings/*.wav            # convergence table
  ./venv/bin/python tg_lab.py recordings/new2.wav --plot out.png
"""
import argparse
import glob
import os

import numpy as np
import soundfile as sf
from scipy.signal import lfilter

STANDARD_BPH = [18000, 19800, 21600, 25200, 28800, 36000]
CANDIDATE_BANDS = [(2000, 6000), (4000, 9000), (5000, 11000), (7000, 13000),
                   (9000, 15000), (11000, 18000), (13000, 21000)]
ENV_RATE = 1000.0


# --------------------------------------------------------------------------
# shared DSP
# --------------------------------------------------------------------------

def rbj(kind, fc, fs, q=0.707):
    w0 = 2 * np.pi * fc / fs
    cw, sw = np.cos(w0), np.sin(w0)
    alpha = sw / (2 * q)
    a0 = 1 + alpha
    if kind == "hp":
        b = np.array([(1 + cw) / 2, -(1 + cw), (1 + cw) / 2]) / a0
    else:
        b = np.array([(1 - cw) / 2, (1 - cw), (1 - cw) / 2]) / a0
    a = np.array([1.0, (-2 * cw) / a0, (1 - alpha) / a0])
    return b, a


def envelope(x, fs, lo, hi, out_rate=ENV_RATE, square=True):
    """Band-pass -> rectify -> decimate. Matches Swift's envelope()."""
    b, a = rbj("hp", lo, fs)
    y = lfilter(b, a, x)
    b, a = rbj("lp", min(hi, fs / 2 - 500), fs)
    y = lfilter(b, a, y)
    y = y * y if square else np.abs(y)
    d = max(1, int(round(fs / out_rate)))
    n = (len(y) // d) * d
    env = y[:n].reshape(-1, d).mean(axis=1)
    return env - env.mean(), fs / d


def dft_mag(e, fs, f):
    n = len(e)
    w = 2 * np.pi * f / fs * np.arange(n)
    return abs(np.dot(e, np.exp(-1j * w))) / n


# --------------------------------------------------------------------------
# current algorithm (Timegrapher.swift)
# --------------------------------------------------------------------------

def select_band(x, fs, manual=None, scan_s=12.0):
    scan = x[:int(scan_s * fs)]
    cands = [manual] if manual else STANDARD_BPH
    best, best_prom = None, -1
    for lo, hi in CANDIDATE_BANDS:
        env, er = envelope(scan, fs, lo, hi)
        mags = sorted(((dft_mag(env, er, b / 3600), b) for b in cands), reverse=True)
        bph = mags[0][1]
        f0 = bph / 3600
        nb = np.mean([dft_mag(env, er, f0 * k) for k in (0.6, 0.75, 1.2, 1.35, 1.6)])
        prom = mags[0][0] / (nb + 1e-12)
        if prom > best_prom:
            best_prom = prom
            best = dict(lo=lo, hi=hi, bph=bph, prom=prom,
                        sep=mags[0][0] / max(mags[1][0], 1e-12) if len(mags) > 1 else 999)
    return best


def lock_in(env, fs, f0):
    """Complex-demodulate at f0, unwrap phase, amplitude-weighted slope fit."""
    n = len(env)
    t = np.arange(n) / fs
    z = env * np.exp(-1j * 2 * np.pi * f0 * t)
    b, a = rbj("lp", 1.5, fs)
    z = lfilter(b, a, z.real) + 1j * lfilter(b, a, z.imag)
    lo, hi = int(0.1 * n), int(0.9 * n)
    if hi - lo < fs:
        return None
    ph = np.unwrap(np.angle(z[lo:hi]))
    tt = t[lo:hi]
    amp = np.abs(z[lo:hi])
    w = amp / amp.sum()
    mx, my = np.dot(w, tt), np.dot(w, ph)
    slope = np.dot(w, (tt - mx) * (ph - my)) / np.dot(w, (tt - mx) ** 2)
    return dict(slope=slope, amp_snr=amp.mean() / (amp.std() + 1e-12),
                phases=ph, times=tt)


def run_current(x, fs, up_to=None, manual=None):
    if up_to:
        x = x[:int(up_to * fs)]
    sel = select_band(x, fs, manual)
    env, er = envelope(x, fs, sel["lo"], sel["hi"])
    f0 = sel["bph"] / 3600
    li = lock_in(env, er, f0)
    if li is None:
        return None
    rate = 86400 * li["slope"] / (2 * np.pi * f0)
    return dict(rate=rate, bph=sel["bph"], prom=sel["prom"],
                amp_snr=li["amp_snr"], f0=f0, li=li)


# --------------------------------------------------------------------------
# beats algorithm (tg-style per-beat matched filter)
# --------------------------------------------------------------------------

def noise_suppress(x, fs, factor=3.0):
    """tg's noise suppressor: zero windows whose energy exceeds the median."""
    w = max(1, int(fs / 50))
    p = np.convolve(x * x, np.ones(w) / w, mode="same")
    step = max(1, int(fs / 2))
    peaks = [p[i:i + step].max() for i in range(0, len(p) - step, step)]
    if not peaks:
        return x
    k = np.median(peaks)
    out = x.copy()
    out[p > factor * k] = 0.0
    return out


def beat_envelope(x, fs, lo, hi, out_rate=4000.0, suppress=True):
    """Band-pass -> (noise suppress) -> rectify -> smooth -> decimate."""
    b, a = rbj("hp", lo, fs)
    y = lfilter(b, a, x)
    b, a = rbj("lp", min(hi, fs / 2 - 500), fs)
    y = lfilter(b, a, y)
    if suppress:
        y = noise_suppress(y, fs)
    y = np.abs(y)
    # Smooth to the envelope timescale before decimating (anti-alias).
    b, a = rbj("lp", out_rate / 3, fs)
    y = lfilter(b, a, y)
    d = max(1, int(round(fs / out_rate)))
    n = (len(y) // d) * d
    env = y[:n].reshape(-1, d).mean(axis=1)
    return env - np.median(env), fs / d


def refine_period(env, er, p0, max_cycles=None):
    """tg-style: autocorrelation peaks at k*p0 for increasing k, averaged.

    Returns (period_samples, sigma) where sigma is the scatter of the per-cycle
    estimates — tg's lock-quality measure.
    """
    n = len(env)
    e = env - env.mean()
    ac = np.correlate(e, e, mode="full")[n - 1:]
    ests = []
    k = 1
    while True:
        lag = p0 * k
        delta = max(4, int(0.02 * er))
        a, b = int(lag - delta), int(lag + delta)
        if b > n * 2 // 3 or (max_cycles and k > max_cycles):
            break
        seg = ac[a:b]
        if len(seg) < 3:
            break
        i = a + int(np.argmax(seg))
        # parabolic refine
        if 0 < i < n - 1:
            y0, y1, y2 = ac[i - 1], ac[i], ac[i + 1]
            den = y0 - 2 * y1 + y2
            frac = 0.5 * (y0 - y2) / den if den != 0 else 0.0
        else:
            frac = 0.0
        ests.append((i + frac) / k)
        k += 1
    if not ests:
        return p0, np.inf
    ests = np.array(ests)
    # Weight later cycles more: they're intrinsically more precise.
    return float(ests.mean()), float(ests.std(ddof=1)) if len(ests) > 1 else np.inf


def fold(env, period, phase=0.0):
    """Fold the envelope at `period` samples (trimmed mean, tg-style)."""
    wf = int(np.floor(period))
    n = len(env)
    cols = []
    for i in range(wf):
        idx = np.round(i + phase + np.arange(0, (n - i) / period) * period).astype(int)
        idx = idx[idx < n]
        cols.append(idx)
    out = np.zeros(wf)
    for i, idx in enumerate(cols):
        if len(idx) == 0:
            continue
        v = np.sort(env[idx])
        # drop the top quintile (noise bursts)
        keep = v[: max(1, len(v) - max(1, len(v) // 5))]
        out[i] = keep.mean()
    return out - np.median(out)


def track_beats(env, er, period, template, t0_idx, search_ms=20.0):
    """Matched-filter each predicted beat; return (beat_index, time_seconds)."""
    tlen = len(template)
    tpl = template - template.mean()
    tnorm = np.linalg.norm(tpl) + 1e-12
    search = int(search_ms * er / 1000)
    n = len(env)
    idx, times, scores = [], [], []
    k = 0
    while True:
        center = t0_idx + k * period
        if center + tlen + search >= n:
            break
        a = int(center) - search
        if a < 0:
            k += 1
            continue
        best, best_off = -np.inf, 0
        corr = np.empty(2 * search + 1)
        for j in range(2 * search + 1):
            seg = env[a + j: a + j + tlen]
            corr[j] = np.dot(seg, tpl) / (np.linalg.norm(seg) + 1e-12) / tnorm
        best_off = int(np.argmax(corr))
        best = corr[best_off]
        # parabolic sub-sample refine
        if 0 < best_off < 2 * search:
            y0, y1, y2 = corr[best_off - 1], corr[best_off], corr[best_off + 1]
            den = y0 - 2 * y1 + y2
            frac = 0.5 * (y0 - y2) / den if den != 0 else 0.0
        else:
            frac = 0.0
        idx.append(k)
        times.append((a + best_off + frac) / er)
        scores.append(best)
        k += 1
    return np.array(idx), np.array(times), np.array(scores)


def robust_fit(k, t, w=None, iters=4):
    """Iteratively reweighted linear fit t = a + b*k. Returns (b, resid, keep)."""
    keep = np.ones(len(k), bool) if w is None else w.copy()
    b = a = 0.0
    for _ in range(iters):
        if keep.sum() < 4:
            break
        A = np.vstack([np.ones(keep.sum()), k[keep]]).T
        coef, *_ = np.linalg.lstsq(A, t[keep], rcond=None)
        a, b = coef
        r = t - (a + b * k)
        s = 1.4826 * np.median(np.abs(r[keep] - np.median(r[keep]))) + 1e-9
        keep = np.abs(r) < 3 * s
    r = t - (a + b * k)
    return b, r, keep


def run_beats(x, fs, up_to=None, manual=None, band=None):
    if up_to:
        x = x[:int(up_to * fs)]
    sel = select_band(x, fs, manual) if band is None else band
    bph = sel["bph"]
    env, er = beat_envelope(x, fs, sel["lo"], sel["hi"])
    p0 = er * 7200.0 / bph          # samples per beat (half-period of tic-toc)
    period, sigma = refine_period(env, er, p0)
    # fold at 2 beats: tic and toc differ, so the true repeat is 2 beats
    wf = fold(env, 2 * period)
    # template = the strongest half of the folded waveform
    half = len(wf) // 2
    peak = int(np.argmax(wf))
    tlen = max(8, int(0.012 * er))          # 12 ms template
    start = max(0, peak - int(0.002 * er))
    tpl = np.roll(wf, -start)[:tlen]
    # first beat: locate the template in the first 2 periods of the envelope
    lead = env[: int(3 * period)]
    c = np.correlate(lead, tpl - tpl.mean(), mode="valid")
    t0_idx = int(np.argmax(c))
    k, t, sc = track_beats(env, er, period, tpl, t0_idx)
    if len(k) < 10:
        return None
    slope, resid, keep = robust_fit(k, t, sc > np.percentile(sc, 10))
    nominal = 7200.0 / bph
    rate = (nominal / slope - 1) * 86400
    # honest slope SE from the per-beat residual scatter
    kk = k[keep]
    sd = resid[keep].std(ddof=1)
    se_slope = sd / (np.std(kk) * np.sqrt(len(kk)) + 1e-12)
    se_rate = 86400 * nominal / slope**2 * se_slope
    # beat error: alternating tic/toc residual asymmetry (ms)
    even = resid[keep][kk[keep.sum() * 0:] % 2 == 0] if False else resid[keep][kk % 2 == 0]
    odd = resid[keep][kk % 2 == 1]
    be = abs(even.mean() - odd.mean()) * 1000 if len(even) and len(odd) else float("nan")
    return dict(rate=rate, bph=bph, period=period / er, sigma=sigma / max(period, 1),
                jitter_ms=sd * 1000, n_beats=int(keep.sum()), n_total=len(k),
                se_rate=se_rate, beat_error_ms=be, score=float(np.median(sc)),
                resid=resid, keep=keep, k=k, t=t, er=er, env=env, tpl=tpl, wf=wf)


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------

def convergence(path, manual=None):
    fs, x = sf.read(path, always_2d=False), None
    x, fs = sf.read(path, always_2d=False)
    if x.ndim == 2:
        x = x.mean(axis=1)
    x = x.astype(np.float64)
    fs = float(sf.info(path).samplerate)
    dur = len(x) / fs
    print(f"\n=== {os.path.basename(path)}  {dur:.1f}s ===")
    print(f"{'t':>6} | {'current':>9} {'ampSNR':>7} | {'beats':>9} {'±':>6} "
          f"{'jit_ms':>7} {'sigma':>8} {'beats#':>7} {'BE_ms':>6}")
    for t in [8, 12, 16, 20, 30, 40, 50, 60, 80, 100, 120, 150, 180]:
        if t > dur:
            break
        cur = run_current(x, fs, up_to=t, manual=manual)
        bt = run_beats(x, fs, up_to=t, manual=manual)
        c1 = f"{cur['rate']:+9.1f}" if cur else "        -"
        c2 = f"{cur['amp_snr']:7.2f}" if cur else "      -"
        if bt:
            b1 = f"{bt['rate']:+9.1f}"
            b2 = f"{bt['se_rate']:6.2f}"
            b3 = f"{bt['jitter_ms']:7.2f}"
            b4 = f"{bt['sigma']:8.5f}"
            b5 = f"{bt['n_beats']:3d}/{bt['n_total']:3d}"
            b6 = f"{bt['beat_error_ms']:6.2f}"
        else:
            b1 = b2 = b3 = b4 = b5 = b6 = "     -"
        print(f"{t:6.0f} | {c1} {c2} | {b1} {b2} {b3} {b4} {b5:>7} {b6}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("wavs", nargs="+")
    ap.add_argument("--bph", type=int, default=None)
    args = ap.parse_args()
    paths = []
    for w in args.wavs:
        paths.extend(sorted(glob.glob(w)))
    for p in paths:
        convergence(p, manual=args.bph)


if __name__ == "__main__":
    main()
