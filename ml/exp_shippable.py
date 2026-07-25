#!/usr/bin/env python3
"""
The version we can actually ship on the phone, validated against the reference.

`run_beats` in tg_lab.py proved the idea but leans on a full autocorrelation
period refinement, which is expensive and unnecessary live. This is the
incremental form intended for Timegrapher.swift:

  band-pass -> rectify -> decimate to 4 kHz envelope
  bootstrap a tick template from the loudest beat in the first seconds
  track each beat by matched filter, predicting from a running linear fit
    (so the period refines itself; no autocorrelation needed)
  once locked, refold at the fitted period for a clean template + retrack
  robust fit over all accepted beats -> rate
  scatter of independent 5 s block rates -> honest +/-

Reports rate, +/-, beat error, jitter and detection rate at increasing T, so we
can see convergence exactly as the app will.

  ./venv/bin/python exp_shippable.py recordings/*.wav
"""
import glob
import sys

import numpy as np
import soundfile as sf
from scipy.signal import lfilter

from tg_lab import rbj, select_band, robust_fit

ENV_RATE = 4000.0
TEMPLATE_MS = 14.0      # tick template length
PRE_MS = 2.0            # template starts this far before the peak
SEARCH_MS = 12.0        # +/- search window around the predicted beat


def envelope4k(x, fs, lo, hi, out_rate=ENV_RATE):
    b, a = rbj("hp", lo, fs)
    y = lfilter(b, a, x)
    b, a = rbj("lp", min(hi, fs / 2 - 500), fs)
    y = lfilter(b, a, y)
    y = np.abs(y)
    b, a = rbj("lp", out_rate / 3, fs)
    y = lfilter(b, a, y)
    d = max(1, int(round(fs / out_rate)))
    n = (len(y) // d) * d
    env = y[:n].reshape(-1, d).mean(axis=1)
    return env - np.median(env), fs / d


def norm_corr(env, tpl, positions):
    """Normalized correlation of tpl against env at each start position."""
    t = tpl - tpl.mean()
    tn = np.linalg.norm(t) + 1e-12
    out = np.empty(len(positions))
    L = len(t)
    for i, p in enumerate(positions):
        seg = env[p:p + L]
        if len(seg) < L:
            out[i] = -1
            continue
        s = seg - seg.mean()
        out[i] = np.dot(s, t) / (np.linalg.norm(s) * tn + 1e-12)
    return out


def parabolic(y0, y1, y2):
    den = y0 - 2 * y1 + y2
    return 0.5 * (y0 - y2) / den if den != 0 else 0.0


def bootstrap_template(env, er, period):
    """Template = the envelope around the loudest peak in the first 2 cycles."""
    L = int(TEMPLATE_MS * er / 1000)
    pre = int(PRE_MS * er / 1000)
    lead = env[: int(2 * period)]
    peak = int(np.argmax(lead))
    start = max(0, peak - pre)
    return env[start:start + L].copy(), start


def track(env, er, period, tpl, t0, n_from=0, prev=None):
    """Matched-filter every beat; predict from a running fit of prior beats."""
    search = int(SEARCH_MS * er / 1000)
    L = len(tpl)
    ks, ts, scores = [], [], []
    if prev is not None:
        ks, ts, scores = list(prev[0]), list(prev[1]), list(prev[2])
    k = n_from
    while True:
        # Predict: running fit once we have enough beats, else nominal period.
        if len(ks) >= 12:
            kk = np.array(ks[-200:], float)
            tt = np.array(ts[-200:], float)
            b = np.polyfit(kk, tt, 1)
            centre = b[0] * k + b[1]
        else:
            centre = (t0 + k * period) / er
        c = int(round(centre * er))
        a = c - search
        if a < 0:
            k += 1
            continue
        if c + search + L >= len(env):
            break
        corr = norm_corr(env, tpl, np.arange(a, a + 2 * search + 1))
        j = int(np.argmax(corr))
        frac = parabolic(corr[j - 1], corr[j], corr[j + 1]) if 0 < j < 2 * search else 0.0
        ks.append(k)
        ts.append((a + j + frac) / er)
        scores.append(float(corr[j]))
        k += 1
    return np.array(ks), np.array(ts), np.array(scores)


def refold(env, er, slope_s, t0_s, n_beats):
    """Average the envelope over all tracked beats at the fitted period."""
    P = slope_s * er
    L = int(np.floor(P))
    acc = np.zeros(L)
    cnt = 0
    for k in range(n_beats):
        s = int(round((t0_s + k * slope_s) * er))
        if s < 0 or s + L >= len(env):
            continue
        acc += env[s:s + L]
        cnt += 1
    if cnt == 0:
        return None
    wf = acc / cnt
    return wf - np.median(wf)


def beat_error_ms(wf, er):
    """Tic-to-toc spacing vs half the cycle, from the folded waveform.

    Found by circular autocorrelation near half the cycle (tg's approach) rather
    than by peak-picking: the escapement makes several sounds per beat, so the
    "second biggest peak" is often the tick's own drop, not the toc.
    """
    n = len(wf)
    w = wf - wf.mean()
    half = n // 2
    span = max(4, int(0.02 * er))
    lo, hi = max(1, half - span), min(n - 1, half + span)
    lags = np.arange(lo, hi + 1)
    ac = np.array([np.dot(w, np.roll(w, -int(L))) for L in lags])
    i = int(np.argmax(ac))
    frac = parabolic(ac[i - 1], ac[i], ac[i + 1]) if 0 < i < len(ac) - 1 else 0.0
    tic_to_toc = lags[i] + frac
    return abs(tic_to_toc - half) / er * 1000


def block_rates(k, t, keep, bph, tau=5.0, step=None):
    """Rate fitted independently in each block of `tau` seconds of beats."""
    k, t = k[keep], t[keep]
    nominal = 7200.0 / bph
    out = []
    if len(t) < 8:
        return np.array(out)
    step = step or tau
    t0 = t[0]
    while t0 <= t[-1] - tau:
        m = (t >= t0) & (t < t0 + tau)
        if m.sum() >= 8:
            b = np.polyfit(k[m], t[m], 1)[0]
            out.append((nominal / b - 1) * 86400)
        t0 += step
    return np.array(out)


def uncertainty_variants(k, t, keep, bph, resid):
    """Candidate published-+/- formulas, so we can pick by measured coverage."""
    nominal = 7200.0 / bph
    kk, rr = k[keep].astype(float), resid[keep]

    def sem(tau, step=None):
        b = block_rates(k, t, keep, bph, tau, step)
        if len(b) < 2:
            return np.nan
        n_ind = max(2.0, len(b) * (step or tau) / tau)
        return b.std(ddof=1) / np.sqrt(n_ind)

    # Textbook SE of the fitted slope, converted to s/day. Known to be far too
    # optimistic on its own (beat timings are correlated), but it scales the
    # right way with time so it is a useful floor once calibrated.
    if len(kk) > 4 and kk.std() > 0:
        se_slope = rr.std(ddof=1) / (np.sqrt(len(kk)) * kk.std())
        naive = 86400 * se_slope / nominal
    else:
        naive = np.nan

    s5, s10 = sem(5.0, 2.5), sem(10.0, 5.0)
    worst = 2 * np.nanmax([s5, s10])
    out = {"sem5": 2 * sem(5.0), "blocks": worst, "naive": naive}
    for c in (5, 10, 20, 30):
        out[f"blocks+{c}x"] = np.nanmax([worst, c * naive])
    return out


TRACK_BANDS = [(4000, 9000), (5000, 11000), (7000, 13000), (9000, 15000),
               (11000, 18000), (13000, 21000), (15000, 23000), (6000, 20000)]


def select_band_by_tracking(x, fs, manual=None, shortlist=3, probe_s=8.0):
    """Shortlist bands spectrally, then pick the one whose beats time cleanest.

    Spectral prominence tells us where the beat *line* is loudest, which is not
    the same as where each beat can be timed most precisely — on some couplings
    the loudest band is a ringing one whose tick shape smears. So probe the top
    few by actually tracking beats for a few seconds and keep the sharpest.
    """
    from tg_lab import dft_mag, envelope
    scan = x[: int(12 * fs)]
    cands = [manual] if manual else STANDARD_BPH_LOCAL
    scored = []
    for lo, hi in TRACK_BANDS:
        env, er = envelope(scan, fs, lo, hi)
        best = max(((dft_mag(env, er, b / 3600), b) for b in cands))
        f0 = best[1] / 3600
        nb = np.mean([dft_mag(env, er, f0 * k) for k in (0.6, 0.75, 1.2, 1.35, 1.6)])
        scored.append((best[0] / (nb + 1e-12), lo, hi, best[1]))
    scored.sort(reverse=True)

    probe = x[: int(probe_s * fs)]
    best = None
    for prom, lo, hi, bph in scored[:shortlist]:
        r = analyze(probe, fs, band=dict(lo=lo, hi=hi, bph=bph), _no_select=True)
        if r is None or r["detect"] < 0.85:
            continue
        # Rank by timing jitter; a smeared tick shows up here and nowhere else.
        key = (r["jitter_ms"], -r["score"])
        if best is None or key < best[0]:
            best = (key, dict(lo=lo, hi=hi, bph=bph, prom=prom))
    if best:
        return best[1]
    return dict(lo=scored[0][1], hi=scored[0][2], bph=scored[0][3], prom=scored[0][0])


STANDARD_BPH_LOCAL = [18000, 19800, 21600, 25200, 28800, 36000]


def analyze(x, fs, up_to=None, band=None, manual=None, _no_select=False):
    if up_to:
        x = x[: int(up_to * fs)]
    sel = band or (select_band(x, fs, manual) if _no_select
                   else select_band_by_tracking(x, fs, manual))
    bph = sel["bph"]
    env, er = envelope4k(x, fs, sel["lo"], sel["hi"])
    nominal = 7200.0 / bph
    period = nominal * er
    tpl, start = bootstrap_template(env, er, period)
    k, t, sc = track(env, er, period, tpl, start)
    if len(k) < 8:
        return None
    slope, resid, keep = robust_fit(k, t, sc > 0.3)
    if keep.sum() < 8:
        return None

    # Refold at the fitted period for a clean template, then retrack once.
    a = np.polyfit(k[keep], t[keep], 1)
    wf = refold(env, er, a[0], a[1], int(k[keep].max()) + 1)
    be_ms = float("nan")
    if wf is not None and len(wf) > 8:
        L = int(TEMPLATE_MS * er / 1000)
        pre = int(PRE_MS * er / 1000)
        peak = int(np.argmax(wf))
        tpl2 = np.roll(wf, -(peak - pre))[:L]
        be_ms = beat_error_ms(wf, er)
        k, t, sc = track(env, er, period, tpl2, start)
        slope, resid, keep = robust_fit(k, t, sc > 0.3)
        if keep.sum() < 8:
            return None

    rate = (nominal / slope - 1) * 86400
    blocks = block_rates(k, t, keep, bph)
    if len(blocks) >= 2:
        unc = blocks.std(ddof=1) / np.sqrt(len(blocks))
    else:
        unc = float("nan")
    variants = uncertainty_variants(k, t, keep, bph, resid)
    return dict(rate=rate, bph=bph, unc=unc, n_blocks=len(blocks), variants=variants,
                jitter_ms=float(resid[keep].std(ddof=1) * 1000),
                detect=float(keep.sum() / len(keep)),
                score=float(np.median(sc[keep])), beat_error_ms=be_ms,
                n_beats=int(keep.sum()), blocks=blocks)


def main(paths):
    ratios = []
    for p in paths:
        x, fs = sf.read(p, always_2d=False)
        if x.ndim == 2:
            x = x.mean(axis=1)
        x = x.astype(np.float64)
        fs = float(fs)
        dur = len(x) / fs
        band = select_band_by_tracking(x, fs)
        final = analyze(x, fs, band=band)
        if final is None:
            print(f"{p}: FAILED")
            continue
        print(f"\n=== {p}  {dur:.0f}s  bph {final['bph']}  band "
              f"{band['lo']/1000:.0f}-{band['hi']/1000:.0f}k  final {final['rate']:+.2f} ===")
        print(f"{'T':>5} {'rate':>8} {'+/-2sem':>8} {'err':>7} {'jit_ms':>7} "
              f"{'detect':>7} {'score':>6} {'BE_ms':>6}")
        for T in [6, 8, 10, 12, 15, 20, 25, 30, 40, 50, 60]:
            if T > dur - 0.5:
                break
            r = analyze(x, fs, up_to=T, band=band)
            if r is None:
                print(f"{T:5.0f}   (no lock)")
                continue
            err = r["rate"] - final["rate"]
            pub = 2 * r["unc"] if np.isfinite(r["unc"]) else np.nan
            if T < dur - 5:
                row = {"err": abs(err)}
                row.update(r["variants"])
                ratios.append(row)
            print(f"{T:5.0f} {r['rate']:+8.2f} {pub:8.2f} {err:+7.2f} "
                  f"{r['jitter_ms']:7.3f} {r['detect']:7.2f} {r['score']:6.2f} "
                  f"{r['beat_error_ms']:6.2f}")
    if ratios:
        print("\n=== +/- estimator comparison (|err| / published, floor 0.5) ===")
        keys = [k for k in ratios[0] if k != "err"]
        for key in keys:
            v = np.array([r["err"] / max(r[key], 0.5) for r in ratios
                          if np.isfinite(r[key])])
            if not len(v):
                continue
            print(f"{key:>14}: covered {100*(v<=1).mean():3.0f}%  median {np.median(v):5.2f}"
                  f"  p90 {np.percentile(v,90):6.2f}  max {v.max():7.2f}  n={len(v)}")


if __name__ == "__main__":
    ps = []
    for a in sys.argv[1:]:
        ps.extend(sorted(glob.glob(a)))
    main(ps)
