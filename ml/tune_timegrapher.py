#!/usr/bin/env python3
"""
Offline harness for the acoustic timegrapher.

A faithful port of WatchSync/WatchSync/Timegrapher.swift: same band probe, same
matched-filter beat tracking, same uncertainty, same stage machine. Feed it a raw
WAV exported from the app and it reproduces what the app would have shown —
including the *live* sequence, so we can check that the reading converges instead
of collapsing, and that the stages fire when they should.

Getting a WAV out of the app takes a diagnostic build — normal builds analyse the
audio in memory and the recording code isn't compiled in at all:

  xcodebuild -project WatchSync/WatchSync.xcodeproj -scheme WatchSync \
      WATCHSYNC_DIAGNOSTIC_RECORDING=DIAGNOSTIC_RECORDING ...

Then take a measurement, Stop, tap Export, and AirDrop the `watch-<ts>.wav`
(+ `.json` sidecar) into ml/recordings/.

Usage:
  ./venv/bin/python tune_timegrapher.py recordings/new2.wav          # live replay
  ./venv/bin/python tune_timegrapher.py recordings/*.wav --summary   # one line each
  ./venv/bin/python tune_timegrapher.py recordings/new2.wav --bph 28800
"""
import argparse
import glob
import json
import os

import numpy as np
import soundfile as sf
from scipy.signal import lfilter

# --- constants, mirroring Timegrapher.swift ---------------------------------

STANDARD_BPH = [18000, 19800, 21600, 25200, 28800, 36000]
CANDIDATE_BANDS = [(4000, 9000), (5000, 11000), (7000, 13000), (9000, 15000),
                   (11000, 18000), (13000, 21000), (15000, 23000), (6000, 20000)]
ENV_RATE = 4000.0
TARGET_PRECISION = 2.0
PROBE_SECONDS = 4.0
BLOCK_SECONDS = 5.0
TEMPLATE_MS = 14.0
TEMPLATE_PRE_MS = 2.0
SEARCH_MS = 12.0
MIN_SCORE = 0.35


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


class Biquad:
    """Stateful biquad, so blocks can be filtered incrementally as in Swift."""

    def __init__(self, b, a):
        self.b, self.a = b, a
        self.zi = np.zeros(2)

    def block(self, x):
        if len(x) == 0:
            return x
        y, self.zi = lfilter(self.b, self.a, x, zi=self.zi)
        return y


def parabolic(y0, y1, y2):
    den = y0 - 2 * y1 + y2
    return 0.5 * (y0 - y2) / den if den != 0 else 0.0


# --- beat tracker -----------------------------------------------------------

class BeatTracker:
    """Times every beat in one listening band. Mirrors BeatTracker in Swift."""

    def __init__(self, lo, hi, bph, fs, target_env_rate=ENV_RATE):
        self.lo, self.hi, self.bph = lo, hi, bph
        self.nominal_cycle = 7200.0 / bph
        self.decim = max(1, int(round(fs / target_env_rate)))
        self.env_rate = fs / self.decim
        self.hp = Biquad(*rbj("hp", lo, fs))
        self.lp = Biquad(*rbj("lp", min(hi, fs / 2 - 500), fs))
        self.smooth = Biquad(*rbj("lp", self.env_rate / 3, fs))
        self.env = np.zeros(0)
        self._pending = np.zeros(0)
        self.template = None
        self.k = np.zeros(0, int)
        self.t = np.zeros(0)
        self.score = np.zeros(0)
        self.keep = np.zeros(0, bool)
        self.next_beat = 0
        self.slope = 0.0
        self.intercept = 0.0
        self.have_fit = False
        self.refolded = 0
        self.beat_error_ms = None

    @property
    def template_length(self):
        return max(8, int(TEMPLATE_MS * self.env_rate / 1000))

    @property
    def template_pre(self):
        return int(TEMPLATE_PRE_MS * self.env_rate / 1000)

    @property
    def search(self):
        return max(2, int(SEARCH_MS * self.env_rate / 1000))

    @property
    def period_samples(self):
        return self.nominal_cycle * self.env_rate

    @property
    def env_seconds(self):
        return len(self.env) / self.env_rate

    def extend(self, src):
        y = self.smooth.block(np.abs(self.lp.block(self.hp.block(src))))
        y = np.concatenate([self._pending, y])
        n = (len(y) // self.decim) * self.decim
        self._pending = y[n:]
        if n:
            self.env = np.concatenate(
                [self.env, y[:n].reshape(-1, self.decim).mean(axis=1)])

    def step(self):
        if self.template is None:
            if self.env_seconds < 2.5 or not self._bootstrap():
                return False
        self._track()
        self._refit()
        # Refold whenever the beat count has doubled since the last one: each
        # refold averages more beats, so the template keeps sharpening, and a
        # bad early template can't poison the whole measurement.
        if self.accepted >= max(60, 2 * self.refolded):
            if self._refold():
                self.refolded = self.accepted
                self._retrack_all()
            else:
                self.refolded = max(self.refolded, self.accepted)
        return True

    def _refine_period(self):
        """Autocorrelation period, searched near k cycles out for precision.

        The nominal period is only good to the watch's own rate error, which is
        enough to track but not to *fold* — over a couple of dozen cycles the
        error smears the averaged tick. Locking the period first makes the very
        first template as sharp as the ones that come later.
        """
        p0 = self.period_samples
        e = self.env - self.env.mean()
        n = len(e)
        k = max(1, min(12, int(n / (2 * p0))))
        tol = 0.005                     # 0.5% covers any real watch
        lo = int(k * p0 * (1 - tol))
        hi = int(k * p0 * (1 + tol)) + 1
        if hi + 4 >= n:
            return p0
        ac = np.array([np.dot(e[:n - lag], e[lag:n]) / (n - lag)
                       for lag in range(lo, hi + 1)])
        i = int(np.argmax(ac))
        frac = parabolic(ac[i - 1], ac[i], ac[i + 1]) if 0 < i < len(ac) - 1 else 0.0
        return (lo + i + frac) / k

    def _bootstrap(self):
        """Build the first template by folding, not by grabbing the loudest peak.

        The loudest peak in the first seconds is as likely to be a knock as a
        tick; folding averages every beat we have and lands on the real thing.
        """
        period = self._refine_period()
        L = int(period)
        if L <= self.template_length * 2 or len(self.env) < 3 * L:
            return False
        cycles = int(len(self.env) / period)
        acc = np.zeros(L)
        for c in range(cycles):
            s = int(round(c * period))
            if s + L <= len(self.env):
                acc += self.env[s:s + L]
        wf = acc / cycles
        wf = wf - np.median(wf)
        peak = int(np.argmax(wf))
        if wf[peak] <= 2 * np.abs(wf).mean():
            return False
        self.template = np.array([wf[(peak - self.template_pre + i) % L]
                                  for i in range(self.template_length)])
        # The fold is phase-locked to the start of the envelope, so beat zero
        # sits at the template's own offset within the first cycle.
        self.slope = period / self.env_rate
        self.intercept = max(0.0, (peak - self.template_pre)) / self.env_rate
        self.have_fit = True
        self.next_beat = 0
        return True

    def _retrack_all(self):
        self.k = np.zeros(0, int)
        self.t = np.zeros(0)
        self.score = np.zeros(0)
        self.keep = np.zeros(0, bool)
        self.next_beat = 0
        self._track()
        self._refit()

    @property
    def slope_is_plausible(self):
        """2% off nominal is 1728 s/day — nothing real, just a fit come apart."""
        return self.nominal_cycle * 0.98 < self.slope < self.nominal_cycle * 1.02

    def _track(self):
        if self.template is None or not self.slope_is_plausible:
            return
        tpl = self.template - self.template.mean()
        tnorm = np.linalg.norm(tpl)
        if tnorm <= 0:
            return
        L, S = len(tpl), self.search
        ks, ts, scores = [], [], []
        while True:
            predicted = self.intercept + self.slope * self.next_beat
            centre = int(round(predicted * self.env_rate))
            if centre + S + L >= len(self.env):
                break
            frm = centre - S
            if frm < 0:
                self.next_beat += 1
                continue
            segs = np.lib.stride_tricks.sliding_window_view(
                self.env[frm:frm + 2 * S + L], L)[: 2 * S + 1]
            segs = segs - segs.mean(axis=1, keepdims=True)
            norms = np.linalg.norm(segs, axis=1)
            corr = (segs @ tpl) / (norms * tnorm + 1e-12)
            j = int(np.argmax(corr))
            frac = parabolic(corr[j - 1], corr[j], corr[j + 1]) if 0 < j < 2 * S else 0.0
            ks.append(self.next_beat)
            ts.append((frm + j + frac) / self.env_rate)
            scores.append(corr[j])
            self.next_beat += 1
        if ks:
            self.k = np.concatenate([self.k, np.array(ks, int)])
            self.t = np.concatenate([self.t, np.array(ts)])
            self.score = np.concatenate([self.score, np.array(scores)])
            self.keep = np.concatenate([self.keep, np.array(scores) >= MIN_SCORE])

    def _refit(self):
        if len(self.k) < 8:
            return
        keep = self.keep.copy()
        for _ in range(4):
            if keep.sum() < 4:
                return
            kk = self.k[keep].astype(float)
            if kk.std() == 0:
                return
            slope, intercept = np.polyfit(kk, self.t[keep], 1)
            if not (self.nominal_cycle * 0.98 < slope < self.nominal_cycle * 1.02):
                return
            resid = self.t - (intercept + slope * self.k)
            med = np.median(resid[keep])
            scale = 1.4826 * np.median(np.abs(resid[keep] - med)) + 1e-9
            keep = (self.score >= MIN_SCORE) & (np.abs(resid) < 3 * scale)
            self.slope, self.intercept, self.have_fit = slope, intercept, True
        self.keep = keep

    def _refold(self):
        if not self.have_fit or not self.slope_is_plausible:
            return False
        L = int(self.slope * self.env_rate)
        if L <= self.template_length * 2:
            return False
        acc = np.zeros(L)
        folded = 0
        for i in np.flatnonzero(self.keep):
            s = int(round((self.intercept + self.slope * self.k[i]) * self.env_rate))
            if s < 0 or s + L >= len(self.env):
                continue
            acc += self.env[s:s + L]
            folded += 1
        if folded < 30:
            return False
        wf = acc / folded
        wf = wf - np.median(wf)
        peak = int(np.argmax(wf))
        self.template = np.array([wf[(peak - self.template_pre + i) % L]
                                  for i in range(self.template_length)])
        self.beat_error_ms = beat_error(wf, self.env_rate)
        self.intercept += (peak - self.template_pre) / self.env_rate
        return True

    @property
    def accepted(self):
        return int(self.keep.sum())

    def quality(self):
        if len(self.k) == 0 or not self.keep.any():
            return 0.0, 0.0, 0.0
        resid = self.t[self.keep] - (self.intercept + self.slope * self.k[self.keep])
        jitter = resid.std(ddof=1) * 1000 if len(resid) > 2 else 0.0
        return jitter, self.keep.sum() / len(self.k), float(np.median(self.score[self.keep]))

    def rate(self):
        if not self.have_fit or self.slope <= 0 or self.accepted < 20:
            return None
        return (self.nominal_cycle / self.slope - 1) * 86400

    def block_rates(self, tau, step):
        out = []
        if len(self.t) == 0:
            return out
        t0, last = self.t[0], self.t[-1]
        while t0 <= last - tau:
            m = self.keep & (self.t >= t0) & (self.t < t0 + tau)
            if m.sum() >= 8:
                s = np.polyfit(self.k[m].astype(float), self.t[m], 1)[0]
                if s > 0:
                    out.append((self.nominal_cycle / s - 1) * 86400)
            t0 += step
        return out

    def naive_se(self):
        if self.accepted <= 8:
            return None
        kk = self.k[self.keep].astype(float)
        resid = self.t[self.keep] - (self.intercept + self.slope * kk)
        if kk.std() == 0:
            return None
        se_slope = resid.std(ddof=1) / (np.sqrt(len(kk)) * kk.std())
        return 86400 * se_slope / self.nominal_cycle


def beat_error(wf, er):
    """Tic-to-toc spacing vs half the cycle, by circular autocorrelation."""
    n = len(wf)
    if n <= 16:
        return None
    w = wf - wf.mean()
    half = n // 2
    span = max(4, int(0.02 * er))
    lo, hi = max(1, half - span), min(n - 1, half + span)
    if hi <= lo + 2:
        return None
    ac = np.array([np.dot(w, np.roll(w, -lag)) for lag in range(lo, hi + 1)])
    i = int(np.argmax(ac))
    frac = parabolic(ac[i - 1], ac[i], ac[i + 1]) if 0 < i < len(ac) - 1 else 0.0
    return abs((lo + i + frac) - half) / er * 1000


# --- timegrapher ------------------------------------------------------------

def coarse_envelope(src, fs, lo, hi):
    b, a = rbj("hp", lo, fs)
    y = lfilter(b, a, src)
    b, a = rbj("lp", min(hi, fs / 2 - 500), fs)
    y = lfilter(b, a, y) ** 2
    d = max(1, int(round(fs / 1000.0)))
    n = (len(y) // d) * d
    e = y[:n].reshape(-1, d).mean(axis=1)
    return e - e.mean(), fs / d


def dft_mag(e, fs, f):
    n = len(e)
    return abs(np.dot(e, np.exp(-2j * np.pi * f / fs * np.arange(n)))) / n


def harmonic_prominence(e, er, bph):
    """Beat-line strength for a candidate rate, summed over harmonics.

    A tick train is impulsive, so its envelope carries real energy at 2f0 and
    3f0 too. Noise can flatter a wrong candidate's fundamental by luck; matching
    the whole comb is much harder to do by accident.
    """
    f0 = bph / 3600
    signal = sum(dft_mag(e, er, f0 * k) for k in (1, 2, 3))
    offsets = (0.6, 0.75, 1.2, 1.35, 1.6, 2.4, 2.7)
    noise = np.mean([dft_mag(e, er, f0 * k) for k in offsets])
    return signal / (3 * noise + 1e-12)


class Timegrapher:
    def __init__(self, fs, manual_bph=None):
        self.fs = fs
        self.manual = manual_bph
        self.tracker = None
        self.probed = False
        self.processed = 0
        self.progress_high_water = 0.0
        self.bad_passes = 0
        self.raw = np.zeros(0)

    def process(self, block):
        self.raw = np.concatenate([self.raw, block])

    def analyze(self):
        n = len(self.raw)
        elapsed = n / self.fs
        if elapsed < 1.5:
            return self._idle("listening", elapsed)

        if not self.probed:
            if elapsed < PROBE_SECONDS:
                return self._idle("listening", elapsed)
            self.tracker = self._probe(self.raw)
            if self.tracker is None:
                # Leave the probe armed: the user may still be positioning.
                self.raw = self.raw[int(PROBE_SECONDS * self.fs):]
                return self._idle("noSignal" if elapsed > 14 else "listening", elapsed)
            self.processed = len(self.raw)
            self.probed = True
            return self._idle("tuning", elapsed)

        if self.tracker is None:
            return self._idle("noSignal" if elapsed > 14 else "tuning", elapsed)
        if n > self.processed:
            self.tracker.extend(self.raw[self.processed:n])
            self.processed = n
        if not self.tracker.step():
            return self._idle("noSignal" if elapsed > 14 else "tuning", elapsed)

        # If the lock stays bad, throw it away and re-probe on fresh audio.
        jitter, detect, _ = self.tracker.quality()
        # Both must be bad: a false lock on room noise mistimes *and* loses
        # beats; a merely noisy watch still detects nearly every beat.
        self.bad_passes = self.bad_passes + 1 if (jitter > 1.0 and detect < 0.6) else 0
        if self.bad_passes >= 16:
            self.tracker, self.probed, self.bad_passes = None, False, 0
            self.raw = np.zeros(0)
            self.processed = 0
            return self._idle("tuning", elapsed)

        return self._build(self.tracker, elapsed)

    def _probe(self, src):
        """Choose band *and* beat rate by tracking candidates, not by spectrum.

        The spectrum alone picks the loudest band (not always the sharpest) and,
        under impulsive noise, the wrong rate — after which the tracker hunts for
        beats that were never there. Jitter separates a real escapement from
        noise by a factor of a hundred, so it decides.
        """
        scan = src[: int(PROBE_SECONDS * self.fs)]
        cands = [self.manual] if self.manual else STANDARD_BPH
        pairs = []
        for lo, hi in CANDIDATE_BANDS:
            e, er = coarse_envelope(scan, self.fs, lo, hi)
            if len(e) <= er:
                continue
            scored = sorted(((harmonic_prominence(e, er, bph), bph) for bph in cands),
                            reverse=True)
            band_prom = scored[0][0]
            # Each band's two best rates: the runner-up is what saves the
            # measurement when noise flatters the wrong one.
            for prom, bph in scored[:2]:
                pairs.append((band_prom, prom, lo, hi, bph))
        if not pairs:
            return None
        pairs.sort(key=lambda t: (-t[0], -t[1]))

        seen, shortlist = set(), []
        for _, _, lo, hi, bph in pairs:
            if len(seen) >= 3 and (lo, hi) not in seen:
                continue
            seen.add((lo, hi))
            shortlist.append((lo, hi, bph))

        winner, winner_key = None, (np.inf, 0.0)
        for lo, hi, bph in shortlist:
            t = BeatTracker(lo, hi, bph, self.fs)
            t.extend(scan)
            if not t.step() or t.accepted < 12:
                continue
            jitter, detect, score = t.quality()
            # A real escapement times to a fraction of a millisecond; anything
            # looser is room noise dressed up as a beat.
            if detect < 0.85 or jitter >= 1.0 or score < 0.6:
                continue
            key = (jitter, -score)
            if key < winner_key:
                winner_key, winner = key, t
        # No fallback to "the strongest spectral line": on a few seconds of room
        # noise that line *is* noise, and locking onto it is worse than waiting.
        return winner

    def _idle(self, stage, elapsed):
        p = min(0.25, elapsed / PROBE_SECONDS * 0.25)
        self.progress_high_water = max(self.progress_high_water, p)
        return dict(stage=stage, bph=self.tracker.bph if self.tracker else 0,
                    rate=None, unc=None, beat_error=None, jitter=0.0, detect=0.0,
                    score=0.0, beats=0, elapsed=elapsed,
                    band=(self.tracker.lo, self.tracker.hi) if self.tracker else (0, 0),
                    progress=self.progress_high_water, remaining=None)

    def _build(self, t, elapsed):
        jitter, detect, score = t.quality()
        rate = t.rate()

        unc = None
        if rate is not None:
            # SEM at several block lengths. Short blocks give a number
            # sooner but overstate the +/- for good — each block's own slope is
            # noisy — so they are only used until longer blocks exist.
            sems = []
            for tau, step in ((BLOCK_SECONDS / 2, BLOCK_SECONDS / 4),
                              (BLOCK_SECONDS, BLOCK_SECONDS / 2),
                              (BLOCK_SECONDS * 2, BLOCK_SECONDS)):
                blocks = t.block_rates(tau, step)
                if len(blocks) < 2:
                    continue
                independent = max(2.0, len(blocks) * step / tau)
                sems.append(np.std(blocks, ddof=1) / np.sqrt(independent))
            # Worst case over the two longest block lengths we can form.
            sd = 2 * max(sems[-2:]) if sems else None
            if sd is not None:
                value = sd
                naive = t.naive_se()
                if naive is not None:
                    value = max(value, 10 * naive)
                unc = max(0.5, value)

        # A wildly wide +/- means we are not really locked; show nothing rather
        # than a number with a meaningless bound attached to it.
        if unc is not None and unc > 25:
            rate, unc = None, None
        if rate is None or unc is None:
            stage = "locking"
        elif detect < 0.85 or jitter > 0.5 or score < 0.6:
            stage = "unstable"
        elif unc <= TARGET_PRECISION:
            stage = "done"
        else:
            stage = "measuring"
        if elapsed > 20 and t.accepted < 10:
            stage = "noSignal"

        remaining = None
        if unc is not None:
            ratio = TARGET_PRECISION / unc
            progress = 0.25 + 0.75 * min(1.0, ratio * ratio)
            if unc > TARGET_PRECISION:
                est = elapsed * ((unc / TARGET_PRECISION) ** 2 - 1)
                # Only worth showing when it's a wait, not a verdict.
                remaining = est if est <= 180 else None
        else:
            progress = min(0.25, elapsed / PROBE_SECONDS * 0.25)
        self.progress_high_water = max(self.progress_high_water, progress)

        return dict(stage=stage, bph=t.bph, rate=rate, unc=unc,
                    beat_error=t.beat_error_ms, jitter=jitter, detect=detect,
                    score=score, beats=t.accepted, elapsed=elapsed,
                    band=(t.lo, t.hi), progress=self.progress_high_water,
                    remaining=remaining)


# --- driver -----------------------------------------------------------------

def fmt_rate(rate, unc):
    """Round the headline exactly as the app does."""
    if rate is None or unc is None:
        return "—"
    if unc >= 15:
        return "%+.0f" % (round(rate / 5) * 5)
    if unc >= 5:
        return "%+.0f" % rate
    return "%+.1f" % rate


def opt(fmt, value, dash="—"):
    return (fmt % value) if value is not None else dash


def replay(path, manual=None, interval=0.5, quiet=False):
    x, fs = sf.read(path, always_2d=False)
    if x.ndim == 2:
        x = x.mean(axis=1)
    x = x.astype(np.float64)
    fs = float(fs)
    tg = Timegrapher(fs, manual)
    chunk = int(interval * fs)
    if not quiet:
        print("\n=== %s  %.1fs ===" % (os.path.basename(path), len(x) / fs))
        print("%6s %10s %8s %6s %5s %6s %6s %5s %5s %9s" %
              ("t", "stage", "shown", "+/-", "prog", "left", "jit", "det", "BE", "band"))
    last = None
    pos = 0
    while pos < len(x):
        tg.process(x[pos:pos + chunk])
        pos += chunk
        last = tg.analyze()
        if quiet:
            continue
        r = last
        band = ("%.0f-%.0fk" % (r["band"][0] / 1000, r["band"][1] / 1000)
                if r["band"][0] else "—")
        print("%6.1f %10s %8s %6s %5.2f %6s %6.3f %5.2f %5s %9s" %
              (r["elapsed"], r["stage"], fmt_rate(r["rate"], r["unc"]),
               opt("%.1f", r["unc"]), r["progress"], opt("%.0fs", r["remaining"]),
               r["jitter"], r["detect"], opt("%.2f", r["beat_error"]), band))
    return last


def compare_sidecar(path, result):
    """If the app exported a .json next to the WAV, diff it against this port."""
    side = os.path.splitext(path)[0] + ".json"
    if not os.path.exists(side):
        return
    with open(side) as f:
        app = json.load(f)
    print("  app :", {k: app.get(k) for k in
                      ("beatsPerHour", "rateSecondsPerDay", "uncertainty")})
    print("  port:", {"beatsPerHour": result["bph"],
                      "rateSecondsPerDay": result["rate"],
                      "uncertainty": result["unc"]})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("wavs", nargs="+")
    ap.add_argument("--bph", type=int, default=None, help="force the beat rate")
    ap.add_argument("--interval", type=float, default=0.5,
                    help="analysis interval, matching the app's timer")
    ap.add_argument("--summary", action="store_true",
                    help="one line per file instead of the live replay")
    args = ap.parse_args()

    paths = []
    for w in args.wavs:
        paths.extend(sorted(glob.glob(w)))

    if args.summary:
        print("%26s %10s %6s %8s %6s %5s %6s %6s" %
              ("file", "stage", "bph", "rate", "+/-", "BE", "jit", "beats"))
    for p in paths:
        r = replay(p, args.bph, args.interval, quiet=args.summary)
        if r is None:
            continue
        if args.summary:
            print("%26s %10s %6d %8s %6s %5s %6.3f %6d" %
                  (os.path.basename(p), r["stage"], r["bph"],
                   fmt_rate(r["rate"], r["unc"]), opt("%.1f", r["unc"]),
                   opt("%.2f", r["beat_error"]), r["jitter"], r["beats"]))
        else:
            compare_sidecar(p, r)


if __name__ == "__main__":
    main()
