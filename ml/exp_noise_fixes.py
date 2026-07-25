#!/usr/bin/env python3
"""
Why the lock is lost at high noise, and what recovers it.

exp_noise.py shows accurate readings up to +18 dB interference and a hard cliff
at +24 dB for broadband and impulsive noise — no reading at all. This asks which
gate is doing the rejecting, then tests fixes:

  suppress   tg's noise suppressor: zero windows whose energy is far above the
             median, so a knock can't drown the beats around it
  relax      loosen the band probe's acceptance gates
  longtpl    a longer tick template (more integration against white noise)

  ./venv/bin/python exp_noise_fixes.py diagnose
  ./venv/bin/python exp_noise_fixes.py compare
"""
import sys

import numpy as np
from scipy.signal import butter, lfilter

import tune_timegrapher as tg
from exp_noise import KINDS, contaminate, load, run, score


# ---------------------------------------------------------------------------
# diagnosis: what does the probe see?
# ---------------------------------------------------------------------------

def probe_report(x, fs, label):
    """Re-run the band shortlist and print why each candidate passes or fails."""
    scan = x[: int(tg.PROBE_SECONDS * fs)]
    shortlist = []
    for lo, hi in tg.CANDIDATE_BANDS:
        e, er = tg.coarse_envelope(scan, fs, lo, hi)
        if len(e) <= er:
            continue
        best = None
        for bph in tg.STANDARD_BPH:
            f0 = bph / 3600
            mag = tg.dft_mag(e, er, f0)
            noise = np.mean([tg.dft_mag(e, er, f0 * k)
                             for k in (0.6, 0.75, 1.2, 1.35, 1.6)])
            prom = mag / (noise + 1e-12)
            if best is None or prom > best[0]:
                best = (prom, bph)
        shortlist.append((best[0], lo, hi, best[1]))
    shortlist.sort(reverse=True)

    print(f"\n--- {label}")
    print(f"{'band':>12} {'prom':>7} {'bph':>6} {'beats':>6} {'detect':>7} "
          f"{'jitter':>7} {'score':>6}  verdict")
    for prom, lo, hi, bph in shortlist[:3]:
        t = tg.BeatTracker(lo, hi, bph, fs)
        t.extend(scan)
        ok = t.step()
        if not ok or t.accepted < 12:
            print(f"{lo/1000:5.0f}-{hi/1000:<4.0f}k {prom:7.1f} {bph:6d} "
                  f"{t.accepted:6d} {'—':>7} {'—':>7} {'—':>6}  no template/too few beats")
            continue
        jitter, detect, sc = t.quality()
        reasons = []
        if detect < 0.85:
            reasons.append(f"detect<0.85")
        if jitter >= 1.0:
            reasons.append(f"jitter>=1ms")
        if sc < 0.6:
            reasons.append(f"score<0.6")
        verdict = "ACCEPTED" if not reasons else "rejected: " + ", ".join(reasons)
        print(f"{lo/1000:5.0f}-{hi/1000:<4.0f}k {prom:7.1f} {bph:6d} "
              f"{t.accepted:6d} {detect:7.2f} {jitter:7.3f} {sc:6.2f}  {verdict}")


def diagnose(path="recordings/new2.wav"):
    x, fs = load(path)
    rng = np.random.default_rng(7)
    probe_report(x, fs, "clean")
    for kind in ("hiss", "knocks", "rustle"):
        for db in (18, 24):
            y = contaminate(x, fs, kind, db, rng)
            probe_report(y, fs, f"{kind} +{db} dB")


# ---------------------------------------------------------------------------
# mitigations
# ---------------------------------------------------------------------------

def noise_suppressed(x, fs, factor=3.0):
    """tg's noise suppressor, applied to the raw signal before analysis.

    Zero any stretch whose short-term energy sits far above the median. A knock
    is thousands of times louder than a tick, so without this it doesn't just
    mask the beats underneath — it drags the whole normalization with it.
    """
    w = max(1, int(fs / 50))
    p = np.convolve(x * x, np.ones(w) / w, mode="same")
    step = max(1, int(fs / 2))
    peaks = [p[i:i + step].max() for i in range(0, max(1, len(p) - step), step)]
    if not peaks:
        return x
    k = np.median(peaks)
    out = x.copy()
    out[p > factor * k] = 0.0
    return out


class RelaxedProbe(tg.Timegrapher):
    """Accept a band on weaker evidence, leaning on the +/- to stay honest."""

    def _probe(self, src):
        scan = src[: int(tg.PROBE_SECONDS * self.fs)]
        cands = [self.manual] if self.manual else tg.STANDARD_BPH
        shortlist = []
        for lo, hi in tg.CANDIDATE_BANDS:
            e, er = tg.coarse_envelope(scan, self.fs, lo, hi)
            if len(e) <= er:
                continue
            best = None
            for bph in cands:
                f0 = bph / 3600
                mag = tg.dft_mag(e, er, f0)
                noise = np.mean([tg.dft_mag(e, er, f0 * k)
                                 for k in (0.6, 0.75, 1.2, 1.35, 1.6)])
                prom = mag / (noise + 1e-12)
                if best is None or prom > best[0]:
                    best = (prom, bph)
            shortlist.append((best[0], lo, hi, best[1]))
        if not shortlist:
            return None
        shortlist.sort(reverse=True)

        winner, winner_key = None, (np.inf, 0.0)
        for prom, lo, hi, bph in shortlist[:3]:
            t = tg.BeatTracker(lo, hi, bph, self.fs)
            t.extend(scan)
            if not t.step() or t.accepted < 12:
                continue
            jitter, detect, sc = t.quality()
            if detect < 0.6 or jitter >= 2.0 or sc < 0.4:   # was .85 / 1.0 / .6
                continue
            key = (jitter, -sc)
            if key < winner_key:
                winner_key, winner = key, t
        return winner


def longer_template(factor=2.0):
    """Longer matched filter: more integration, at the cost of time resolution."""
    class LongTemplate(tg.BeatTracker):
        @property
        def template_length(self):
            return max(8, int(tg.TEMPLATE_MS * factor * self.env_rate / 1000))
    return LongTemplate


VARIANTS = {
    "baseline":  dict(),
    "suppress":  dict(pre=noise_suppressed),
    "relax":     dict(gr=RelaxedProbe),
    "longtpl":   dict(tracker=longer_template(2.0)),
    "supp+relax": dict(pre=noise_suppressed, gr=RelaxedProbe),
}


# ---------------------------------------------------------------------------
# the real fix: choose the beat rate by tracking it, not by one spectral line
# ---------------------------------------------------------------------------

def harmonic_prominence(e, er, bph):
    """Beat-line strength summed over harmonics.

    A tick train is impulsive, so its envelope carries strong energy at 2f0 and
    3f0 as well. A wrong candidate can match the fundamental by luck — impulsive
    noise has broadband structure — but it cannot match the whole comb.
    """
    f0 = bph / 3600
    sig = sum(tg.dft_mag(e, er, f0 * k) for k in (1, 2, 3))
    noise = np.mean([tg.dft_mag(e, er, f0 * k)
                     for k in (0.6, 0.75, 1.2, 1.35, 1.6, 2.4, 2.7)])
    return sig / (3 * noise + 1e-12)


class PairProbe(tg.Timegrapher):
    """Probe (band, beat rate) *pairs* by actually tracking each one.

    The spectral shortlist alone picks the beat rate, and under impulsive noise
    it picks the wrong one — after which the tracker hunts for beats that were
    never there. Timing jitter separates a real escapement from noise by a factor
    of a hundred (0.05 ms vs 8 ms), so let it decide the rate too.
    """
    HARMONIC = True
    N_BANDS = 3
    N_RATES = 2

    def _probe(self, src):
        scan = src[: int(tg.PROBE_SECONDS * self.fs)]
        cands = [self.manual] if self.manual else tg.STANDARD_BPH
        pairs = []
        for lo, hi in tg.CANDIDATE_BANDS:
            e, er = tg.coarse_envelope(scan, self.fs, lo, hi)
            if len(e) <= er:
                continue
            scored = []
            for bph in cands:
                if self.HARMONIC:
                    prom = harmonic_prominence(e, er, bph)
                else:
                    f0 = bph / 3600
                    noise = np.mean([tg.dft_mag(e, er, f0 * k)
                                     for k in (0.6, 0.75, 1.2, 1.35, 1.6)])
                    prom = tg.dft_mag(e, er, f0) / (noise + 1e-12)
                scored.append((prom, bph))
            scored.sort(reverse=True)
            best_band_prom = scored[0][0]
            for prom, bph in scored[: self.N_RATES]:
                pairs.append((best_band_prom, prom, lo, hi, bph))
        if not pairs:
            return None
        # Best bands first, then their best rates.
        pairs.sort(key=lambda t: (-t[0], -t[1]))
        seen_bands, shortlist = set(), []
        for band_prom, prom, lo, hi, bph in pairs:
            if len(seen_bands) >= self.N_BANDS and (lo, hi) not in seen_bands:
                continue
            seen_bands.add((lo, hi))
            shortlist.append((lo, hi, bph))

        winner, winner_key = None, (np.inf, 0.0)
        for lo, hi, bph in shortlist:
            t = tg.BeatTracker(lo, hi, bph, self.fs)
            t.extend(scan)
            if not t.step() or t.accepted < 12:
                continue
            jitter, detect, sc = t.quality()
            if detect < 0.85 or jitter >= 1.0 or sc < 0.6:
                continue
            key = (jitter, -sc)
            if key < winner_key:
                winner_key, winner = key, t
        return winner


class PairProbePlain(PairProbe):
    HARMONIC = False


VARIANTS["harmonic+pairs"] = dict(gr=PairProbe)
VARIANTS["pairs only"] = dict(gr=PairProbePlain)
VARIANTS["pairs+suppress"] = dict(gr=PairProbe, pre=noise_suppressed)


def run_variant(x, fs, spec):
    y = spec["pre"](x, fs) if "pre" in spec else x
    grapher = spec.get("gr", tg.Timegrapher)
    tracker = spec.get("tracker")
    original_tracker, original_grapher = tg.BeatTracker, tg.Timegrapher
    if tracker is not None:
        tg.BeatTracker = tracker
    try:
        g = grapher(fs)
        chunk = int(0.5 * fs)
        pos, last = 0, None
        while pos < len(y):
            g.process(y[pos:pos + chunk])
            pos += chunk
            last = g.analyze()
        return last
    finally:
        tg.BeatTracker, tg.Timegrapher = original_tracker, original_grapher


def compare(paths=None):
    paths = paths or ["recordings/new2.wav", "recordings/rolex2.wav",
                      "recordings/caseoff.wav"]
    rng = np.random.default_rng(99)
    results = {name: {"err": [], "lost": 0, "n": 0} for name in VARIANTS}
    for p in paths:
        x, fs = load(p)
        truth = run(x, fs)
        if truth is None or truth["rate"] is None:
            continue
        truth = truth["rate"]
        for kind in ("hiss", "knocks", "rustle"):
            for db in (18, 24, 30):
                y = contaminate(x, fs, kind, db, rng)
                line = f"{p.split('/')[-1][:10]:>10} {kind:>7} {db:3d} |"
                for name, spec in VARIANTS.items():
                    r = run_variant(y, fs, spec)
                    results[name]["n"] += 1
                    if r is None or r["rate"] is None:
                        results[name]["lost"] += 1
                        line += f" {name}:{'lost':>6}"
                    else:
                        e = abs(r["rate"] - truth)
                        results[name]["err"].append(e)
                        line += f" {name}:{e:6.2f}"
                print(line)

    print(f"\n{'variant':>12} | {'median err':>10} {'p90':>7} {'lost':>5} {'of':>4}")
    for name, r in results.items():
        e = np.array(r["err"]) if r["err"] else np.array([np.nan])
        print(f"{name:>12} | {np.nanmedian(e):10.2f} {np.nanpercentile(e, 90):7.2f} "
              f"{r['lost']:5d} {r['n']:4d}")


def mitigation_sweep(paths):
    compare(paths)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "diagnose"
    if cmd == "diagnose":
        diagnose(*sys.argv[2:])
    else:
        compare(sys.argv[2:] or None)
