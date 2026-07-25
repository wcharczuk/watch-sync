# Watch Accuracy Measurement: Technical Approach

Measure a mechanical watch's drift (seconds/day) by tracking the sweep second
hand in phone video. This document describes what we actually built and
validated, the error budget we worked through, and why each piece is there.

> **Status (May 2026):** Solved & ground-truth-validated for **stable/propped
> capture: −9 → −2 s/day on a Tudor Pelagos, matching its near-spec rate, with
> ±2 s/day precision.** Handheld: the second hand is tracked robustly under any
> motion (the perception problem is solved), but absolute accuracy under *heavy*
> motion is limited by sensor physics (motion blur + rolling shutter), not
> algorithm. Reference pipeline: `ml/validate_v9.py`.

---

## The core idea

The second hand is the only thing on the dial moving at ~6°/s. Everything else
(markers, text, date, hour/minute/GMT hands, bezel) is static or near-static.
So: sample the dial into a **kymograph** — a 2-D array `[time, angle]` of the
brightness in a thin annulus around the dial — and the second hand is the one
feature that traces a **diagonal at 6°/s** while everything else is vertical.
Fit the slope of that diagonal over a couple of minutes; the deviation from
exactly 6.000°/s, scaled by 86400/6, is the drift in s/day.

To hit ±5 s/day you need the slope good to ~3.5e-4 °/s — i.e. the second-hand
angle stable/unbiased to **~0.04° over two minutes**. That precision budget is
what makes every sub-degree systematic below matter.

---

## Pipeline (validate_v9.py)

```
video ─▶ real per-frame timestamps (CMSampleBuffer PTS / CAP_PROP_POS_MSEC)
      ─▶ detect watch face, generous square crop
      ─▶ per-frame watch CENTER (Hough per frame, nan-interp + temporal smooth)
      ─▶ ITERATIVE CENTER-NULLING:
            build kymograph (annulus 0.50–0.85 R, 720 angle bins) about center
            temporal HIGH-PASS per angle bin (kill static features)
            Radon velocity lock (≈6°/s) + sub-bin centroid track
            measure the once-per-rev (1st angular harmonic) → center offset
            shift sampling center to null it; repeat until harmonic ≈ 0
      ─▶ confidence/quality-gated robust (trimmed) PLAIN linear fit on real time
      ─▶ drift = (slope − 6.000)/6.000 × 86400  s/day
```

### 1. Real timestamps — not assumed fps
iPhone video is **59.94 fps variable-frame-rate with dropped frames** (a 1/60 s
grid with occasional 18.3 ms gaps), and clips even differ from each other
(59.928 vs 59.972 fps). Assuming a nominal 30 or 60 fps biased the rate by
~100 s/day. Always use each frame's real presentation timestamp. (Verified
against `ffprobe` PTS; the phone clock itself is good to ±4 s/day.)

### 2. Temporal high-pass per angle bin — the signal extractor
The breakthrough that makes the second hand pop out. Subtracting a *background
image* and taking |frame − bg| leaves vertical banding (static edges never
cancel under sub-pixel jitter), and on hard dials that banding **outscores** the
second hand. Instead, subtract a per-angle moving-average **along time**
(window ≈ 1.5 s) in kymograph space: anything static/slow is low-frequency →
removed; the second hand sweeps ~9° through any bin in 1.5 s → a clean pulse →
preserved. This raised the second-hand velocity-peak SNR 3–5× and is what lets
the Radon lock land at exactly 6°/s.

### 3. Watch-center tracking — the handheld key
Over a 2-min handheld clip the watch *translates* tens of pixels within the
frame. A fixed crop lets the sampling center slide off the true pivot, which
(see §4) injects a time-varying angle bias. We follow the center per frame.
Sequential (frame-to-frame) tracking **accumulates drift** and invents motion
(it hallucinated 66 px on a propped clip) — use an **absolute, non-accumulating**
center: detect per frame and temporally smooth. (Per-frame phase-correlation
centers were tried and were *worse* — noisier than smoothed detection.)

### 4. Center-offset nulling — the dominant systematic
The detected (bezel/case) circle center is offset from the true second-hand
**pivot** by a sub-pixel amount. A center offset makes the measured angle a
**once-per-revolution sinusoid** (Δθ ≈ −(eₓcosθ + e_y sinθ)/R). This biases a
1-revolution slope hugely (~−290 s/day) and only *partly* cancels over ~2 rev
(leaving the ~−50 s/day we chased for a long time). Fix: estimate the offset
from the 1st angular harmonic of the angle residual, shift the sampling center
to null it, and iterate (converges in ~3 steps to <0.03°). After nulling, a
plain linear fit is unbiased — no harmonic-vs-plain or estimator choice.

### 5. Robust fit + gating
Sub-bin intensity-weighted centroid of the high-pass ridge per frame; a
per-frame confidence (window-peak / row-MAD) gates out motion-blurred frames; a
trimmed (iteratively reweighted) linear fit rejects outliers. Slope → s/day.

---

## What did NOT work (and why we removed it)

- **Single-frame ML / angle regression** — can't tell the thin second hand from
  the thin red GMT hand; latches onto thicker hands. No temporal info.
- **Per-frame Vision registration** — ~750 ms/frame; far too slow. Hardware
  video stabilization is free and sufficient for translation/rotation.
- **De-rotation (12th-harmonic, ORB, log-polar)** — a **red herring**. iPhone
  hardware stabilization already removes most rotation; the 12th-harmonic method
  *hallucinated* −120° of rotation and made results worse. Removed entirely.
- **Background-image subtraction** — leaves vertical banding; replaced by the
  temporal high-pass (§2).
- **Phase-correlation per-frame center** — noisier than smoothed detection;
  dropped SNR. The center wasn't the time-varying problem on heavy-motion clips.
- **Time-varying harmonic de-biasing** — extra DOF just wander the slope without
  reducing RMS → the heavy-motion residual is not a low-order angle harmonic.

---

## Error budget (ordered by impact, as resolved)

| Source | Effect | Resolution |
|---|---|---|
| Assumed fps (VFR) | ~100 s/day | Real per-frame PTS |
| Finding the hand at all | total failure on hard dials | Temporal high-pass kymograph |
| Watch translation (handheld) | watch exits crop → garbage | Per-frame absolute center tracking |
| **Center offset (bezel ≠ pivot)** | **~50–290 s/day** (1st harmonic) | **Iterative center-nulling** |
| Camera rotation | none in practice | Hardware-stabilized; de-rotation removed |
| Phone video clock | ≤ ±4 s/day | Negligible (verified vs ffprobe PTS) |
| **Motion blur + rolling shutter** | **~±30 s/day under heavy handheld motion** | **Capture-side only** — steady/propped capture, or audio |

Net: **±2 s/day** achievable with stable capture and ≥2 revolutions (≥~2 min).

---

## Capture guidance

- **Prop or lay the watch flat, phone steady** — eliminates the rolling-shutter /
  perspective-tilt regime that defeats heavy handheld.
- **≥ 2 full revolutions** (≥ ~2 min). Below 1 rev the slope fit has no leverage
  against any structured error (even synthetic swings ±300 s/day at ⅓ rev).
- **Lock focus & exposure**; avoid glare/reflection sliding across the crystal.
- **4K helps** — the thin second hand needs enough pixels; low-res archive clips
  fail the velocity lock at SNR ~10–30.
- Watch face roughly **parallel** to the sensor (minimize tilt).

---

## Code map

- `ml/validate_v9.py` — the reference pipeline (offline validation).
- `ml/validate_v7.py` — earlier consolidation (real-ts, high-pass, fit; no center-nulling).
- `ml/exp_*.py` — the investigation: `exp_highpass` (signal), `exp_synth`
  (estimator validation), `exp_abscenter` (absolute center), `exp_window`
  (proved the center-offset signature), `exp_estbias` (estimator is clean),
  `exp_timevarying` (ruled out phase-corr center / time-varying harmonics).
- `ml/cache/<name>_gen.npy` + `_gen_meta.npz` — cached generous crops + centers
  so re-runs skip the slow per-frame Hough.
- iOS: `WatchSync/WatchSync/{CameraManager,SecondHandAnalyzer,AccuracyView}.swift`.

---

## Drift math

```
slope  = d(angle)/d(time)        # measured, deg/s, on real timestamps
drift  = (slope − 6.000) / 6.000 × 86400   # s/day  (+ = watch runs fast)
σ_drift = SE(slope) / 6.000 × 86400
```
Validated: stable Pelagos slope ≈ 5.9994°/s → −9 s/day (→ −2 after full nulling),
matching the watch's near-spec rate.
