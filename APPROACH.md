# Watch Accuracy Measurement: Technical Approach

Two independent measurements of a mechanical watch's drift (seconds/day):

1. **Acoustic timegrapher** (what the app ships) — listen to the escapement and
   time every beat. First number in ~5.5 s, ±2 s/day by ~14 s. See
   [Part 2](#part-2-the-acoustic-timegrapher) below.
2. **Video second-hand tracking** — track the sweep hand in phone video.
   Solved and ground-truth-validated for propped capture; limited by sensor
   physics when handheld. That's Part 1, immediately following.

---

# Part 1: video second-hand tracking

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

---

# Part 2: the acoustic timegrapher

> **Status (July 2026):** shipping. A first number at **~5.5 s** (worst 6.5 s),
> **±2 s/day by ~14 s**, settling near ±1, verified across nine device
> recordings. Reference:
> `WatchSync/WatchSync/Timegrapher.swift`, mirrored line-for-line by
> `ml/tune_timegrapher.py` (identical outputs on every recording).

## The core idea

An escapement ticks ~8 times a second, and each tick is a sharp transient whose
*shape* repeats to a fraction of a millisecond. So don't measure a frequency —
**time every individual beat** against the phone's audio clock and fit a straight
line through beat time vs beat number. The slope is the rate. Roughly 500 beats
in a minute, each timed to ~0.05 ms, is an enormous amount of leverage; that's
why this converges in seconds where the video method needs two minutes.

## Pipeline

```
mic (48 kHz, .measurement mode — no AGC)
  ─▶ BAND PROBE (once, at 4 s): shortlist bands by spectral beat-line
       prominence, then actually track beats in the top 3 and keep the band
       with the lowest timing jitter (rejecting any that can't time a beat to
       under 1 ms — that's room noise, not an escapement)
  ─▶ band-pass ─▶ rectify ─▶ smooth ─▶ decimate to a 4 kHz envelope
  ─▶ BOOTSTRAP: autocorrelation period lock (searched k cycles out), fold the
       envelope at it ─▶ tick template
  ─▶ TRACK: matched-filter each beat within ±12 ms of where the running fit
       predicts it; parabolic sub-sample peak ─▶ beat time
  ─▶ REFOLD whenever the beat count doubles: average all tracked beats into a
       sharper template, re-time everything with it
  ─▶ robust (IRLS) fit of beat time vs beat number ─▶ rate = (nominal/slope − 1) × 86400
  ─▶ ± from the scatter of independent sub-window rates
```

### Why not track the beat frequency?
The previous version demodulated at the beat frequency and fitted the lock-in
phase slope. It works, eventually — but the phase wanders for the better part of
a minute, so early readings were not merely imprecise, they were *wrong and
confident*: one recording read −52 s/day at 8 s and −22 at 20 s, converging on
−2.8. Timing beats reads −2.3 at 8 s and −2.7 at 20 s. Across the corpus the
lock-in swung 5–50 s/day before settling; beat tracking moves under 1.5.

### Band selection — measured, not assumed
Escapement energy lands anywhere from ~4 to ~20 kHz depending on the movement and
how the case couples to the phone. Below ~4 kHz is hopeless everywhere (handling
and room rumble; timing jitter ~6 ms vs ~0.05 ms up high). Above that, band
choice barely matters — every band from 5–23 kHz agreed within 0.3 s/day on eight
of nine recordings. But the *spectrally loudest* band is not always the sharpest:
on one recording the strongest beat line came from a ringing band that smeared
every tick (jitter 0.13 ms, ±2.9) while a neighbouring band gave 0.045 ms and
±0.4. So the probe ranks candidates by measured timing jitter, not by spectrum.
There was no benefit found in combining bands or in fancier spectral weighting.

### The uncertainty is the whole confidence story
There is no invented "confidence %" anywhere. The published ± is:

```
±  =  max( 2 × SEM of independent sub-window rates,
           10 × the fit's own standard error,
           0.5 )                                     ← the phone's crystal
```

The SEM is computed at 2.5 s, 5 s and 10 s block lengths, taking the **worst of
the two longest available**. Short blocks are what let a number appear at ~5.5 s
instead of ~8 s, but each short block's own slope is noisy, so keeping them in
the mix forever would overstate the ± permanently and the reading would never
reach ±2. Dropping them as longer blocks become available gets both.

- The fit's **own** SE underestimates the real error by 10–20×: beat timings are
  correlated, because both the contact and the watch wander. Alone it covered
  only 53% of observed errors.
- **Sub-window scatter** is the honest signal, measured at two block lengths so
  an unlucky chopping of the record can't make a drifting reading look settled.
- Calibrated against every recording, this covers ~92% of actual deviations and
  is typically ~4× conservative, which is the right direction to be wrong in.
  Because it is a conservative *bound*, the headline keeps one decimal well past
  ±1 — the typical error is about a fifth of the published figure.

The old UI *had* a sound sub-window estimate and then threw it away, displaying
instead the spread of successive whole-recording readings. Those readings share
nearly all their data, so their spread is near zero regardless of the truth —
which is exactly why the bar could fall while the headline said the reading was
good, and why a wrong number could look settled.

### What the quality metrics actually predict
| Metric | Predicts error? | Use |
|---|---|---|
| Lock-in amplitude SNR (old "Signal") | **No** — a recording with SNR 12 was off by 7 s/day | removed |
| Spectral beat-line prominence | weakly | band shortlisting only |
| **Per-beat timing jitter** (ms) | **yes** — 0.03–0.11 clean, >0.25 degrading | band probe + "unstable" gate |
| **Beat detection rate** | **yes** | "unstable" gate |
| **Sub-window rate scatter** | **yes — the best** | the published ± |

## Stages the UI exposes
`listening → tuning (band + beat rate) → locking (timing ticks) → measuring
(number + shrinking ±) → done (± ≤ 2 s/day)`, plus `unstable` (losing ticks —
actionable) and `noSignal`. The analyzer owns the stage; the view never forms its
own opinion from raw metrics. Progress is monotonic by construction, and the
headline is rounded to the precision the ± supports — a tenth normally, whole
numbers past ±5, nearest 5 past ±15 (no "−30.4 ± 8"). While there is no number
yet, the screen shows mic level and then a live beat count, so the few seconds
before the first reading don't look like a hang.

A lock that goes bad — jitter over 1 ms *and* under 60% of beats detected, for
8 s — is thrown away and re-probed. Both conditions are required: a false lock on
room noise mistimes *and* loses beats, whereas a merely noisy watch still detects
nearly every beat, and restarting there would discard a usable measurement.

## Free extras from beat tracking
- **Beat error** — tic-to-toc spacing from the folded waveform, via circular
  autocorrelation near half the cycle (peak-picking finds the tick's own drop
  instead, giving nonsense like 5.75 ms).
- **Paper tape** — every beat plotted by its cumulative offset from a perfect
  clock, with the fitted line drawn through it. The headline number is the slope
  of that line, so the user can check it by eye.

## Noise: where it breaks, and why the rate detector matters most

Injected-interference tests (`ml/exp_noise.py`) add controlled noise to clean
recordings and score against that recording's own clean reading. Four kinds, at
levels in dB relative to the recording's in-band RMS:

| interference | accurate to | fails by |
|---|---|---|
| narrowband tone (fan, coil whine) | **+30 dB** and beyond | never — the band probe routes around it |
| broadband hiss | **+18 dB** | lost lock at +24 |
| bursty rustle (fabric, hands) | **+18 dB** | lost lock at +24 |
| impulsive knocks | **+18 dB** | lost lock at +24 |

Two things worth stating. Errors stay under ~0.9 s/day right up to the cliff, and
past it the result is **no reading**, never a confident wrong one — every failure
shows up as a wider ±, the `unstable` stage, or a refusal to lock.

The cliff was *not* the acceptance gates. Under impulsive noise the **beat rate**
was misidentified — 21600 chosen for a 28800 movement — after which the tracker
hunts for beats that were never there and everything downstream is garbage. The
spectral score looks at one line, and noise has enough broadband structure to
flatter the wrong one. Fixed by scoring candidates on a **harmonic sum** (a tick
train has real energy at 2f0 and 3f0; noise rarely matches the whole comb) and by
probing **(band, rate) pairs** — each band's two best rates — then letting
measured timing jitter choose. Jitter separates a real escapement from noise by
a factor of a hundred (0.05 ms vs 8 ms). This recovers cases that were previously
lost and leaves all nine clean recordings bit-identical.

### Mitigations that did *not* work
- **tg's noise suppressor** (zeroing windows above the median energy) made things
  *worse* — it lost cases the baseline handled. It suits tg's peak-picking
  detector; our matched filter is already robust to impulsive noise, and blanking
  samples corrupts the correlation it depends on.
- **Relaxing the probe gates** — more locks, but worse readings. The gates are
  rejecting genuine garbage.
- **A longer template** (2× integration) improved accuracy on the cases that
  survived but lost more of them; the extra length spans real beat-to-beat
  variation.

## Limits
- The reading is against the phone's crystal (~±0.5 s/day), hence the ± floor.
  tg solves this with a GPS/reference calibration mode; we don't, yet.
- Amplitude (degrees) is not computed — it needs the movement's lift angle and
  pulse-width measurement, which tg does in `compute_amplitude`.
- One recording (`new3`) drifts −12 → −35 s/day over 40 s with rising jitter.
  Both estimators agree, so it is the recording, not the algorithm; the ± and
  the `unstable` stage flag it rather than hiding it.

## Prior art
`vacaboja/tg` (Marcello Mamino) is the reference open-source timegrapher and the
source of several ideas here: filter → rectify → envelope, fold to a template,
cross-correlate to locate events, beat error from the waveform autocorrelation,
trimmed-mean folding, and a noise suppressor that zeroes windows above the median
energy. tg displays the rate from the period measured over a ≤16 s window and
lets the user read long-term rate off the paper strip by eye; we fit the whole
record and publish a measured ±.

## Acoustic code map
- `WatchSync/WatchSync/Timegrapher.swift` — the shipping analyzer.
- `ml/tune_timegrapher.py` — faithful port + live replay of a recorded WAV
  (`--summary` for one line per file). Use it to tune before touching Swift.
  Getting a WAV out of the app needs a **diagnostic build**. Create
  `WatchSync/Config/Local.xcconfig` (gitignored) containing:

      WATCHSYNC_DIAGNOSTIC_RECORDING = DIAGNOSTIC_RECORDING

  and build normally — Xcode's Run button honours it, which a `-D` on the
  xcodebuild command line does not. Delete the file to turn it off again.

  The structure is deliberate: the only committed definition
  (`Config/Diagnostics.xcconfig`) is empty and pulls the local file in with
  `#include?`, so a fresh clone cannot record and the "on" value cannot be
  committed by accident. The source additionally requires `DEBUG`
  (`#if DEBUG && DIAGNOSTIC_RECORDING`), so a Release build cannot contain the
  recording code even if the setting were switched on by mistake. In a normal
  build a measurement analyses the audio in memory and leaves nothing behind.

- `ml/tg_lab.py` — the original comparison of lock-in vs beat tracking.
- `ml/exp_uncertainty.py` — what predicts the error (± calibration).
- `ml/exp_band.py` — band sweep: prominence vs jitter ranking.
- `ml/exp_shippable.py` — the offline form of the incremental algorithm.
- `ml/recordings/*.wav` — device recordings the above are validated against.
