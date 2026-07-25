import Foundation

/// A point on the rate-drift trace (cumulative timing offset over the recording).
struct PhasePoint {
    let time: Double        // seconds
    let offsetMs: Double    // cumulative timing offset (ms); slope = rate
}

/// Result of one analysis pass.
struct TimegrapherResult {
    /// Nominal beat rate (auto-detected or manual override).
    let beatsPerHour: Int
    /// Beat frequency (beats/second) = beatsPerHour / 3600.
    let beatsPerSecond: Double
    /// Rate deviation: positive = fast, negative = slow (s/day).
    let rateSecondsPerDay: Double
    /// 1σ uncertainty on the rate (s/day).
    let rateUncertainty: Double
    /// Lock-in amplitude SNR — how cleanly the tick stands above the noise.
    let amplitudeSNR: Double
    /// Tick-line prominence over the local noise floor (signal quality).
    let lineSeparation: Double
    /// 0…1 overall quality estimate.
    let confidence: Double
    /// Seconds of audio analyzed.
    let elapsedSeconds: Double
    /// True once the signal is clean enough to trust the reading.
    let isCalibrated: Bool
    /// The auto-selected band-pass window (Hz) the reading was taken from.
    let bandLowHz: Double
    let bandHighHz: Double
    /// Cumulative timing-offset trace for the drift visualization.
    let trace: [PhasePoint]
}

/// A second-order IIR biquad (transposed direct form II).
struct Biquad {
    var b0 = 1.0, b1 = 0.0, b2 = 0.0, a1 = 0.0, a2 = 0.0
    var z1 = 0.0, z2 = 0.0

    mutating func reset() { z1 = 0; z2 = 0 }

    mutating func process(_ x: Double) -> Double {
        let y = b0 * x + z1
        z1 = b1 * x - a1 * y + z2
        z2 = b2 * x - a2 * y
        return y
    }

    static func highpass(fc: Double, fs: Double, q: Double = 0.707) -> Biquad {
        let w0 = 2 * Double.pi * fc / fs
        let cw = cos(w0), sw = sin(w0), alpha = sw / (2 * q)
        let a0 = 1 + alpha
        return Biquad(b0: ((1 + cw) / 2) / a0, b1: (-(1 + cw)) / a0, b2: ((1 + cw) / 2) / a0,
                      a1: (-2 * cw) / a0, a2: (1 - alpha) / a0)
    }

    static func lowpass(fc: Double, fs: Double, q: Double = 0.707) -> Biquad {
        let w0 = 2 * Double.pi * fc / fs
        let cw = cos(w0), sw = sin(w0), alpha = sw / (2 * q)
        let a0 = 1 + alpha
        return Biquad(b0: ((1 - cw) / 2) / a0, b1: (1 - cw) / a0, b2: ((1 - cw) / 2) / a0,
                      a1: (-2 * cw) / a0, a2: (1 - alpha) / a0)
    }
}

/// Acoustic timegrapher.
///
/// The escapement tick's discriminating energy sits high (~5–15 kHz); lower
/// frequencies are dominated by handling/room rumble. So: band-pass 5–15 kHz →
/// square (energy) → decimate to an ~8 kHz envelope. The beat rate is found by a
/// sharp DFT over candidate rates, and the rate deviation by lock-in
/// phase-tracking at the nominal beat frequency — which integrates over every
/// beat and stays robust even when individual ticks are buried in noise.
final class Timegrapher {

    static let standardRates: [(bph: Int, bps: Double)] = [
        (18000, 5.0), (19800, 5.5), (21600, 6.0),
        (25200, 7.0), (28800, 8.0), (36000, 10.0),
    ]

    static let standardBPH = standardRates.map { $0.bph }

    /// Candidate band-pass windows scanned per measurement. The escapement's
    /// discriminating tick energy lands anywhere from ~5 kHz to ~20 kHz
    /// depending on the movement/case, so we pick the band with the strongest
    /// beat line rather than assuming one.
    static let candidateBands: [(lo: Double, hi: Double)] = [
        (2_000, 6_000), (4_000, 9_000), (5_000, 11_000), (7_000, 13_000),
        (9_000, 15_000), (11_000, 18_000), (13_000, 21_000),
    ]

    // Envelope rate — the beat and its harmonics live below ~40 Hz, so ~1 kHz is
    // plenty and keeps the DFT/lock-in cheap on the live path.
    static let envTargetRate = 1_000.0

    // Config
    private var sampleRate = 48_000.0
    private var manualBPH: Int?
    private let maxSeconds = 180.0

    // Raw mono audio, guarded by lock (filtering happens in analyze()).
    private var raw: [Float] = []
    private let lock = NSLock()

    // Incremental state: the band is chosen once, then the chosen-band envelope
    // is extended with only the new audio each pass (so analyze() stays cheap as
    // the recording grows, instead of re-filtering everything).
    private var bandChosen = false
    private var chLo = 5_000.0, chHi = 15_000.0, chBPH = 28_800
    private var chSep = 1.0, chProm = 0.0
    private var incHP = Biquad(), incLP = Biquad()
    private var incAcc = 0.0, incCount = 0, incDecim = 48
    private var incEnvRate = 1_000.0
    private var incEnv: [Double] = []
    private var processedRaw = 0

    // MARK: Setup

    func reset(sampleRate: Double) {
        lock.lock()
        self.sampleRate = sampleRate
        raw.removeAll(keepingCapacity: true)
        raw.reserveCapacity(Int(sampleRate * 60))
        bandChosen = false
        incEnv.removeAll(keepingCapacity: true)
        incAcc = 0; incCount = 0; processedRaw = 0
        lock.unlock()
    }

    func setManualBPH(_ bph: Int?) {
        lock.lock(); manualBPH = bph; lock.unlock()
    }

    // MARK: Ingest (audio thread) — just store raw; filtering is deferred.

    func process(_ samples: UnsafePointer<Float>, count: Int) {
        lock.lock()
        if Double(raw.count) < sampleRate * maxSeconds {
            raw.append(contentsOf: UnsafeBufferPointer(start: samples, count: count))
        }
        lock.unlock()
    }

    /// Band-pass (HP→LP) → square → decimate to ~8 kHz, returning a mean-removed
    /// energy envelope and its rate. `count` limits how much of `src` to use.
    private func envelope(_ src: [Float], count: Int, fs: Double, lo: Double, hi: Double) -> (env: [Double], er: Double) {
        var hp = Biquad.highpass(fc: lo, fs: fs)
        var lp = Biquad.lowpass(fc: min(hi, fs / 2 - 500), fs: fs)
        let decim = max(1, Int((fs / Timegrapher.envTargetRate).rounded()))
        let er = fs / Double(decim)
        var env = [Double](); env.reserveCapacity(count / decim + 1)
        var acc = 0.0, c = 0
        for i in 0..<count {
            let y = lp.process(hp.process(Double(src[i])))
            acc += y * y; c += 1
            if c >= decim { env.append(acc / Double(decim)); acc = 0; c = 0 }
        }
        var m = 0.0
        for v in env { m += v }
        if !env.isEmpty { m /= Double(env.count); for i in 0..<env.count { env[i] -= m } }
        return (env, er)
    }

    /// Pick the (band, bph) with the strongest beat line over the first ~12 s.
    private func selectBand(_ src: [Float], fs: Double, manual: Int?)
        -> (lo: Double, hi: Double, bph: Int, separation: Double, prominence: Double) {
        let scanCount = min(src.count, Int(12 * fs))
        let cands = manual.map { [$0] } ?? Timegrapher.standardBPH
        var best = (lo: 5_000.0, hi: 15_000.0, bph: manual ?? 28_800, separation: 1.0, prominence: 0.0)
        var bestProm = -1.0
        for band in Timegrapher.candidateBands {
            let (env, er) = envelope(src, count: scanCount, fs: fs, lo: band.lo, hi: band.hi)
            if env.count < Int(er) { continue }
            var mags = cands.map { (bph: $0, mag: dftMag(env, fs: er, f: Double($0) / 3600, window: env.count)) }
            mags.sort { $0.mag > $1.mag }
            let bph = mags[0].bph
            let sep = mags.count > 1 ? mags[0].mag / max(mags[1].mag, 1e-12) : 999
            let f0 = Double(bph) / 3600
            let noiseF = [f0 * 0.6, f0 * 0.75, f0 * 1.2, f0 * 1.35, f0 * 1.6]
            var nb = 0.0
            for nf in noiseF { nb += dftMag(env, fs: er, f: nf, window: env.count) }
            nb /= Double(noiseF.count)
            let prom = mags[0].mag / (nb + 1e-12)
            if prom > bestProm { bestProm = prom; best = (band.lo, band.hi, bph, sep, prom) }
        }
        return best
    }

    // MARK: Analyze (background queue)

    func analyze() -> TimegrapherResult? {
        lock.lock()
        let n = raw.count
        let fs = sampleRate
        let manual = manualBPH
        let elapsed = Double(n) / fs
        // Snapshot only what's needed: everything (once) to choose the band, or
        // just the new tail thereafter — avoids copying the whole buffer.
        var fullSrc: [Float] = []
        var tail: [Float] = []
        if !bandChosen {
            if elapsed >= 5 { fullSrc = raw }
        } else if n > processedRaw {
            tail = Array(raw[processedRaw..<n])
        }
        lock.unlock()

        guard elapsed > 3.0 else { return nil }

        if !bandChosen {
            guard elapsed >= 5, !fullSrc.isEmpty else { return nil }
            let sel = selectBand(fullSrc, fs: fs, manual: manual)
            chLo = sel.lo; chHi = sel.hi; chBPH = sel.bph
            chSep = sel.separation; chProm = sel.prominence
            incDecim = max(1, Int((fs / Timegrapher.envTargetRate).rounded()))
            incEnvRate = fs / Double(incDecim)
            incHP = .highpass(fc: chLo, fs: fs)
            incLP = .lowpass(fc: min(chHi, fs / 2 - 500), fs: fs)
            incAcc = 0; incCount = 0; incEnv.removeAll(keepingCapacity: true)
            extendEnv(fullSrc, from: 0, to: fullSrc.count)
            processedRaw = fullSrc.count
            bandChosen = true
        } else if !tail.isEmpty {
            extendEnv(tail, from: 0, to: tail.count)
            processedRaw = n
        }

        let f0 = Double(chBPH) / 3600.0
        guard incEnv.count > Int(incEnvRate * 1.5),
              let li = lockIn(incEnv, fs: incEnvRate, f0: f0) else { return nil }

        let rate = 86_400.0 * li.slope / (2 * Double.pi * f0)
        // Honest uncertainty: the phase wanders with a multi-second correlation
        // time, so a residual-based SE is wildly overconfident. Instead take the
        // scatter of independent sub-window rates.
        let rateUnc = li.rateUncPerDay

        // --- Signal quality (0…1): is the tick clearly heard? ---
        // Whether the *reading* is trustworthy also depends on how settled it is
        // over time, which the view model judges from the live reading history —
        // the sub-window ± is too pessimistic to gate on directly (it counts
        // short-term wander that averages out).
        let snrTerm = max(0.0, min(1.0, (li.ampSNR - 1.0) / 4.0))
        let promTerm = max(0.0, min(1.0, (chProm - 3.0) / 12.0))
        let confidence = snrTerm * promTerm
        let isCalibrated = elapsed > 8 && li.ampSNR > 1.8 && chProm > 5

        return TimegrapherResult(
            beatsPerHour: chBPH,
            beatsPerSecond: f0,
            rateSecondsPerDay: rate,
            rateUncertainty: rateUnc,
            amplitudeSNR: li.ampSNR,
            lineSeparation: chProm,
            confidence: confidence,
            elapsedSeconds: elapsed,
            isCalibrated: isCalibrated,
            bandLowHz: chLo,
            bandHighHz: chHi,
            trace: li.trace
        )
    }

    /// Extend the chosen-band envelope with src[from..<to], carrying filter and
    /// decimation state so the envelope is continuous across calls.
    private func extendEnv(_ src: [Float], from: Int, to: Int) {
        var hp = incHP, lp = incLP
        var acc = incAcc, c = incCount
        let d = incDecim
        for i in from..<to {
            let y = lp.process(hp.process(Double(src[i])))
            acc += y * y; c += 1
            if c >= d { incEnv.append(acc / Double(d)); acc = 0; c = 0 }
        }
        incHP = hp; incLP = lp; incAcc = acc; incCount = c
    }

    // MARK: - DSP helpers

    /// Magnitude of the envelope's DFT bin at frequency f (over the first
    /// `window` samples). Sharp frequency resolution cleanly separates adjacent
    /// candidate rates (e.g. 7 vs 8 Hz).
    private func dftMag(_ e: [Double], fs: Double, f: Double, window: Int) -> Double {
        let n = min(e.count, window)
        let w = 2 * Double.pi * f / fs
        var re = 0.0, im = 0.0
        // Incremental phasor to avoid a cos/sin per sample.
        var cr = 1.0, ci = 0.0
        let dc = cos(w), ds = sin(w)
        for i in 0..<n {
            re += e[i] * cr
            im -= e[i] * ci
            let ncr = cr * dc - ci * ds
            ci = cr * ds + ci * dc
            cr = ncr
            if i & 8191 == 0 {   // renormalize the phasor periodically
                let m = (cr * cr + ci * ci).squareRoot()
                if m > 0 { cr /= m; ci /= m }
            }
        }
        return (re * re + im * im).squareRoot() / Double(n)
    }

    private struct LockIn {
        let slope: Double          // rad/sec of phase drift
        let rateUncPerDay: Double  // honest ± from sub-window scatter (s/day)
        let ampSNR: Double
        let trace: [PhasePoint]
    }

    /// Complex-demodulate at f0, low-pass to the slow amplitude/phase, then fit
    /// the unwrapped phase vs time. Slope = frequency offset → rate.
    private func lockIn(_ e: [Double], fs: Double, f0: Double) -> LockIn? {
        let n = e.count
        let w = 2 * Double.pi * f0 / fs
        var zr = [Double](repeating: 0, count: n)
        var zi = [Double](repeating: 0, count: n)
        var cr = 1.0, ci = 0.0
        let dc = cos(w), ds = sin(w)
        for i in 0..<n {
            zr[i] = e[i] * cr          // e * cos
            zi[i] = -e[i] * ci         // e * (-sin)
            let ncr = cr * dc - ci * ds
            ci = cr * ds + ci * dc
            cr = ncr
            if i & 8191 == 0 {
                let m = (cr * cr + ci * ci).squareRoot()
                if m > 0 { cr /= m; ci /= m }
            }
        }

        // Low-pass both quadratures to isolate the slowly varying amplitude.
        var lpR = Biquad.lowpass(fc: 1.5, fs: fs)
        var lpI = Biquad.lowpass(fc: 1.5, fs: fs)
        for i in 0..<n { zr[i] = lpR.process(zr[i]) }
        for i in 0..<n { zi[i] = lpI.process(zi[i]) }

        // Trim 10% at each end (filter transients).
        let a = Int(0.1 * Double(n)), b = Int(0.9 * Double(n))
        guard b - a > Int(fs) else { return nil }

        // Unwrap phase and accumulate stats for an amplitude-weighted fit of
        // phase vs time — down-weighting low-amplitude stretches (momentary
        // coupling dropouts) where the phase estimate is unreliable.
        var prev = atan2(zi[a], zr[a])
        var unwrapped = prev
        var sumAmp = 0.0, sumAmp2 = 0.0, cnt = 0.0
        var sw = 0.0, swx = 0.0, swy = 0.0, swxx = 0.0, swxy = 0.0
        var phases: [Double] = []
        phases.reserveCapacity(b - a)
        for i in a..<b {
            let ph = atan2(zi[i], zr[i])
            var d = ph - prev
            while d > Double.pi { d -= 2 * Double.pi }
            while d < -Double.pi { d += 2 * Double.pi }
            unwrapped += d
            prev = ph
            phases.append(unwrapped)

            let t = Double(i) / fs
            let amp = (zr[i] * zr[i] + zi[i] * zi[i]).squareRoot()
            sw += amp; swx += amp * t; swy += amp * unwrapped
            swxx += amp * t * t; swxy += amp * t * unwrapped
            sumAmp += amp; sumAmp2 += amp * amp; cnt += 1
        }
        let sxxCentered = swxx - swx * swx / sw
        guard sxxCentered > 0 else { return nil }
        let slope = (swxy - swx * swy / sw) / sxxCentered

        let ampMean = sumAmp / cnt
        let ampVar = max(0, sumAmp2 / cnt - ampMean * ampMean)
        let ampSNR = ampMean / (ampVar.squareRoot() + 1e-12)

        // Uncertainty from the scatter of independent ~8 s sub-window rates.
        let toRate = 86_400.0 / (2 * Double.pi * f0)
        let chunk = Int(8 * fs)
        var chunkRates: [Double] = []
        var s0 = 0
        while s0 + chunk <= phases.count {
            var cx = 0.0, cy = 0.0, cxx = 0.0, cxy = 0.0, cc = 0.0
            for k in s0..<(s0 + chunk) {
                let t = Double(a + k) / fs
                cx += t; cy += phases[k]; cxx += t * t; cxy += t * phases[k]; cc += 1
            }
            let den = cxx - cx * cx / cc
            if den > 0 { chunkRates.append((cxy - cx * cy / cc) / den * toRate) }
            s0 += chunk
        }
        let rateUncPerDay: Double
        if chunkRates.count >= 2 {
            let m = chunkRates.reduce(0, +) / Double(chunkRates.count)
            let v = chunkRates.reduce(0) { $0 + ($1 - m) * ($1 - m) } / Double(chunkRates.count - 1)
            rateUncPerDay = v.squareRoot() / Double(chunkRates.count).squareRoot()
        } else {
            rateUncPerDay = 10.0   // too short to trust
        }

        // Build a downsampled drift trace (cumulative offset in ms).
        var trace: [PhasePoint] = []
        let stride = max(1, phases.count / 120)
        let toMs = 1000.0 / (2 * Double.pi * f0)
        let ph0 = phases.first ?? 0
        var k = 0
        while k < phases.count {
            let i = a + k
            trace.append(PhasePoint(time: Double(i) / fs, offsetMs: (phases[k] - ph0) * toMs))
            k += stride
        }

        return LockIn(slope: slope, rateUncPerDay: rateUncPerDay, ampSNR: ampSNR, trace: trace)
    }
}
