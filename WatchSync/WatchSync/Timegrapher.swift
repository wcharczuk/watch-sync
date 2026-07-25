import Foundation

/// One independent sub-window measurement: a few seconds of beats fitted on
/// their own. These are exactly what the ± is computed from.
struct RateSample {
    let time: Double        // seconds into the recording (block centre)
    let rate: Double        // s/day from this block alone
}

/// How well the tick was heard over a short window — the honest report on the
/// *measurement*, as opposed to any verdict about the watch.
struct QualitySample {
    let time: Double        // seconds into the recording (window centre)
    /// Fraction of expected beats that matched the tick template, 0…1.
    let detection: Double
    /// Timing scatter of the beats that did match (ms). This is what limits
    /// accuracy; detection alone can dip without the reading suffering.
    let jitterMs: Double
}

/// What the measurement is doing right now. One source of truth — the view
/// renders this instead of forming its own opinion from the raw metrics.
enum MeasurementStage {
    /// Audio is live but there isn't enough of it to work with yet.
    case listening
    /// Choosing the listening band and identifying the beat rate.
    case tuning
    /// Beat rate known; timing individual ticks, no number to stand behind yet.
    case locking
    /// Tracking, with a published rate that is still tightening.
    case measuring
    /// The rate has reached the target precision.
    case done
    /// Tracking, but ticks are being lost — the user can fix this.
    case unstable
    /// Nothing that looks like an escapement, after a fair amount of listening.
    case noSignal
}

/// Result of one analysis pass.
struct TimegrapherResult {
    let stage: MeasurementStage

    /// Nominal beat rate (auto-detected or manual override); 0 until identified.
    let beatsPerHour: Int
    /// Rate deviation: positive = fast, negative = slow (s/day). nil until there
    /// is enough independent data to stand behind a number.
    let rateSecondsPerDay: Double?
    /// Published ± on the rate (s/day).
    let uncertainty: Double?
    /// Tic-to-toc asymmetry (ms) — the classic timegrapher beat-error figure.
    let beatErrorMs: Double?

    // Signal quality — "can I hear it", kept separate from "how sure is the number".
    /// Per-beat timing scatter (ms). Under ~0.15 ms is a clean tick.
    let jitterMs: Double
    /// Fraction of expected beats that matched the tick template.
    let detectionRate: Double
    /// Median template match score (0…1).
    let matchScore: Double

    let beatsTracked: Int
    let elapsedSeconds: Double
    let bandLowHz: Double
    let bandHighHz: Double

    /// 0…1 toward the target precision. Monotonic by construction.
    let progress: Double
    /// Estimated seconds still needed to reach the target precision.
    let secondsRemaining: Double?

    /// Independent sub-window rates.
    let rateSamples: [RateSample]
    /// Per-window tick-detection quality, for the chart.
    let qualitySamples: [QualitySample]
    /// How many times the listening band had to be re-chosen mid-measurement.
    let retuneCount: Int
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

// MARK: - Beat tracker

/// Times every beat in one listening band.
///
/// Band-pass → rectify → decimate to a 4 kHz envelope; build a tick template by
/// averaging beats; matched-filter each beat, predicting where it should fall
/// from a running fit of the beats already timed — so the period refines itself
/// and no separate period estimator is needed. The rate is the slope of beat
/// time vs beat number.
private final class BeatTracker {
    static let templateMs = 14.0     // tick template length
    static let templatePreMs = 2.0   // template starts this far before the peak
    static let searchMs = 12.0       // ± search window around a predicted beat
    static let minScore = 0.35       // template match floor for accepting a beat

    let lo: Double, hi: Double, bph: Int
    let nominalCycle: Double         // seconds per tic-toc cycle
    private(set) var envRate: Double

    private var hp: Biquad, lp: Biquad, smooth: Biquad
    private var acc = 0.0, count = 0
    private let decim: Int
    private(set) var env: [Double] = []

    private(set) var template: [Double] = []
    private(set) var beatK: [Int] = []
    private(set) var beatT: [Double] = []
    private(set) var beatScore: [Double] = []
    private(set) var beatKeep: [Bool] = []
    private var nextBeat = 0

    private(set) var slope = 0.0         // seconds per cycle
    private(set) var intercept = 0.0
    private(set) var haveFit = false
    /// Beat count at the last refold; 0 = never folded.
    private(set) var refolded = 0
    private(set) var beatErrorMs: Double?

    init(lo: Double, hi: Double, bph: Int, sampleRate: Double, targetEnvRate: Double) {
        self.lo = lo
        self.hi = hi
        self.bph = bph
        self.nominalCycle = 7200.0 / Double(bph)
        self.decim = max(1, Int((sampleRate / targetEnvRate).rounded()))
        self.envRate = sampleRate / Double(decim)
        self.hp = .highpass(fc: lo, fs: sampleRate)
        self.lp = .lowpass(fc: min(hi, sampleRate / 2 - 500), fs: sampleRate)
        self.smooth = .lowpass(fc: envRate / 3, fs: sampleRate)
    }

    private var templateLength: Int { max(8, Int(Self.templateMs * envRate / 1000)) }
    private var templatePre: Int { Int(Self.templatePreMs * envRate / 1000) }
    private var searchSamples: Int { max(2, Int(Self.searchMs * envRate / 1000)) }

    var periodSamples: Double { nominalCycle * envRate }
    var envSeconds: Double { Double(env.count) / envRate }

    // MARK: Envelope

    func extend(_ src: [Float]) {
        env.reserveCapacity(env.count + src.count / decim + 1)
        for v in src {
            let band = lp.process(hp.process(Double(v)))
            let y = smooth.process(abs(band))
            acc += y; count += 1
            if count >= decim { env.append(acc / Double(decim)); acc = 0; count = 0 }
        }
    }

    // MARK: Tracking

    /// Advance the tracker over whatever envelope has arrived. Returns false if
    /// there is still nothing that looks like a tick.
    @discardableResult
    func step() -> Bool {
        if template.isEmpty {
            guard envSeconds >= 2.5, bootstrapTemplate() else { return false }
        }
        trackNewBeats()
        refit()
        // Refold whenever the beat count has doubled since the last one: each
        // refold averages more beats, so the template keeps sharpening, and one
        // bad early template can't poison the whole measurement.
        if acceptedCount >= max(60, 2 * refolded) {
            if refoldTemplate() {
                refolded = acceptedCount
                retrackAll()
            } else {
                refolded = max(refolded, acceptedCount)
            }
        }
        return true
    }

    /// Autocorrelation period, searched several cycles out for precision.
    ///
    /// The nominal period is only as good as the watch's own rate error — fine
    /// for tracking, not for *folding*: across a couple of dozen cycles that
    /// error smears the averaged tick. Locking the period first makes the very
    /// first template as sharp as the ones that come later.
    private func refinePeriod() -> Double {
        let p0 = periodSamples
        let n = env.count
        var mean = 0.0
        for v in env { mean += v }
        mean /= Double(n)
        let k = max(1, min(12, Int(Double(n) / (2 * p0))))
        let tol = 0.005                     // 0.5% covers any real watch
        let lo = Int(Double(k) * p0 * (1 - tol))
        let hi = Int(Double(k) * p0 * (1 + tol)) + 1
        guard lo > 0, hi + 4 < n else { return p0 }
        var ac = [Double](repeating: 0, count: hi - lo + 1)
        for (idx, lag) in (lo...hi).enumerated() {
            var s = 0.0
            for i in 0..<(n - lag) { s += (env[i] - mean) * (env[i + lag] - mean) }
            ac[idx] = s / Double(n - lag)
        }
        var best = 0
        for i in 0..<ac.count where ac[i] > ac[best] { best = i }
        var frac = 0.0
        if best > 0, best < ac.count - 1 {
            let y0 = ac[best - 1], y1 = ac[best], y2 = ac[best + 1]
            let den = y0 - 2 * y1 + y2
            if den != 0 { frac = 0.5 * (y0 - y2) / den }
        }
        return (Double(lo + best) + frac) / Double(k)
    }

    /// Build the first template by folding, not by grabbing the loudest peak:
    /// the loudest peak in the first seconds is as likely to be a knock as a
    /// tick, whereas folding averages every beat heard so far.
    private func bootstrapTemplate() -> Bool {
        let period = refinePeriod()
        let L = Int(period)
        guard L > templateLength * 2, env.count >= 3 * L else { return false }
        let cycles = Int(Double(env.count) / period)
        guard cycles >= 3 else { return false }
        var acc = [Double](repeating: 0, count: L)
        var folded = 0
        for c in 0..<cycles {
            let s = Int((Double(c) * period).rounded())
            guard s + L <= env.count else { continue }
            for j in 0..<L { acc[j] += env[s + j] }
            folded += 1
        }
        guard folded >= 3 else { return false }
        var wf = acc.map { $0 / Double(folded) }
        var sorted = wf
        sorted.sort()
        let baseline = sorted[sorted.count / 2]
        for i in 0..<L { wf[i] -= baseline }

        var peak = 0
        for i in 0..<L where wf[i] > wf[peak] { peak = i }
        var absMean = 0.0
        for v in wf { absMean += abs(v) }
        absMean /= Double(L)
        // A silent or hiss-only recording folds flat — nothing to lock onto.
        guard wf[peak] > 2 * absMean else { return false }

        var tpl = [Double](repeating: 0, count: templateLength)
        for i in 0..<templateLength {
            tpl[i] = wf[((peak - templatePre + i) % L + L) % L]
        }
        template = tpl
        // The fold is phase-locked to the start of the envelope, so beat zero
        // sits at the template's own offset within the first cycle.
        slope = period / envRate
        intercept = Double(max(0, peak - templatePre)) / envRate
        haveFit = true
        nextBeat = 0
        return true
    }

    private func retrackAll() {
        beatK.removeAll(keepingCapacity: true)
        beatT.removeAll(keepingCapacity: true)
        beatScore.removeAll(keepingCapacity: true)
        beatKeep.removeAll(keepingCapacity: true)
        nextBeat = 0
        trackNewBeats()
        refit()
    }

    /// A fitted period this far from nominal isn't a watch — it's a fit that has
    /// come apart on noise. 2% is 1728 s/day, so nothing real is ever rejected.
    private var slopeIsPlausible: Bool {
        slope > nominalCycle * 0.98 && slope < nominalCycle * 1.02
    }

    private func trackNewBeats() {
        guard !template.isEmpty, slopeIsPlausible else { return }
        let L = template.count
        let search = searchSamples
        var tpl = template
        var tmean = 0.0
        for v in tpl { tmean += v }
        tmean /= Double(L)
        for i in 0..<L { tpl[i] -= tmean }
        var tnorm = 0.0
        for v in tpl { tnorm += v * v }
        tnorm = tnorm.squareRoot()
        guard tnorm > 0 else { return }

        var corr = [Double](repeating: 0, count: 2 * search + 1)
        while true {
            let predicted = intercept + slope * Double(nextBeat)
            let centre = Int((predicted * envRate).rounded())
            let from = centre - search
            guard centre + search + L < env.count else { break }
            if from < 0 { nextBeat += 1; continue }

            var bestJ = 0
            for j in 0...(2 * search) {
                let s = from + j
                var mean = 0.0
                for i in 0..<L { mean += env[s + i] }
                mean /= Double(L)
                var dot = 0.0, norm = 0.0
                for i in 0..<L {
                    let d = env[s + i] - mean
                    dot += d * tpl[i]
                    norm += d * d
                }
                corr[j] = dot / (norm.squareRoot() * tnorm + 1e-12)
                if corr[j] > corr[bestJ] { bestJ = j }
            }
            var frac = 0.0
            if bestJ > 0, bestJ < 2 * search {
                let y0 = corr[bestJ - 1], y1 = corr[bestJ], y2 = corr[bestJ + 1]
                let den = y0 - 2 * y1 + y2
                if den != 0 { frac = 0.5 * (y0 - y2) / den }
            }
            beatK.append(nextBeat)
            beatT.append((Double(from + bestJ) + frac) / envRate)
            beatScore.append(corr[bestJ])
            beatKeep.append(corr[bestJ] >= Self.minScore)
            nextBeat += 1
        }
    }

    /// Iteratively reweighted least squares of beat time vs beat number. The
    /// reweighting drops beats swamped by a knock or a momentary dropout.
    private func refit() {
        guard beatK.count >= 8 else { return }
        var keep = beatKeep
        for _ in 0..<4 {
            var sw = 0.0, sx = 0.0, sy = 0.0, sxx = 0.0, sxy = 0.0
            for i in 0..<beatK.count where keep[i] {
                let x = Double(beatK[i]), y = beatT[i]
                sw += 1; sx += x; sy += y; sxx += x * x; sxy += x * y
            }
            guard sw >= 4 else { return }
            let den = sxx - sx * sx / sw
            guard den > 0 else { return }
            // Keep the previous fit rather than accept an implausible period: a
            // runaway slope would send the next tracking pass hunting for beats
            // that will never arrive.
            let newSlope = (sxy - sx * sy / sw) / den
            guard newSlope > nominalCycle * 0.98, newSlope < nominalCycle * 1.02 else { return }
            let newIntercept = (sy - newSlope * sx) / sw

            var resid = [Double](repeating: 0, count: beatK.count)
            for i in 0..<beatK.count {
                resid[i] = beatT[i] - (newIntercept + newSlope * Double(beatK[i]))
            }
            let scale = 1.4826 * medianAbsDeviation(resid, keep: keep) + 1e-9
            for i in 0..<beatK.count {
                keep[i] = beatScore[i] >= Self.minScore && abs(resid[i]) < 3 * scale
            }
            slope = newSlope
            intercept = newIntercept
            haveFit = true
        }
        beatKeep = keep
    }

    private func medianAbsDeviation(_ v: [Double], keep: [Bool]) -> Double {
        var vals: [Double] = []
        vals.reserveCapacity(v.count)
        for i in 0..<v.count where keep[i] { vals.append(v[i]) }
        guard !vals.isEmpty else { return 0 }
        vals.sort()
        let med = vals[vals.count / 2]
        var dev = vals.map { abs($0 - med) }
        dev.sort()
        return dev[dev.count / 2]
    }

    /// Average every tracked beat at the fitted period for a clean template, and
    /// read the tic-to-toc spacing (beat error) off the same folded waveform.
    private func refoldTemplate() -> Bool {
        guard haveFit, slopeIsPlausible else { return false }
        let L = Int(slope * envRate)
        guard L > templateLength * 2 else { return false }
        var acc = [Double](repeating: 0, count: L)
        var folded = 0
        for i in 0..<beatK.count where beatKeep[i] {
            let start = Int(((intercept + slope * Double(beatK[i])) * envRate).rounded())
            guard start >= 0, start + L < env.count else { continue }
            for j in 0..<L { acc[j] += env[start + j] }
            folded += 1
        }
        guard folded >= 30 else { return false }
        var wf = acc.map { $0 / Double(folded) }
        var sorted = wf
        sorted.sort()
        let baseline = sorted[sorted.count / 2]
        for i in 0..<L { wf[i] -= baseline }

        var peak = 0
        for i in 0..<L where wf[i] > wf[peak] { peak = i }
        var tpl = [Double](repeating: 0, count: templateLength)
        for i in 0..<templateLength {
            tpl[i] = wf[((peak - templatePre + i) % L + L) % L]
        }
        template = tpl
        beatErrorMs = Self.beatError(wf, sampleRate: envRate)
        // The refold moves where the template's origin sits inside the beat, so
        // re-anchor the fit to keep predictions centred.
        intercept += Double(peak - templatePre) / envRate
        return true
    }

    /// Tic-to-toc spacing against half a cycle, from the folded waveform. Found
    /// by circular autocorrelation rather than by picking a second peak: an
    /// escapement makes several sounds per beat, so the runner-up peak is
    /// usually the tick's own drop, not the toc.
    static func beatError(_ wf: [Double], sampleRate: Double) -> Double? {
        let n = wf.count
        guard n > 16 else { return nil }
        var mean = 0.0
        for v in wf { mean += v }
        mean /= Double(n)
        let w = wf.map { $0 - mean }
        let half = n / 2
        let span = max(4, Int(0.02 * sampleRate))
        let lo = max(1, half - span), hi = min(n - 1, half + span)
        guard hi > lo + 2 else { return nil }
        var ac = [Double](repeating: 0, count: hi - lo + 1)
        for (idx, lag) in (lo...hi).enumerated() {
            var s = 0.0
            for i in 0..<n { s += w[i] * w[(i + lag) % n] }
            ac[idx] = s
        }
        var best = 0
        for i in 0..<ac.count where ac[i] > ac[best] { best = i }
        var frac = 0.0
        if best > 0, best < ac.count - 1 {
            let y0 = ac[best - 1], y1 = ac[best], y2 = ac[best + 1]
            let den = y0 - 2 * y1 + y2
            if den != 0 { frac = 0.5 * (y0 - y2) / den }
        }
        let ticToToc = Double(lo + best) + frac
        return abs(ticToToc - Double(half)) / sampleRate * 1000
    }

    // MARK: Quality

    var acceptedCount: Int {
        var c = 0
        for k in beatKeep where k { c += 1 }
        return c
    }

    /// Per-beat timing scatter (ms), match rate and median match score.
    func quality() -> (jitterMs: Double, detection: Double, score: Double) {
        guard !beatK.isEmpty else { return (0, 0, 0) }
        var resid: [Double] = [], scores: [Double] = []
        for i in 0..<beatK.count where beatKeep[i] {
            resid.append(beatT[i] - (intercept + slope * Double(beatK[i])))
            scores.append(beatScore[i])
        }
        var jitter = 0.0
        if resid.count > 2 {
            var m = 0.0
            for v in resid { m += v }
            m /= Double(resid.count)
            var s = 0.0
            for v in resid { s += (v - m) * (v - m) }
            jitter = (s / Double(resid.count - 1)).squareRoot() * 1000
        }
        scores.sort()
        let med = scores.isEmpty ? 0 : scores[scores.count / 2]
        return (jitter, Double(resid.count) / Double(beatK.count), med)
    }

    var rateSecondsPerDay: Double? {
        guard haveFit, slope > 0, acceptedCount >= 20 else { return nil }
        return (nominalCycle / slope - 1) * 86_400
    }

    /// Detection rate and timing jitter over short windows, so the chart can
    /// show how well the tick is being heard as the measurement proceeds.
    func qualitySamples(window: Double) -> [QualitySample] {
        var out: [QualitySample] = []
        guard !beatT.isEmpty else { return out }
        var t0 = beatT[0]
        let last = beatT[beatT.count - 1]
        while t0 < last {
            var total = 0, kept = 0
            var resid: [Double] = []
            for i in 0..<beatK.count where beatT[i] >= t0 && beatT[i] < t0 + window {
                total += 1
                if beatKeep[i] {
                    kept += 1
                    resid.append(beatT[i] - (intercept + slope * Double(beatK[i])))
                }
            }
            if total >= 4 {
                var jitter = 0.0
                if resid.count > 2 {
                    var m = 0.0
                    for v in resid { m += v }
                    m /= Double(resid.count)
                    var sq = 0.0
                    for v in resid { sq += (v - m) * (v - m) }
                    jitter = (sq / Double(resid.count - 1)).squareRoot() * 1000
                }
                out.append(QualitySample(time: t0 + window / 2,
                                         detection: Double(kept) / Double(total),
                                         jitterMs: jitter))
            }
            t0 += window
        }
        return out
    }

    /// Rates fitted independently in non-overlapping blocks, each tagged with
    /// the middle of the window it came from.
    func blockSamples(tau: Double) -> [RateSample] {
        var out: [RateSample] = []
        guard !beatT.isEmpty else { return out }
        var t0 = beatT[0]
        let last = beatT[beatT.count - 1]
        while t0 <= last - tau {
            var sw = 0.0, sx = 0.0, sy = 0.0, sxx = 0.0, sxy = 0.0
            for i in 0..<beatK.count where beatKeep[i] && beatT[i] >= t0 && beatT[i] < t0 + tau {
                let x = Double(beatK[i]), y = beatT[i]
                sw += 1; sx += x; sy += y; sxx += x * x; sxy += x * y
            }
            if sw >= 8 {
                let den = sxx - sx * sx / sw
                if den > 0 {
                    let slope = (sxy - sx * sy / sw) / den
                    if slope > 0 {
                        out.append(RateSample(time: t0 + tau / 2,
                                              rate: (nominalCycle / slope - 1) * 86_400))
                    }
                }
            }
            t0 += tau
        }
        return out
    }

    /// Rates fitted independently in blocks of `tau` seconds of beats.
    func blockRates(tau: Double, step: Double) -> [Double] {
        var rates: [Double] = []
        guard !beatT.isEmpty else { return rates }
        var t0 = beatT[0]
        let last = beatT[beatT.count - 1]
        while t0 <= last - tau {
            var sw = 0.0, sx = 0.0, sy = 0.0, sxx = 0.0, sxy = 0.0
            for i in 0..<beatK.count where beatKeep[i] && beatT[i] >= t0 && beatT[i] < t0 + tau {
                let x = Double(beatK[i]), y = beatT[i]
                sw += 1; sx += x; sy += y; sxx += x * x; sxy += x * y
            }
            if sw >= 8 {
                let den = sxx - sx * sx / sw
                if den > 0 {
                    let s = (sxy - sx * sy / sw) / den
                    if s > 0 { rates.append((nominalCycle / s - 1) * 86_400) }
                }
            }
            t0 += step
        }
        return rates
    }

    /// Textbook SE of the fitted slope in s/day. Far too optimistic on its own —
    /// beat timings are correlated — but it scales correctly with time, so it
    /// serves as a calibrated floor under the block-scatter estimate.
    func naiveSE() -> Double? {
        var ks: [Double] = [], resid: [Double] = []
        for i in 0..<beatK.count where beatKeep[i] {
            ks.append(Double(beatK[i]))
            resid.append(beatT[i] - (intercept + slope * Double(beatK[i])))
        }
        guard ks.count > 8 else { return nil }
        var km = 0.0, rm = 0.0
        for v in ks { km += v }
        km /= Double(ks.count)
        for v in resid { rm += v }
        rm /= Double(resid.count)
        var kvar = 0.0, rvar = 0.0
        for v in ks { kvar += (v - km) * (v - km) }
        for v in resid { rvar += (v - rm) * (v - rm) }
        kvar = (kvar / Double(ks.count)).squareRoot()
        rvar = (rvar / Double(resid.count - 1)).squareRoot()
        guard kvar > 0 else { return nil }
        let seSlope = rvar / (Double(ks.count).squareRoot() * kvar)
        return 86_400 * seSlope / nominalCycle
    }
}

// MARK: - Timegrapher

/// Acoustic timegrapher.
///
/// The rate comes from timing *individual beats* against the phone's audio clock,
/// the way a real timegrapher does — not from the phase of the beat-frequency
/// line. A lock-in phase slope needs a minute or more before it stops wandering
/// by tens of s/day, which is why an early reading used to collapse toward the
/// truth rather than converge on it; timing beats gives a reading good to about
/// ±1 s/day within fifteen seconds.
///
/// The published ± is the scatter of independent sub-window rates, floored by a
/// calibrated multiple of the fit's own standard error. The fit SE alone is
/// 10–20× too optimistic because beat timings are correlated: both the contact
/// and the watch wander.
final class Timegrapher {

    static let standardRates: [(bph: Int, bps: Double)] = [
        (18000, 5.0), (19800, 5.5), (21600, 6.0),
        (25200, 7.0), (28800, 8.0), (36000, 10.0),
    ]

    static let standardBPH = standardRates.map { $0.bph }

    /// Candidate listening bands. Escapement energy lands anywhere from ~4 kHz
    /// to ~20 kHz depending on the movement and how the case couples to the
    /// phone, so we pick rather than assume. Nothing below 4 kHz: down there the
    /// tick is buried in handling and room rumble and every beat times badly.
    static let candidateBands: [(lo: Double, hi: Double)] = [
        (4_000, 9_000), (5_000, 11_000), (7_000, 13_000), (9_000, 15_000),
        (11_000, 18_000), (13_000, 21_000), (15_000, 23_000), (6_000, 20_000),
    ]

    /// Envelope rate for beat tracking. 4 kHz = 0.25 ms per sample; sub-sample
    /// interpolation of the correlation peak takes timing well below that.
    static let envRate = 4_000.0

    /// The ± at which we call the reading done. A mechanical watch is specified
    /// in whole s/day, so resolving past ±2 tells the owner nothing more.
    static let targetPrecision = 2.0

    /// Is the tracking good enough to trust, by the same rule the band probe
    /// uses? Jitter leads: a tick timed to a few tens of microseconds is a real
    /// escapement even if some beats were missed, whereas noise mistimes by
    /// milliseconds. Gating on detection alone called a 0.019 ms lock unstable.
    static func trackingIsHealthy(_ q: (jitterMs: Double, detection: Double, score: Double))
        -> Bool {
        let clean = q.jitterMs < 0.3 && q.detection >= 0.6 && q.score >= 0.5
        let solid = q.jitterMs < 0.5 && q.detection >= 0.85 && q.score >= 0.6
        return clean || solid
    }

    /// Seconds of audio gathered before the band probe first runs.
    private static let probeSeconds = 4.0
    /// How far the probe window may grow while retrying.
    private static let maxProbeSeconds = 16.0
    private static let blockSeconds = 5.0

    // Config
    private var sampleRate = 48_000.0
    private var manualBPH: Int?
    private let maxSeconds = 300.0

    /// Raw mono audio not yet handed to the tracker, guarded by lock. It is
    /// drained as it is consumed — only the band probe ever needs to look back,
    /// so holding a whole session here would cost tens of megabytes for nothing.
    private var raw: [Float] = []
    private var ingested = 0
    private let lock = NSLock()

    private var tracker: BeatTracker?
    private var probed = false
    /// Consecutive analysis passes with unusable tracking, for lock recovery.
    private var badPasses = 0
    /// How many times the band had to be re-chosen mid-measurement.
    private var retuneCount = 0
    private var progressHighWater = 0.0

    // MARK: Setup

    func reset(sampleRate: Double) {
        lock.lock()
        self.sampleRate = sampleRate
        raw.removeAll(keepingCapacity: true)
        raw.reserveCapacity(Int(sampleRate * Timegrapher.probeSeconds * 2))
        ingested = 0
        lock.unlock()

        tracker = nil
        probed = false
        badPasses = 0
        retuneCount = 0
        progressHighWater = 0
    }

    func setManualBPH(_ newValue: Int?) {
        // A different beat rate invalidates everything downstream of the probe,
        // and the audio it ran on has already been drained — so start over.
        lock.lock()
        manualBPH = newValue
        raw.removeAll(keepingCapacity: true)
        ingested = 0
        lock.unlock()

        tracker = nil
        probed = false
        badPasses = 0
        progressHighWater = 0
    }

    // MARK: Ingest (audio thread) — just store raw; filtering is deferred.

    func process(_ samples: UnsafePointer<Float>, count: Int) {
        lock.lock()
        if Double(ingested) < sampleRate * maxSeconds {
            raw.append(contentsOf: UnsafeBufferPointer(start: samples, count: count))
            ingested += count
        }
        lock.unlock()
    }

    // MARK: Analyze (background queue)

    func analyze() -> TimegrapherResult {
        lock.lock()
        let fs = sampleRate
        let manual = manualBPH
        let elapsed = Double(ingested) / fs
        let probeCount = Int(Timegrapher.probeSeconds * fs)
        var probeSrc: [Float] = []
        var tail: [Float] = []
        if !probed {
            // Nothing is drained until the probe has succeeded: if it fails we
            // want to try again over a *longer* window rather than a fresh short
            // one. A quiet recording that can't be identified in 4 s often can
            // be in 8, and throwing the audio away made "Ready" drag for no
            // reason. Capped so the probe can't grow without bound.
            if raw.count >= probeCount {
                probeSrc = Array(raw.prefix(Int(Timegrapher.maxProbeSeconds * fs)))
            }
        } else {
            tail = raw
            raw.removeAll(keepingCapacity: true)
        }
        lock.unlock()

        guard elapsed >= 1.5 else { return idle(.listening, elapsed) }

        // 1. Choose the band and beat rate, once, by trying the best candidates
        //    for real rather than trusting the spectrum alone.
        if !probed {
            guard !probeSrc.isEmpty else {
                // Still refilling after a failed probe — say so once it has been
                // long enough that the user deserves to know nothing is landing.
                return idle(elapsed > 14 ? .noSignal : .listening, elapsed)
            }
            tracker = probeBands(probeSrc, fs: fs, manual: manual)
            guard tracker != nil else {
                // Nothing to lock onto yet. The audio stays buffered so the next
                // attempt sees a longer window; once it stops growing there is
                // genuinely nothing there to find.
                if Double(raw.count) >= Timegrapher.maxProbeSeconds * fs {
                    lock.lock()
                    raw.removeFirst(raw.count / 2)
                    lock.unlock()
                }
                return idle(elapsed > 14 ? .noSignal : .tuning, elapsed)
            }
            lock.lock()
            raw.removeFirst(min(probeSrc.count, raw.count))
            lock.unlock()
            probed = true
            return idle(.tuning, elapsed)
        }

        guard let tracker else {
            return idle(elapsed > 14 ? .noSignal : .tuning, elapsed)
        }
        if !tail.isEmpty {
            tracker.extend(tail)
        }
        guard tracker.step() else {
            return idle(elapsed > 14 ? .noSignal : .tuning, elapsed)
        }

        // If the lock stays bad — the watch slipped, or the probe caught a
        // false beat before the watch was even in place — throw it away and
        // re-probe on fresh audio rather than grinding on unusable beats.
        // Both must be bad: a false lock on room noise mistimes *and* loses
        // beats. A merely noisy watch still detects nearly every beat, and
        // restarting would throw away a usable — if coarse — measurement.
        let q = tracker.quality()
        if q.jitterMs > 1.0 && q.detection < 0.6 {
            badPasses += 1
        } else {
            badPasses = 0
        }
        if badPasses >= 16 {          // ~8 s at the 0.5 s analysis cadence
            self.tracker = nil
            probed = false
            badPasses = 0
            retuneCount += 1
            return idle(.tuning, elapsed)
        }

        return buildResult(tracker, elapsed: elapsed)
    }

    // MARK: Band + beat-rate selection

    /// Choose the listening band *and* the beat rate by trying candidates for
    /// real rather than trusting the spectrum.
    ///
    /// Two things the spectrum alone gets wrong. It picks the *loudest* band,
    /// which isn't always the sharpest — a ringing band can carry a strong beat
    /// line while smearing every tick. And under impulsive noise (a knock, the
    /// watch shifting) it picks the wrong *rate*: injected-noise tests had it
    /// choosing 21600 for a 28800 movement, after which the tracker hunts for
    /// beats that were never there and the measurement is simply lost.
    ///
    /// So: shortlist (band, rate) pairs on a harmonic-summed spectral score,
    /// then track each one. Timing jitter separates a real escapement from noise
    /// by a factor of a hundred — 0.05 ms against 8 ms — so it decides.
    private func probeBands(_ src: [Float], fs: Double, manual: Int?) -> BeatTracker? {
        let scanCount = src.count
        let candidates = manual.map { [$0] } ?? Timegrapher.standardBPH

        var pairs: [(bandProm: Double, prom: Double, lo: Double, hi: Double, bph: Int)] = []
        for band in Timegrapher.candidateBands {
            let (e, er) = coarseEnvelope(src, count: scanCount, fs: fs,
                                         lo: band.lo, hi: band.hi)
            guard e.count > Int(er) else { continue }
            var scored: [(prom: Double, bph: Int)] = []
            for candidate in candidates {
                scored.append((harmonicProminence(e, fs: er, bph: candidate), candidate))
            }
            scored.sort { $0.prom > $1.prom }
            guard let bandProm = scored.first?.prom else { continue }
            // Keep each band's two best rates: the runner-up is what saves the
            // measurement when noise flatters the wrong one.
            for entry in scored.prefix(2) {
                pairs.append((bandProm, entry.prom, band.lo, band.hi, entry.bph))
            }
        }
        guard !pairs.isEmpty else { return nil }
        pairs.sort { $0.bandProm != $1.bandProm ? $0.bandProm > $1.bandProm : $0.prom > $1.prom }

        // Candidates from the three most promising bands.
        var seenBands: Set<String> = []
        var shortlist: [(lo: Double, hi: Double, bph: Int)] = []
        for pair in pairs {
            let key = "\(pair.lo)-\(pair.hi)"
            if seenBands.count >= 3, !seenBands.contains(key) { continue }
            seenBands.insert(key)
            shortlist.append((pair.lo, pair.hi, pair.bph))
        }

        let probe = Array(src[0..<scanCount])
        var winner: BeatTracker?
        var winnerKey = (Double.infinity, 0.0)
        for entry in shortlist {
            let t = BeatTracker(lo: entry.lo, hi: entry.hi, bph: entry.bph,
                                sampleRate: fs, targetEnvRate: Timegrapher.envRate)
            t.extend(probe)
            guard t.step(), t.acceptedCount >= 12 else { continue }
            let q = t.quality()
            // A real escapement times to a fraction of a millisecond. Anything
            // looser is room noise dressed up as a beat.
            // Jitter is the real discriminator — a genuine escapement times to
            // a few tens of microseconds, noise to milliseconds — so a very
            // clean tick is accepted even if some beats were missed. Requiring
            // 85% detection alone rejected a band with 0.013 ms jitter.
            let clean = q.jitterMs < 0.3 && q.detection >= 0.6 && q.score >= 0.5
            let solid = q.jitterMs < 1.0 && q.detection >= 0.85 && q.score >= 0.6
            guard clean || solid else { continue }
            let key = (q.jitterMs, -q.score)
            if key < winnerKey { winnerKey = key; winner = t }
        }
        // No fallback to "the strongest spectral line": on a few seconds of room
        // noise that line is noise, and locking onto it is worse than waiting.
        // Returning nil leaves the probe armed to try the next few seconds —
        // which is what the user needs while they're still positioning the watch.
        return winner
    }

    /// A cheap 1 kHz energy envelope, only used to identify band and beat rate.
    private func coarseEnvelope(_ src: [Float], count: Int, fs: Double,
                                lo: Double, hi: Double) -> ([Double], Double) {
        var hp = Biquad.highpass(fc: lo, fs: fs)
        var lp = Biquad.lowpass(fc: min(hi, fs / 2 - 500), fs: fs)
        let decim = max(1, Int((fs / 1_000.0).rounded()))
        let er = fs / Double(decim)
        var e = [Double]()
        e.reserveCapacity(count / decim + 1)
        var acc = 0.0, c = 0
        for i in 0..<count {
            let y = lp.process(hp.process(Double(src[i])))
            acc += y * y; c += 1
            if c >= decim { e.append(acc / Double(decim)); acc = 0; c = 0 }
        }
        var m = 0.0
        for v in e { m += v }
        if !e.isEmpty {
            m /= Double(e.count)
            for i in 0..<e.count { e[i] -= m }
        }
        return (e, er)
    }

    /// Beat-line strength for a candidate rate, summed over harmonics.
    ///
    /// A tick train is impulsive, so its envelope carries real energy at 2f0 and
    /// 3f0 as well. Noise can flatter a wrong candidate's fundamental by luck;
    /// matching the whole comb is much harder to do by accident.
    private func harmonicProminence(_ e: [Double], fs: Double, bph: Int) -> Double {
        let f0 = Double(bph) / 3600
        var signal = 0.0
        for k in [1.0, 2.0, 3.0] { signal += dftMag(e, fs: fs, f: f0 * k) }
        var noise = 0.0
        let offsets = [0.6, 0.75, 1.2, 1.35, 1.6, 2.4, 2.7]
        for k in offsets { noise += dftMag(e, fs: fs, f: f0 * k) }
        noise /= Double(offsets.count)
        return signal / (3 * noise + 1e-12)
    }

    /// Magnitude of the envelope's DFT bin at frequency f.
    private func dftMag(_ e: [Double], fs: Double, f: Double) -> Double {
        let n = e.count
        let w = 2 * Double.pi * f / fs
        var re = 0.0, im = 0.0
        var cr = 1.0, ci = 0.0
        let dc = cos(w), ds = sin(w)
        for i in 0..<n {
            re += e[i] * cr
            im -= e[i] * ci
            let ncr = cr * dc - ci * ds
            ci = cr * ds + ci * dc
            cr = ncr
            if i & 8191 == 0 {
                let m = (cr * cr + ci * ci).squareRoot()
                if m > 0 { cr /= m; ci /= m }
            }
        }
        return (re * re + im * im).squareRoot() / Double(n)
    }

    // MARK: Result assembly

    private func idle(_ stage: MeasurementStage, _ elapsed: Double) -> TimegrapherResult {
        let p = min(0.25, elapsed / Timegrapher.probeSeconds * 0.25)
        progressHighWater = max(progressHighWater, p)
        return TimegrapherResult(
            stage: stage, beatsPerHour: tracker?.bph ?? 0, rateSecondsPerDay: nil,
            uncertainty: nil, beatErrorMs: nil, jitterMs: 0, detectionRate: 0,
            matchScore: 0, beatsTracked: 0, elapsedSeconds: elapsed,
            bandLowHz: tracker?.lo ?? 0, bandHighHz: tracker?.hi ?? 0,
            progress: progressHighWater, secondsRemaining: nil, rateSamples: [],
            qualitySamples: [], retuneCount: retuneCount)
    }

    private func buildResult(_ t: BeatTracker, elapsed: Double) -> TimegrapherResult {
        let q = t.quality()
        var rate = t.rateSecondsPerDay

        // Published ±: the scatter of independent sub-window rates, measured at
        // two block lengths so an unlucky chopping of the record can't make a
        // drifting reading look settled, floored by 10× the fit's own SE (the
        // factor that makes it honest, calibrated against recorded sessions).
        var unc: Double?
        if rate != nil {
            // SEM at several block lengths. Short blocks put a number on screen
            // sooner, but they overstate the ± for good — each short block's own
            // slope is noisy — so they're only used until longer blocks exist.
            var sems: [Double] = []
            for (tau, step) in [(Timegrapher.blockSeconds / 2, Timegrapher.blockSeconds / 4),
                                (Timegrapher.blockSeconds, Timegrapher.blockSeconds / 2),
                                (Timegrapher.blockSeconds * 2, Timegrapher.blockSeconds)] {
                let blocks = t.blockRates(tau: tau, step: step)
                guard blocks.count >= 2 else { continue }
                var m = 0.0
                for v in blocks { m += v }
                m /= Double(blocks.count)
                var s = 0.0
                for v in blocks { s += (v - m) * (v - m) }
                // Overlapping blocks are correlated; count only the independent ones.
                let independent = max(2.0, Double(blocks.count) * step / tau)
                sems.append((s / Double(blocks.count - 1)).squareRoot() / independent.squareRoot())
            }
            // Worst case over the two longest block lengths we can form.
            let sd = sems.isEmpty ? nil : 2 * sems.suffix(2).max()!
            if let sd {
                var value = sd
                if let naive = t.naiveSE() { value = max(value, 10 * naive) }
                // The phone's own crystal is only good to about half a second a day.
                unc = max(0.5, value)
            }
        }

        // A wildly wide ± means we aren't really locked; show nothing rather
        // than a number with a meaningless bound attached to it.
        if let u = unc, u > 25 { rate = nil; unc = nil }

        // Stage.
        var stage: MeasurementStage
        if rate == nil || unc == nil {
            stage = .locking
        } else if !Timegrapher.trackingIsHealthy(q) {
            stage = .unstable
        } else if unc! <= Timegrapher.targetPrecision {
            stage = .done
        } else {
            stage = .measuring
        }
        if elapsed > 20, t.acceptedCount < 10 { stage = .noSignal }

        // Progress and time remaining. The ± falls as 1/√time, so what's left is
        // (current/target)² − 1 of the time spent so far.
        var remaining: Double?
        var progress: Double
        if let u = unc {
            let ratio = Timegrapher.targetPrecision / u
            progress = 0.25 + 0.75 * min(1.0, ratio * ratio)
            if u > Timegrapher.targetPrecision {
                let need = (u * u) / (Timegrapher.targetPrecision * Timegrapher.targetPrecision)
                let estimate = elapsed * (need - 1)
                // Only worth showing when it's a wait rather than a verdict.
                remaining = estimate <= 180 ? max(1, estimate) : nil
            }
        } else {
            progress = min(0.25, elapsed / Timegrapher.probeSeconds * 0.25)
        }
        progressHighWater = max(progressHighWater, progress)

        // Independent sub-window measurements, non-overlapping so each really is
        // a separate piece of evidence rather than a smoothed version of its
        // neighbour.
        let samples = t.blockSamples(tau: Timegrapher.blockSeconds)
        let quality = t.qualitySamples(window: 2.0)

        return TimegrapherResult(
            stage: stage,
            beatsPerHour: t.bph,
            rateSecondsPerDay: rate,
            uncertainty: unc,
            beatErrorMs: t.beatErrorMs,
            jitterMs: q.jitterMs,
            detectionRate: q.detection,
            matchScore: q.score,
            beatsTracked: t.acceptedCount,
            elapsedSeconds: elapsed,
            bandLowHz: t.lo,
            bandHighHz: t.hi,
            progress: progressHighWater,
            secondsRemaining: remaining,
            rateSamples: samples,
            qualitySamples: quality,
            retuneCount: retuneCount)
    }
}
