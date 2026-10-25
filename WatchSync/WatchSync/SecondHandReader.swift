import Foundation

/// How far the watch is from the phone's clock, right now.
struct OffsetReading {
    enum Stage { case searching, measuring, done }
    /// A hand that glides (mechanical) or one that jumps once a second (quartz).
    enum HandMotion { case sweep, tick }

    var stage: Stage = .searching
    /// Watch minus phone, in seconds, within (−30, +30]. Positive = watch ahead.
    /// Only the second hand is read, so a whole minute of error is invisible.
    var offsetSeconds: Double?
    var uncertainty: Double?
    var motion: HandMotion = .sweep
    /// How far the second hand stands out of the stack, in noise widths.
    var snr: Double = 0
    /// 0…1 toward a finished reading.
    var progress: Double = 0
    /// False when the inner ring couldn't be read, so an off-centre dial is
    /// uncorrected and the ± is widened to say so.
    var centerCorrected = false
    /// Degrees the dial is turned clockwise from upright in the guide, read off
    /// its hour markers and already taken out of the offset. Nil when the dial
    /// has no ring of markers to read, and the offset trusts the alignment.
    var dialTwist: Double?
    /// Capture time of the newest frame behind this reading (Unix seconds).
    var measuredAt: Double = 0
}

/// Reads the second hand's position against the clock from timestamped frames.
///
/// The question is "what does the watch say at phone time T?", and one picture
/// can't answer it reliably: on a still, the second hand is one thin line among
/// hour, minute and GMT hands, and every single-frame reader we tried latched
/// onto the wrong one. So this takes a few seconds of pictures, each with its
/// own capture timestamp, and uses the one property nothing else on the dial
/// has — it is the only thing that moves.
///
///   1. Sample each frame into an angular profile around the guide centre.
///   2. Subtract a per-angle temporal median. Everything static vanishes; the
///      hand, which sits at any one angle only briefly, survives. (A median
///      rather than a moving average: a quartz hand parks for a full second,
///      and an average leaves ghosts either side of it that outscore the hand.)
///   3. Rotate every row back by where the *phone's* second hand would be at
///      that frame's timestamp, and sum. A hand keeping any fixed offset from
///      the phone piles up in one column; that column is the offset. No rate
///      search — over ten seconds even a terrible watch is at 6°/s.
///   4. Read the peak per frame at two radii. A line through the pivot crosses
///      two rings at slightly different angles when the guide isn't centred on
///      the pivot, so the difference between the rings measures the miss and
///      takes it out — otherwise 5% off-centre is most of a second.
///   5. Read the dial's own twist off its hour markers. The hand's angle is
///      measured against the picture, and the picture is only as upright as
///      the phone was held: four degrees of twist — hard to see, easy to do —
///      is two thirds of a second. The markers are static, so they survive
///      where the hand was removed, and twelve of them say where 12 really is.
///
/// Not thread-safe: feed and analyse it from one queue.
final class SecondHandReader {
    /// Guide circle radius as a fraction of half the frame's short side. The
    /// overlay draws the same circle, which is the only alignment there is.
    static let guideRadiusFraction = 0.86
    /// Seconds of locked data that make a finished reading.
    static let targetSpan = 8.0

    private static let bins = 720
    private static let binsPerSecond = Double(bins) / 60
    private static let radialSamples = 24
    /// Seconds of frames kept. Longer buys nothing: the answer is a phase, not
    /// a slope, so it doesn't sharpen with baseline the way a rate does.
    private static let window = 10.0
    private static let minSpan = 3.0
    private static let lockSNR = 8.0
    /// Dial alignment by eye is good to about a degree, a sixth of a second,
    /// and no amount of averaging improves on it.
    private static let sweepFloor = 0.15
    /// With the twist read off the markers, what's left is how well a marker's
    /// middle can be found.
    private static let sweepFloorAligned = 0.08
    /// Where hour markers might be, as fractions of the guide radius. Three
    /// overlapping bands, because how well the dial fills the circle varies
    /// and so does where a dial keeps its markers.
    private static let markerBands = [(0.60, 0.74), (0.72, 0.86), (0.84, 0.98)]
    /// The markers don't move, so they are sampled from every sixth frame.
    private static let markerEvery = 6
    /// Twist beyond this isn't a held-by-hand error, it's a different lock.
    private static let maxTwist = 10.0
    /// A tick is timed between two frames, so the frame period is the floor.
    private static let tickFloor = 0.05

    private struct Ring {
        var offsets: [Int32] = []
        var wx: [Float] = []
        var wy: [Float] = []
        /// Mean of 1/r across the band: how strongly a centre miss shows up.
        var leverage = 0.0
    }

    private var outerRing = Ring()
    private var innerRing = Ring()
    private var markerRings: [Ring] = []
    private var lutKey = (width: 0, height: 0, bytesPerRow: 0)

    private var times: [Double] = []
    private var outer: [Float] = []
    private var inner: [Float] = []
    private var markers = [[Float]](repeating: [], count: SecondHandReader.markerBands.count)
    private var framesSeen = 0

    func reset() {
        times.removeAll(keepingCapacity: true)
        outer.removeAll(keepingCapacity: true)
        inner.removeAll(keepingCapacity: true)
        for b in markers.indices { markers[b].removeAll(keepingCapacity: true) }
        framesSeen = 0
    }

    // MARK: Sampling

    /// Add one frame. `luma` is an upright 8-bit plane with the guide circle at
    /// its centre; `time` is when it was captured, in Unix seconds.
    func process(luma: UnsafePointer<UInt8>, width: Int, height: Int,
                 bytesPerRow: Int, time: Double) {
        guard width > 32, height > 32 else { return }
        if let last = times.last, time <= last { return }
        if lutKey != (width, height, bytesPerRow) {
            lutKey = (width, height, bytesPerRow)
            let radius = Self.guideRadiusFraction * Double(min(width, height)) / 2
            // Outer: where only the second hand reaches. Inner: clear of the
            // hand's counterweight, which would otherwise read as a hand
            // pointing the other way.
            outerRing = makeRing(from: 0.55, to: 0.80, radius: radius)
            innerRing = makeRing(from: 0.28, to: 0.45, radius: radius)
            markerRings = Self.markerBands.map { makeRing(from: $0.0, to: $0.1, radius: radius) }
            reset()
        }
        times.append(time)
        sample(luma, ring: outerRing, into: &outer)
        sample(luma, ring: innerRing, into: &inner)
        if framesSeen % Self.markerEvery == 0 {
            for b in markers.indices {
                sample(luma, ring: markerRings[b], into: &markers[b])
                if markers[b].count > 60 * Self.bins { markers[b].removeFirst(20 * Self.bins) }
            }
        }
        framesSeen += 1

        // Trim in batches so the arrays aren't shuffled down on every frame.
        if time - times[0] > Self.window + 2 {
            let cut = times.firstIndex { $0 >= time - Self.window } ?? 0
            times.removeFirst(cut)
            outer.removeFirst(cut * Self.bins)
            inner.removeFirst(cut * Self.bins)
        }
    }

    private func makeRing(from r0: Double, to r1: Double, radius: Double) -> Ring {
        let (width, height, bytesPerRow) = lutKey
        let cx = Double(width) / 2, cy = Double(height) / 2
        var ring = Ring()
        let count = Self.bins * Self.radialSamples
        ring.offsets.reserveCapacity(count)
        ring.wx.reserveCapacity(count)
        ring.wy.reserveCapacity(count)
        var leverage = 0.0
        for j in 0..<Self.bins {
            // Clockwise from 12, as a watch is read.
            let a = Double(j) * 2 * .pi / Double(Self.bins)
            for k in 0..<Self.radialSamples {
                let f = r0 + (r1 - r0) * (Double(k) + 0.5) / Double(Self.radialSamples)
                let x = min(max(cx + radius * f * sin(a), 0), Double(width) - 1.001)
                let y = min(max(cy - radius * f * cos(a), 0), Double(height) - 1.001)
                let ix = Int(x), iy = Int(y)
                ring.offsets.append(Int32(iy * bytesPerRow + ix))
                ring.wx.append(Float(x - Double(ix)))
                ring.wy.append(Float(y - Double(iy)))
                if j == 0 { leverage += 1 / f }
            }
        }
        ring.leverage = leverage / Double(Self.radialSamples)
        return ring
    }

    private func sample(_ luma: UnsafePointer<UInt8>, ring: Ring, into store: inout [Float]) {
        let bytesPerRow = lutKey.bytesPerRow
        let start = store.count
        store.reserveCapacity(start + Self.bins)
        var total: Float = 0
        ring.offsets.withUnsafeBufferPointer { offsets in
            ring.wx.withUnsafeBufferPointer { wx in
                ring.wy.withUnsafeBufferPointer { wy in
                    var idx = 0
                    for _ in 0..<Self.bins {
                        var acc: Float = 0
                        for _ in 0..<Self.radialSamples {
                            let o = Int(offsets[idx])
                            let p00 = Float(luma[o]), p01 = Float(luma[o + 1])
                            let p10 = Float(luma[o + bytesPerRow])
                            let p11 = Float(luma[o + bytesPerRow + 1])
                            let top = p00 + (p01 - p00) * wx[idx]
                            let bottom = p10 + (p11 - p10) * wx[idx]
                            acc += top + (bottom - top) * wy[idx]
                            idx += 1
                        }
                        store.append(acc)
                        total += acc
                    }
                }
            }
        }
        // Normalise to a mean of 100, so auto-exposure stepping or mains
        // flicker doesn't make every static feature look like it changed.
        let gain = 100 * Float(Self.bins) / max(total, 1)
        for i in start..<store.count { store[i] *= gain }
    }

    // MARK: Analysis

    func analyze() -> OffsetReading {
        var reading = OffsetReading()
        let n = times.count
        guard n >= 30, let newest = times.last, newest - times[0] >= Self.minSpan else {
            return reading
        }
        reading.measuredAt = newest
        let bins = Self.bins

        // Where the phone's own second hand points at each frame, in bins.
        let phone = times.map { $0.truncatingRemainder(dividingBy: 60) * Self.binsPerSecond }

        let outerMoving = movingPart(of: outer)
        let innerMoving = movingPart(of: inner)

        // The stack: every row rotated into the phone's frame and summed.
        var stack = [Float](repeating: 0, count: bins)
        outerMoving.withUnsafeBufferPointer { rows in
            for i in 0..<n {
                let whole = Int(phone[i])
                let f = Float(phone[i] - Double(whole))
                let row = i * bins
                for c in 0..<bins {
                    let a = (c + whole) % bins
                    let b = a + 1 == bins ? 0 : a + 1
                    stack[c] += rows[row + a] * (1 - f) + rows[row + b] * f
                }
            }
        }
        var smooth = [Float](repeating: 0, count: bins)
        for c in 0..<bins {
            var s: Float = 0
            for k in -2...2 { s += stack[(c + k + bins) % bins] }
            smooth[c] = s / 5
        }
        var peak = 0
        for c in 1..<bins where smooth[c] > smooth[peak] { peak = c }
        // Judged against its own neighbourhood, not the whole stack. A shaken
        // frame leaves residue on every static edge, and over a ten-second
        // capture each one smears into a stripe 60° wide — a broad hump that
        // towers over the global median with no hand in it at all. A hand is a
        // spike on whatever it sits on. The neighbourhood starts 2 s out so a
        // ticking hand's one-second plateau isn't counted as its own floor.
        var around: [Float] = []
        for k in 24...90 {
            around.append(smooth[(peak + k) % bins])
            around.append(smooth[(peak - k + bins) % bins])
        }
        around.sort()
        let floor = around[around.count / 2]
        let mad = around.map { abs($0 - floor) }.sorted()[around.count / 2]
        reading.snr = Double((smooth[peak] - floor) / (1.4826 * mad + 1e-6))
        guard reading.snr >= Self.lockSNR else { return reading }

        // Per frame: where the hand is relative to the stack's peak, in each
        // ring. ±1.5 s around the peak holds every position a ticking hand
        // takes, wherever in its one-second plateau the peak happened to land.
        var outerPos = [Double](repeating: 0, count: n)
        var outerConf = [Double](repeating: 0, count: n)
        var innerPos = [Double](repeating: 0, count: n)
        var innerConf = [Double](repeating: 0, count: n)
        for i in 0..<n {
            (outerPos[i], outerConf[i]) = locate(in: outerMoving, row: i, phone: phone[i],
                                                 center: Double(peak), halfWidth: 18)
            // The inner ring is only searched beside the outer answer: a centre
            // miss moves it a few degrees, nothing else should move it at all.
            (innerPos[i], innerConf[i]) = locate(in: innerMoving, row: i, phone: phone[i],
                                                 center: Double(peak) + outerPos[i].rounded(),
                                                 halfWidth: 16)
            innerPos[i] += outerPos[i].rounded()
        }
        let outerGate = max(3.5, 0.4 * percentile(outerConf, 0.75))
        let innerGate = max(3.5, 0.4 * percentile(innerConf, 0.75))
        let seen = (0..<n).filter { outerConf[$0] >= outerGate }
        guard seen.count >= 20 else { return reading }
        let both = seen.filter { innerConf[$0] >= innerGate }

        // Centre correction. The rings disagree by the miss times the
        // difference in their leverage; scale that up to the outer ring's own
        // share and remove it. Smoothed over neighbours first, because the
        // correction amplifies noise and the miss changes only slowly.
        let gain = outerRing.leverage / (innerRing.leverage - outerRing.leverage)
        var frames: [Int] = []
        var position: [Double] = []
        if both.count >= max(20, seen.count / 2) {
            for i in both {
                let near = both.filter { abs(times[$0] - times[i]) <= 0.75 }
                guard near.count >= 5 else { continue }
                let miss = percentile(near.map { outerPos[$0] - innerPos[$0] }, 0.5)
                frames.append(i)
                position.append(outerPos[i] + gain * miss)
            }
        }
        if frames.count >= 20 {
            reading.centerCorrected = true
        } else {
            frames = seen
            position = seen.map { outerPos[$0] }
        }

        // A dial turned clockwise carries the hand's angle forward with it.
        reading.dialTwist = dialTwist()
        let twist = (reading.dialTwist ?? 0) / 6

        let seconds = position.map { $0 / Self.binsPerSecond }
        let base = Double(peak) / Self.binsPerSecond - twist
        var offset: Double
        var uncertainty: Double
        if let phase = tickPhase(frames: frames, seconds: seconds) {
            // The hand shows floor(T + offset): between ticks it falls behind a
            // sweeping phone hand, so its apparent offset is a sawtooth. The
            // *instant* of the jump gives the fraction of a second — from
            // timestamps alone, immune to how the dial is aligned — and the
            // angle only has to choose the whole second.
            let level = percentile(frames.indices.map {
                seconds[$0] + frac(times[frames[$0]] + phase)
            }, 0.5)
            offset = (base + level - phase).rounded() + phase
            uncertainty = Self.tickFloor
            reading.motion = .tick
        } else {
            let trimmed = seconds.sorted()
            let cut = trimmed.count / 10
            let core = trimmed[cut..<(trimmed.count - cut)]
            offset = base + core.reduce(0, +) / Double(core.count)
            // Scatter between independent stretches of the capture, not the
            // scatter of frames: neighbouring frames share their errors.
            let chunks = 4
            var means: [Double] = []
            for k in 0..<chunks {
                let lo = k * seconds.count / chunks, hi = (k + 1) * seconds.count / chunks
                guard hi > lo else { continue }
                means.append(seconds[lo..<hi].reduce(0, +) / Double(hi - lo))
            }
            let mean = means.reduce(0, +) / Double(means.count)
            let variance = means.map { ($0 - mean) * ($0 - mean) }.reduce(0, +)
                / Double(max(1, means.count - 1))
            let scatter = (variance / Double(means.count)).squareRoot()
            let floor = reading.dialTwist == nil ? Self.sweepFloor : Self.sweepFloorAligned
            uncertainty = (scatter * scatter + floor * floor).squareRoot()
            reading.motion = .sweep
        }
        if !reading.centerCorrected {
            uncertainty = (uncertainty * uncertainty + 0.3 * 0.3).squareRoot()
        }

        offset = offset.truncatingRemainder(dividingBy: 60)
        if offset > 30 { offset -= 60 }
        if offset <= -30 { offset += 60 }
        reading.offsetSeconds = offset
        reading.uncertainty = uncertainty

        let span = times[frames[frames.count - 1]] - times[frames[0]]
        reading.progress = min(1, span / Self.targetSpan)
        reading.stage = span >= Self.targetSpan ? .done : .measuring
        return reading
    }

    /// |profile − temporal median| per angle, each row then centred and scaled
    /// by its own level so one shaken frame can't outvote twenty steady ones.
    private func movingPart(of ring: [Float]) -> [Float] {
        let n = times.count, bins = Self.bins
        var out = [Float](repeating: 0, count: n * bins)
        // Median over ±2.5 s: short enough to follow a drifting hand-held
        // frame, long enough that a ticking hand — parked a full second at each
        // mark — is still in the minority at the ends of the buffer.
        let reach = 2.5
        let picks = 31
        var values = [Float](repeating: 0, count: picks)
        var lo = 0, hi = 0
        var b0 = 0
        ring.withUnsafeBufferPointer { src in
            while b0 < n {
                // The background barely changes in a third of a second, so one
                // median serves a block of frames.
                var b1 = b0 + 1
                while b1 < n, times[b1] - times[b0] < 0.3 { b1 += 1 }
                let mid = (times[b0] + times[b1 - 1]) / 2
                while times[lo] < mid - reach { lo += 1 }
                while hi + 1 < n, times[hi + 1] <= mid + reach { hi += 1 }
                let available = hi - lo + 1
                let count = min(picks, available)
                for a in 0..<bins {
                    for q in 0..<count {
                        let i = count == 1 ? lo : lo + q * (available - 1) / (count - 1)
                        // Insertion sort: thirty values, mostly in order already.
                        let v = src[i * bins + a]
                        var p = q
                        while p > 0, values[p - 1] > v {
                            values[p] = values[p - 1]
                            p -= 1
                        }
                        values[p] = v
                    }
                    let median = values[count / 2]
                    for i in b0..<b1 { out[i * bins + a] = abs(src[i * bins + a] - median) }
                }
                b0 = b1
            }
        }
        for i in 0..<n {
            let row = i * bins
            var mean: Float = 0
            for a in 0..<bins { mean += out[row + a] }
            mean /= Float(bins)
            let scale = 1 / (mean + 0.5)
            for a in 0..<bins { out[row + a] = (out[row + a] - mean) * scale }
        }
        return out
    }

    // MARK: Dial twist

    /// How far the dial is turned clockwise from upright, in degrees, from the
    /// band of markers that reads most cleanly. Nil when none does.
    private func dialTwist() -> Double? {
        var best: (twist: Double, count: Int, rms: Double)?
        for band in markers {
            guard let fit = markerFit(band) else { continue }
            if let b = best, (b.count, -b.rms) >= (fit.count, -fit.rms) { continue }
            best = fit
        }
        return best?.twist
    }

    /// Finds twelve evenly spaced markers in one band and fits where they sit.
    ///
    /// Evenly spaced is the test: fold the ring onto one 30° sector and markers
    /// pile up while hands, date windows and lettering don't. Then each marker
    /// is located on its own and the twelve positions fitted with a constant —
    /// the twist — plus a once-around sinusoid, which is what a guide that
    /// isn't centred on the dial does to them and would otherwise leak in.
    private func markerFit(_ band: [Float]) -> (twist: Double, count: Int, rms: Double)? {
        let bins = Self.bins
        let rows = band.count / bins
        guard rows >= 8 else { return nil }
        let sector = bins / 12

        // The static dial: per-angle median over time, which is also what
        // takes the second hand out.
        var profile = [Float](repeating: 0, count: bins)
        var column = [Float](repeating: 0, count: rows)
        for a in 0..<bins {
            for r in 0..<rows { column[r] = band[r * bins + a] }
            column.sort()
            profile[a] = column[rows / 2]
        }
        // Markers as departures from the dial around them, whichever way they
        // depart: bright on dark and dark on bright read the same.
        var raw = [Float](repeating: 0, count: bins)
        var window = [Float](repeating: 0, count: sector + 1)
        for a in 0..<bins {
            for k in 0...sector { window[k] = profile[(a + k - sector / 2 + bins) % bins] }
            window.sort()
            raw[a] = abs(profile[a] - window[sector / 2])
        }
        var q = [Float](repeating: 0, count: bins)
        for a in 0..<bins {
            q[a] = (raw[(a + bins - 1) % bins] + raw[a] + raw[(a + 1) % bins]) / 3
        }

        // Fold, and blur by a minute's worth so a minute track — sixty equal
        // ticks, five to a sector — goes flat and can't pass for the hours.
        var folded = [Float](repeating: 0, count: sector)
        for a in 0..<bins { folded[a % sector] += q[a] }
        var blurred = [Float](repeating: 0, count: sector)
        for j in 0..<sector {
            for k in -6...6 { blurred[j] += folded[(j + k + sector) % sector] }
        }
        var top = 0
        for j in 1..<sector where blurred[j] > blurred[top] { top = j }
        let coarse = top > sector / 2 ? top - sector : top
        guard abs(Double(coarse)) * 360 / Double(bins) <= Self.maxTwist else { return nil }

        // Each marker on its own, near where the fold says it should be.
        var angles: [Double] = []
        var found: [Double] = []
        var heights: [Float] = []
        for m in 0..<12 {
            let center = coarse + m * sector
            func at(_ k: Int) -> Float { q[((center + k) % bins + bins) % bins] }
            var peak = -8
            for k in -8...8 where at(k) > at(peak) { peak = k }
            var around = (-sector / 2...sector / 2).map(at)
            around.sort()
            // A marker stands clear of its own sector; texture doesn't.
            guard at(peak) > 3 * around[around.count / 2] + 0.5 else { continue }
            // Its middle: the centroid of everything above half its height.
            let half = at(peak) / 2
            var lo = peak, hi = peak
            while lo > peak - 12, at(lo - 1) > half { lo -= 1 }
            while hi < peak + 12, at(hi + 1) > half { hi += 1 }
            var weight = 0.0, moment = 0.0
            for k in lo...hi {
                weight += Double(at(k) - half)
                moment += Double(at(k) - half) * Double(k)
            }
            guard weight > 0 else { continue }
            angles.append(Double(m) * .pi / 6)
            found.append(moment / weight)
            heights.append(at(peak))
        }
        // The odd one out — a date window, a logo — is far taller or shorter
        // than its fellows.
        guard heights.count >= 8 else { return nil }
        let typical = heights.sorted()[heights.count / 2]
        var keep = heights.indices.filter { heights[$0] > 0.35 * typical }

        var fit = (constant: 0.0, sine: 0.0, cosine: 0.0)
        var rms = 0.0
        for pass in 0..<2 {
            guard keep.count >= 8,
                  let solved = fitSinusoid(angles: keep.map { angles[$0] }, values: keep.map { found[$0] })
            else { return nil }
            fit = solved
            let residual = keep.map {
                found[$0] - fit.constant - fit.sine * sin(angles[$0]) - fit.cosine * cos(angles[$0])
            }
            rms = (residual.map { $0 * $0 }.reduce(0, +) / Double(residual.count)).squareRoot()
            if pass == 0 {
                keep = zip(keep, residual).filter { abs($0.1) <= 3 }.map { $0.0 }
            }
        }
        guard rms <= 1.5 else { return nil }
        let twist = (Double(coarse) + fit.constant) * 360 / Double(bins)
        guard abs(twist) <= Self.maxTwist else { return nil }
        return (twist, keep.count, rms)
    }

    /// Least squares for value ≈ c + s·sin(angle) + k·cos(angle).
    private func fitSinusoid(angles: [Double], values: [Double])
        -> (constant: Double, sine: Double, cosine: Double)? {
        var m = [[Double]](repeating: [Double](repeating: 0, count: 4), count: 3)
        for (angle, value) in zip(angles, values) {
            let basis = [1, sin(angle), cos(angle)]
            for i in 0..<3 {
                for j in 0..<3 { m[i][j] += basis[i] * basis[j] }
                m[i][3] += basis[i] * value
            }
        }
        for i in 0..<3 {
            let pivot = (i..<3).max { abs(m[$0][i]) < abs(m[$1][i]) }!
            guard abs(m[pivot][i]) > 1e-6 else { return nil }
            m.swapAt(i, pivot)
            for r in 0..<3 where r != i {
                let f = m[r][i] / m[i][i]
                for c in i..<4 { m[r][c] -= f * m[i][c] }
            }
        }
        return (m[0][3] / m[0][0], m[1][3] / m[1][1], m[2][3] / m[2][2])
    }

    /// The brightest point of one row near `center` (in the phone's rotating
    /// frame), as a sub-bin position relative to `center`, with its height.
    private func locate(in rows: [Float], row: Int, phone: Double,
                        center: Double, halfWidth: Int) -> (Double, Double) {
        let bins = Self.bins
        let origin = Int(center.rounded()) - halfWidth
        let whole = Int(phone)
        let f = Float(phone - Double(whole))
        let base = row * bins
        let count = 2 * halfWidth + 1
        var best = 0
        var bestValue = -Float.infinity
        var values = [Float](repeating: 0, count: count)
        for k in 0..<count {
            let a = ((origin + k + whole) % bins + bins) % bins
            let b = a + 1 == bins ? 0 : a + 1
            values[k] = rows[base + a] * (1 - f) + rows[base + b] * f
            if values[k] > bestValue {
                bestValue = values[k]
                best = k
            }
        }
        var weight = 0.0, moment = 0.0
        for k in max(0, best - 3)...min(count - 1, best + 3) where values[k] > 0 {
            weight += Double(values[k])
            moment += Double(values[k]) * Double(k)
        }
        let position = weight > 0 ? moment / weight : Double(best)
        return (position - Double(halfWidth), Double(bestValue))
    }

    /// If the hand jumps once a second, the fraction of a second at which it
    /// does, as the phase φ with offset ≡ φ (mod 1). Nil for a sweeping hand.
    private func tickPhase(frames: [Int], seconds: [Double]) -> Double? {
        // A sweeping hand holds its offset; a ticking one wanders across a full
        // second of it. Anything under half a second isn't worth testing.
        guard percentile(seconds, 0.9) - percentile(seconds, 0.1) > 0.45 else { return nil }
        func spread(_ values: [Double]) -> Double {
            let m = percentile(values, 0.5)
            return values.map { abs($0 - m) }.reduce(0, +) / Double(values.count)
        }
        let steps = 200
        var costs = [Double](repeating: 0, count: steps)
        for s in 0..<steps {
            let phase = Double(s) / Double(steps)
            costs[s] = spread(frames.indices.map { seconds[$0] + frac(times[frames[$0]] + phase) })
        }
        guard let best = costs.indices.min(by: { costs[$0] < costs[$1] }) else { return nil }
        // Undoing the sawtooth has to flatten it properly, or this is just a
        // noisy sweep.
        guard costs[best] < 0.12, costs[best] * 2.5 < spread(seconds) else { return nil }
        // Frames arrive every 1/30 s, so a range of phases fit equally well:
        // everything between the last frame before the jump and the first
        // after. Take the middle of that range.
        let tolerance = costs[best] * 1.02 + 1e-4
        var first = best, last = best
        while last - first < steps - 1, costs[((first - 1) % steps + steps) % steps] <= tolerance {
            first -= 1
        }
        while last - first < steps - 1, costs[(last + 1) % steps] <= tolerance { last += 1 }
        return frac(Double(first + last) / 2 / Double(steps))
    }

    private func frac(_ x: Double) -> Double { x - x.rounded(.down) }

    private func percentile(_ values: [Double], _ p: Double) -> Double {
        guard !values.isEmpty else { return 0 }
        let sorted = values.sorted()
        return sorted[min(sorted.count - 1, Int(Double(sorted.count) * p))]
    }
}
