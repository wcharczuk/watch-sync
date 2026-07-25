import SwiftUI
import UIKit
import Combine

/// Measurement screen.
///
/// Two questions get asked here, and the old screen answered them with four
/// overlapping widgets that could disagree — a "confidence" bar that fell while
/// the headline said the reading was good. They are separate questions with
/// separate answers, so each gets exactly one place on screen:
///
///   Can I hear the watch?     → the signal row. Actionable: press firmer,
///                               find a quieter room.
///   How sure is the number?   → the ± and the precision bar. Not actionable:
///                               it only wants time, and it only goes forward.
struct AccuracyView: View {
    @StateObject private var viewModel = TimegrapherViewModel()
    @State private var showHelp = false
    @State private var showDetail = false
    @State private var shareItems: [Any]?

    var body: some View {
        ZStack {
            Color(uiColor: .systemBackground).ignoresSafeArea()
            switch viewModel.audio.permission {
            case .denied:
                permissionDeniedView
            case .granted:
                content
            case .unknown:
                ProgressView("Requesting microphone access…")
                    .onAppear { viewModel.audio.requestPermission() }
            }
        }
        .sheet(isPresented: $showHelp) { HelpView() }
    }

    // MARK: Permission

    private var permissionDeniedView: some View {
        VStack(spacing: 16) {
            Spacer()
            Image(systemName: "mic.slash.fill")
                .font(.system(size: 48))
                .foregroundColor(.secondary)
            Text("Microphone Access Required")
                .font(.headline)
            Text("WatchSync listens to your watch ticking. Open Settings to grant microphone permission.")
                .font(.subheadline)
                .foregroundColor(.secondary)
                .multilineTextAlignment(.center)
            Button("Open Settings") {
                if let url = URL(string: UIApplication.openSettingsURLString) {
                    UIApplication.shared.open(url)
                }
            }
            .buttonStyle(.borderedProminent)
            Spacer()
        }
        .padding()
    }

    // MARK: Content

    private var content: some View {
        VStack(spacing: 14) {
            rateReadout
            precisionBar
            warmupFeedback
            // The tape takes whatever room is left: it's the part worth looking
            // at, and letting it stretch keeps the idle screen from being mostly
            // dead space.
            paperTape
                .frame(minHeight: 128, maxHeight: .infinity)
            metricsRow
            if showDetail { detailLine }
            actionButtons
        }
        .padding(.horizontal)
        .padding(.top, 12)
        .padding(.bottom, 12)
        .onAppear { viewModel.audio.requestPermission() }
        .onDisappear { viewModel.stop() }
        .sheet(isPresented: Binding(
            get: { shareItems != nil },
            set: { if !$0 { shareItems = nil } }
        )) {
            if let items = shareItems { ShareSheet(items: items) }
        }
    }

    private var stage: MeasurementStage {
        guard viewModel.isMeasuring, let r = viewModel.result else { return .listening }
        return r.stage
    }

    // MARK: Rate readout

    private var rateReadout: some View {
        VStack(spacing: 6) {
            HStack(spacing: 6) {
                stageIcon
                Text(stageHeadline)
                    .font(.system(size: 14, weight: .semibold))
            }
            .foregroundColor(stageColor)

            if let r = viewModel.result, let rate = r.rateSecondsPerDay,
               let unc = r.uncertainty {
                Text(formattedRate(rate, precision: unc))
                    .font(.system(size: 60, weight: .semibold, design: .rounded))
                    .foregroundColor(rateColor(rate))
                    .contentTransition(.numericText())
                Text(String(format: "s/day   ± %@", formattedUncertainty(unc)))
                    .font(.system(size: 15, design: .monospaced))
                    .foregroundColor(.secondary)
            } else {
                Text("—")
                    .font(.system(size: 60, weight: .medium, design: .rounded))
                    .foregroundColor(Color(uiColor: .tertiaryLabel))
                Text(stageGuidance)
                    .font(.system(size: 13))
                    .foregroundColor(.secondary)
                    .multilineTextAlignment(.center)
            }
        }
        .frame(height: 126)
    }

    /// Never show more digits than the ± supports — a reading of "−30.4 ± 8" is
    /// three digits of theatre around one digit of knowledge.
    ///
    /// The published ± is a deliberately conservative *bound*; measured against
    /// recorded sessions the typical error is about a fifth of it. So a tenth is
    /// meaningful well past ±1, and only genuinely coarse readings lose it.
    private func formattedRate(_ rate: Double, precision: Double) -> String {
        if precision >= 15 { return String(format: "%+.0f", (rate / 5).rounded() * 5) }
        if precision >= 5 { return String(format: "%+.0f", rate) }
        return String(format: "%+.1f", rate)
    }

    private func formattedUncertainty(_ unc: Double) -> String {
        unc >= 10 ? String(format: "%.0f", unc) : String(format: "%.1f", unc)
    }

    /// Live proof that something is happening during the few seconds before a
    /// number can be published — mic level while we're finding the beat, then
    /// the beat count once ticks are being timed. Disappears with the first
    /// reading, which is the real feedback.
    @ViewBuilder
    private var warmupFeedback: some View {
        if viewModel.isMeasuring, viewModel.result?.rateSecondsPerDay == nil {
            let beats = viewModel.result?.beatsTracked ?? 0
            HStack(spacing: 8) {
                Text(beats > 0 ? "Beats timed" : "Mic level")
                    .font(.system(size: 11))
                    .foregroundColor(.secondary)
                if beats > 0 {
                    Text("\(beats)")
                        .font(.system(size: 11, weight: .medium, design: .monospaced))
                        .foregroundColor(.primary)
                        .contentTransition(.numericText())
                    Spacer()
                    if let bph = viewModel.result?.beatsPerHour, bph > 0 {
                        Text("\(bph) bph")
                            .font(.system(size: 11, design: .monospaced))
                            .foregroundColor(.secondary)
                    }
                } else {
                    GeometryReader { geo in
                        ZStack(alignment: .leading) {
                            Capsule().fill(Color(uiColor: .tertiarySystemFill))
                            Capsule()
                                .fill(Color.green)
                                .frame(width: geo.size.width
                                       * CGFloat(viewModel.audio.inputLevel))
                        }
                    }
                    .frame(height: 4)
                }
            }
            .frame(height: 14)
        }
    }

    // MARK: Stage presentation

    private var stageHeadline: String {
        guard viewModel.isMeasuring else { return "Ready" }
        switch stage {
        case .listening: return "Listening…"
        case .tuning: return "Finding the beat…"
        case .locking: return "Timing the ticks…"
        case .measuring: return "Measuring"
        case .done: return "Reading complete"
        case .unstable: return "Losing the tick"
        case .noSignal: return "Can't hear a tick"
        }
    }

    private var stageGuidance: String {
        guard viewModel.isMeasuring else {
            return "Tap Start, then hold the watch against the microphone"
        }
        switch stage {
        case .listening: return "Hold the watch against the bottom of the phone"
        case .tuning:
            return "Identifying the movement's beat rate and the best band to listen in"
        case .locking: return "Locked on — timing each beat against the phone's clock"
        case .measuring: return ""
        case .done: return "You can stop now"
        case .unstable: return "Press firmer, or move somewhere quieter"
        case .noSignal:
            return "Press the caseback or crystal to the microphone. A silent room matters a lot."
        }
    }

    private var stageColor: Color {
        switch stage {
        case .done: return .green
        case .measuring, .locking: return .primary
        case .unstable, .noSignal: return .orange
        case .listening, .tuning: return .secondary
        }
    }

    @ViewBuilder
    private var stageIcon: some View {
        if !viewModel.isMeasuring {
            Image(systemName: "mic.fill")
        } else {
            switch stage {
            case .listening, .tuning:
                ProgressView().controlSize(.small)
            case .locking, .measuring:
                Image(systemName: "waveform")
            case .done:
                Image(systemName: "checkmark.circle.fill")
            case .unstable, .noSignal:
                Image(systemName: "exclamationmark.triangle.fill")
            }
        }
    }

    // MARK: Precision bar

    /// Progress toward a reading worth trusting. It only ever moves forward: it
    /// tracks how much independent data has been gathered, not how much the last
    /// two readings happened to agree.
    @ViewBuilder
    private var precisionBar: some View {
        if viewModel.isMeasuring {
            let r = viewModel.result
            VStack(spacing: 5) {
                GeometryReader { geo in
                    ZStack(alignment: .leading) {
                        Capsule().fill(Color(uiColor: .tertiarySystemFill))
                        Capsule()
                            .fill(stage == .done ? Color.green : Color.accentColor)
                            .frame(width: geo.size.width * CGFloat(r?.progress ?? 0))
                            .animation(.easeOut(duration: 0.4), value: r?.progress ?? 0)
                    }
                }
                .frame(height: 6)
                HStack {
                    Text(precisionCaption)
                        .font(.system(size: 12))
                        .foregroundColor(stage == .done ? .green : .secondary)
                    Spacer()
                    if let r, let remaining = r.secondsRemaining, stage == .measuring {
                        Text("≈\(Int(remaining.rounded()))s to ±\(Int(Timegrapher.targetPrecision))")
                            .font(.system(size: 12, design: .monospaced))
                            .foregroundColor(.secondary)
                    }
                }
            }
        }
    }

    private var precisionCaption: String {
        switch stage {
        case .done: return "Precision reached"
        case .measuring:
            // No time estimate means the ± is barely improving — more seconds
            // won't rescue this one, so say so instead of promising progress.
            return viewModel.result?.secondsRemaining == nil
                ? "Noisy signal — hold steadier for a tighter reading"
                : "Tightening the reading"
        case .unstable: return "Reading is drifting — hold steady"
        case .locking: return "Gathering beats"
        default: return "Warming up"
        }
    }

    // MARK: Paper tape

    private var paperTape: some View {
        PaperTapeView(result: viewModel.result)
            .background(
                RoundedRectangle(cornerRadius: 14)
                    .fill(Color(uiColor: .secondarySystemBackground))
            )
    }

    // MARK: Metrics

    private var metricsRow: some View {
        HStack(spacing: 10) {
            metric(label: "Beat rate", value: beatRateText, color: .primary)
            metric(label: "Beat error", value: beatErrorText, color: beatErrorColor)
            metric(label: "Signal", value: signalLabel, color: signalColor)
        }
        .onTapGesture { showDetail.toggle() }
    }

    private var beatRateText: String {
        guard let r = viewModel.result, r.beatsPerHour > 0 else { return "—" }
        return "\(r.beatsPerHour)"
    }

    private var beatErrorText: String {
        guard let r = viewModel.result, let ms = r.beatErrorMs else { return "—" }
        return String(format: "%.1f ms", ms)
    }

    private var beatErrorColor: Color {
        guard let r = viewModel.result, let ms = r.beatErrorMs else { return .primary }
        if ms < 0.5 { return .green }
        if ms < 1.0 { return .yellow }
        return .orange
    }

    private func metric(label: String, value: String, color: Color) -> some View {
        VStack(spacing: 3) {
            Text(value)
                .font(.system(size: 16, weight: .medium, design: .monospaced))
                .foregroundColor(color)
                .lineLimit(1)
                .minimumScaleFactor(0.7)
            Text(label)
                .font(.system(size: 10))
                .foregroundColor(.secondary)
                .textCase(.uppercase)
                .tracking(0.5)
        }
        .frame(maxWidth: .infinity)
        .padding(.vertical, 10)
        .background(
            RoundedRectangle(cornerRadius: 10)
                .fill(Color(uiColor: .secondarySystemBackground))
        )
    }

    /// How well we can hear the tick — a judgement about the *audio*, never
    /// about the number. It's the one thing on screen the user can act on.
    private var signalLabel: String {
        guard viewModel.isMeasuring, let r = viewModel.result, r.beatsTracked > 0 else {
            return "—"
        }
        if r.detectionRate < 0.85 || r.jitterMs > 0.5 { return "weak" }
        if r.jitterMs > 0.2 || r.matchScore < 0.85 { return "fair" }
        return "strong"
    }

    private var signalColor: Color {
        switch signalLabel {
        case "strong": return .green
        case "fair": return .yellow
        case "weak": return .orange
        default: return .primary
        }
    }

    // MARK: Detail line (tap the metrics row to reveal)

    /// Internals and the two settings worth overriding. Reachable while idle —
    /// that's when you'd set them, and there is no result to show yet.
    private var detailLine: some View {
        VStack(spacing: 4) {
            if let r = viewModel.result {
                Text(String(format: "%.0f–%.0f kHz · jitter %.2f ms · match %.0f%% · %d beats · %.0fs",
                            r.bandLowHz / 1000, r.bandHighHz / 1000, r.jitterMs,
                            r.detectionRate * 100, r.beatsTracked, r.elapsedSeconds))
                    .font(.system(size: 11, design: .monospaced))
                    .foregroundColor(.secondary)
            }
            beatRatePicker
        }
    }

    private var beatRatePicker: some View {
        VStack(spacing: 2) {
            HStack(spacing: 8) {
                Text("Beat rate")
                    .font(.system(size: 12))
                    .foregroundColor(.secondary)
                Picker("Beat rate", selection: $viewModel.selectedBPH) {
                    Text("Auto").tag(Int?.none)
                    ForEach(Timegrapher.standardRates, id: \.bph) { r in
                        Text("\(r.bph)").tag(Int?.some(r.bph))
                    }
                }
                .pickerStyle(.menu)
                .onChange(of: viewModel.selectedBPH) { _, newValue in
                    viewModel.setManualBPH(newValue)
                }
                Spacer()
            }
        }
    }

    // MARK: Actions

    private var actionButtons: some View {
        HStack(spacing: 12) {
            Button(action: { viewModel.toggle() }) {
                HStack(spacing: 8) {
                    Image(systemName: viewModel.isMeasuring
                          ? "stop.circle.fill" : "play.circle.fill")
                    Text(viewModel.isMeasuring ? "Stop" : "Start measuring")
                }
                .font(.system(size: 17, weight: .semibold))
                .frame(maxWidth: .infinity)
                .frame(height: 50)
            }
            .buttonStyle(.borderedProminent)
            .tint(viewModel.isMeasuring ? .red : .green)

#if DIAGNOSTIC_RECORDING
            if !viewModel.isMeasuring, viewModel.audio.lastRecordingURL != nil {
                Button(action: { shareItems = viewModel.exportItems() }) {
                    Image(systemName: "square.and.arrow.up")
                        .font(.system(size: 22, weight: .medium))
                        .frame(width: 50, height: 50)
                }
                .buttonStyle(.bordered)
                .tint(.blue)
            }
#endif

            Button(action: { showHelp = true }) {
                Image(systemName: "questionmark.circle")
                    .font(.system(size: 22, weight: .medium))
                    .frame(width: 50, height: 50)
            }
            .buttonStyle(.bordered)
            .tint(.secondary)
        }
    }

    private func rateColor(_ rate: Double) -> Color {
        let a = abs(rate)
        if a < 6 { return .green }
        if a < 15 { return .yellow }
        return .orange
    }
}

// MARK: - Paper tape

/// The classic timegrapher strip: every beat plotted by how far it has drifted
/// from a perfect clock. A level line means the watch is on rate; the tilt *is*
/// the rate deviation, and the tightness of the dots is the signal quality. It's
/// what makes the headline number checkable rather than something to take on
/// faith.
private struct PaperTapeView: View {
    let result: TimegrapherResult?

    var body: some View {
        Canvas { context, size in
            let inset: CGFloat = 10
            let midY = size.height / 2

            var centre = Path()
            centre.move(to: CGPoint(x: 0, y: midY))
            centre.addLine(to: CGPoint(x: size.width, y: midY))
            context.stroke(centre, with: .color(.secondary.opacity(0.25)),
                           style: StrokeStyle(lineWidth: 1, dash: [3, 3]))

            guard let r = result, r.trace.count > 3 else {
                context.draw(Text("waiting for beats")
                    .font(.system(size: 11))
                    .foregroundColor(.secondary),
                             at: CGPoint(x: size.width / 2, y: midY))
                return
            }

            let ys = r.trace.map(\.offsetMs)
            let minY = ys.min() ?? -1, maxY = ys.max() ?? 1
            let mid = (maxY + minY) / 2
            let span = max(0.4, maxY - minY)
            let t0 = r.trace[0].time
            let tSpan = max(0.5, r.trace[r.trace.count - 1].time - t0)
            let plotH = (size.height - 2 * inset) / 2

            func point(_ p: BeatPoint) -> CGPoint {
                CGPoint(x: (p.time - t0) / tSpan * (size.width - 2 * inset) + inset,
                        y: midY - CGFloat((p.offsetMs - mid) / span) * plotH)
            }

            // The beats themselves.
            for p in r.trace {
                let pt = point(p)
                let dot = Path(ellipseIn: CGRect(x: pt.x - 1.3, y: pt.y - 1.3,
                                                 width: 2.6, height: 2.6))
                context.fill(dot, with: .color(p.accepted
                                               ? .green.opacity(0.85)
                                               : .orange.opacity(0.5)))
            }

            // The fitted line the headline number comes from.
            guard r.rateSecondsPerDay != nil else { return }
            var sw = 0.0, sx = 0.0, sy = 0.0, sxx = 0.0, sxy = 0.0
            for p in r.trace where p.accepted {
                sw += 1; sx += p.time; sy += p.offsetMs
                sxx += p.time * p.time; sxy += p.time * p.offsetMs
            }
            guard sw >= 3 else { return }
            let den = sxx - sx * sx / sw
            guard den > 0 else { return }
            let slope = (sxy - sx * sy / sw) / den
            let intercept = (sy - slope * sx) / sw
            let a = r.trace[0].time, b = r.trace[r.trace.count - 1].time
            var line = Path()
            line.move(to: point(BeatPoint(time: a, offsetMs: intercept + slope * a,
                                          accepted: true)))
            line.addLine(to: point(BeatPoint(time: b, offsetMs: intercept + slope * b,
                                             accepted: true)))
            context.stroke(line, with: .color(.accentColor.opacity(0.9)),
                           style: StrokeStyle(lineWidth: 1.5, lineCap: .round))
        }
    }
}

// MARK: - View model

final class TimegrapherViewModel: ObservableObject {
    @Published var result: TimegrapherResult?
    @Published var isMeasuring = false
    /// nil = auto-detect the beat rate.
    @Published var selectedBPH: Int?

    let audio = AudioCaptureManager()
    private let analyzer = Timegrapher()
    private let analysisQueue = DispatchQueue(label: "timegrapher.analysis", qos: .userInitiated)
    private var timer: DispatchSourceTimer?
    private var cancellables = Set<AnyCancellable>()

    init() {
        audio.onBuffer = { [weak self] ptr, count in
            self?.analyzer.process(ptr, count: count)
        }
        // `audio` is a nested ObservableObject, so SwiftUI won't observe its
        // changes (permission, input level) through this view model unless we
        // re-broadcast them.
        audio.objectWillChange
            .sink { [weak self] in self?.objectWillChange.send() }
            .store(in: &cancellables)
    }

    func toggle() { isMeasuring ? stop() : start() }

    func setManualBPH(_ bph: Int?) { analyzer.setManualBPH(bph) }

    func start() {
        guard audio.permission == .granted else {
            audio.requestPermission()
            return
        }
        result = nil
        isMeasuring = true
        let bph = selectedBPH
        audio.start(prepare: { [weak self] sampleRate in
            // Runs on the audio queue before samples flow; analyzer is thread-safe.
            self?.analyzer.reset(sampleRate: sampleRate)
            self?.analyzer.setManualBPH(bph)
        }, onStarted: { [weak self] in
            self?.startAnalysisTimer()
        })
    }

    private func startAnalysisTimer() {
        let t = DispatchSource.makeTimerSource(queue: analysisQueue)
        t.schedule(deadline: .now() + 1.0, repeating: 0.5)
        t.setEventHandler { [weak self] in
            guard let self else { return }
            let r = self.analyzer.analyze()
            DispatchQueue.main.async {
                guard self.isMeasuring else { return }
                self.result = r
            }
        }
        t.resume()
        timer = t
    }

    func stop() {
        timer?.cancel()
        timer = nil
        audio.stop()
        isMeasuring = false
    }

#if DIAGNOSTIC_RECORDING
    /// The raw WAV plus a JSON sidecar of the app's computed metrics, for
    /// offline DSP tuning and Swift-vs-Python parity checks.
    func exportItems() -> [Any] {
        var items: [Any] = []
        if let url = audio.lastRecordingURL { items.append(url) }
        if let r = result, let jsonURL = writeMetricsJSON(r, wav: audio.lastRecordingURL) {
            items.append(jsonURL)
        }
        return items
    }

    private func writeMetricsJSON(_ r: TimegrapherResult, wav: URL?) -> URL? {
        let dict: [String: Any] = [
            "wav": wav?.lastPathComponent ?? "",
            "beatsPerHour": r.beatsPerHour,
            "rateSecondsPerDay": r.rateSecondsPerDay as Any,
            "uncertainty": r.uncertainty as Any,
            "beatErrorMs": r.beatErrorMs as Any,
            "jitterMs": r.jitterMs,
            "detectionRate": r.detectionRate,
            "matchScore": r.matchScore,
            "beatsTracked": r.beatsTracked,
            "bandLowHz": r.bandLowHz,
            "bandHighHz": r.bandHighHz,
            "elapsedSeconds": r.elapsedSeconds,
            "manualBPH": selectedBPH as Any,
        ]
        guard let data = try? JSONSerialization.data(
            withJSONObject: dict, options: [.prettyPrinted, .sortedKeys]) else { return nil }
        let base = wav?.deletingPathExtension().lastPathComponent ?? "measurement"
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        let url = docs.appendingPathComponent("\(base).json")
        try? data.write(to: url)
        return url
    }
#endif
}

// MARK: - Share sheet

private struct ShareSheet: UIViewControllerRepresentable {
    let items: [Any]
    func makeUIViewController(context: Context) -> UIActivityViewController {
        UIActivityViewController(activityItems: items, applicationActivities: nil)
    }
    func updateUIViewController(_ vc: UIActivityViewController, context: Context) {}
}

// MARK: - Help

private struct HelpView: View {
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 18) {
                    section(
                        title: "How it works",
                        body: "WatchSync times every single beat of your watch's escapement against the phone's audio clock — the same thing a shop timegrapher does. Roughly 8 ticks a second, each timed to a twentieth of a millisecond, is why a trustworthy reading takes seconds rather than minutes."
                    )
                    section(
                        title: "What the stages mean",
                        body: "Finding the beat: identifying the movement's beat rate and the frequency band its tick is sharpest in. Timing the ticks: locked on, collecting beats. Measuring: the reading is good and getting tighter. Reading complete: the ± has reached ±\(Int(Timegrapher.targetPrecision)) s/day."
                    )
                    section(
                        title: "Read the ±, not the last digit",
                        body: "The ± is measured, not guessed: the app splits the recording into independent chunks and sees how much they disagree. The headline is rounded to the precision the ± actually supports, so it won't show you decimals it can't back up."
                    )
                    section(
                        title: "Signal vs precision",
                        body: "Signal is about the audio — if it says weak or fair, press the watch firmer against the microphone or find a quieter room. Precision is about time: it only needs you to keep holding still."
                    )
                    section(
                        title: "Set up the shot",
                        body: "Press the watch firmly against your phone's microphone (the bottom edge on most iPhones). Through the crystal, with the strap on, works fine — you don't need the caseback off. A silent room matters a lot."
                    )
                    section(
                        title: "The paper tape",
                        body: "Each dot is one beat, plotted by how far it has drifted from a perfect clock. The tilt of the line is the rate — that's where the headline number comes from. Tight dots on a straight line mean a clean measurement; scattered dots mean the microphone is losing the tick."
                    )
                    section(
                        title: "Beat error",
                        body: "The tick-to-tock spacing. Under 0.5 ms is healthy; much above 1 ms means the balance isn't swinging symmetrically about its rest point, which a watchmaker can adjust."
                    )
                    section(
                        title: "Beat rate",
                        body: "Left on Auto, the app identifies the rate itself (18000–36000 bph). Tap the metrics row to override it if your movement has an unusual rate."
                    )
                    section(
                        title: "Recordings",
                        body: "Nothing is recorded. The audio is analysed as it arrives and discarded — the code that could write it to disk isn't in this build at all."
                    )
                    section(
                        title: "The limit",
                        body: "The reading is against your phone's crystal, which is itself good to roughly half a second a day — so ± never claims better than that."
                    )
                }
                .padding()
            }
            .navigationTitle("How to measure")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Button("Done") { dismiss() }.fontWeight(.semibold)
                }
            }
        }
    }

    private func section(title: String, body: String) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(title).font(.system(size: 16, weight: .semibold))
            Text(body)
                .font(.system(size: 15))
                .foregroundColor(.secondary)
                .fixedSize(horizontal: false, vertical: true)
        }
    }
}
