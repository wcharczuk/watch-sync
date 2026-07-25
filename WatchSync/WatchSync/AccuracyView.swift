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
            rateChart
                .frame(height: 132)
            metricsRow
            if showDetail { detailLine }
            Spacer(minLength: 0)
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
        case .unstable:
            return roomIsLoud ? "Too much noise — somewhere quieter would help"
                              : "Losing contact — press the watch gently onto the mic"
        case .noSignal:
            // Two very different failures wear the same face. The room's level
            // tells them apart: measured across test recordings, a quiet room
            // sits near 0.1 and talking near 0.3, so the advice can be specific
            // instead of a list of things to try.
            return roomIsLoud
                ? "Too noisy here. Find a quieter spot, and don't talk while measuring — speech drowns the tick completely."
                : "Rest the watch face-down on the microphone and press gently."
        }
    }

    /// Is the room loud enough to be the problem? Quiet rooms measure ~0.1 on
    /// this meter, a conversation ~0.3.
    private var roomIsLoud: Bool { viewModel.audio.inputLevel > 0.2 }

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

    private var rateChart: some View {
        RateAxisView(result: viewModel.result)
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

#if DEBUG && DIAGNOSTIC_RECORDING
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

// MARK: - Rate axis

/// One axis, in seconds per day: where the reading sits, and how well we know it.
///
/// No spec bands, no chronometer zones, no verdict about the watch — those are
/// judgements this app has no business making. What it shows is the measurement:
/// every independent sub-window as a tick, the published rate as a marker, and
/// the ± as a bar through it that visibly shrinks as the reading tightens.
///
/// Earlier attempts plotted cumulative drift (a straight line by construction,
/// restating the number) and per-window detection bars (honest, but about the
/// microphone rather than the answer).
private struct RateAxisView: View {
    let result: TimegrapherResult?

    var body: some View {
        Canvas { context, size in
            let inset: CGFloat = 22
            let axisY = size.height * 0.56
            let plotW = size.width - 2 * inset

            guard let r = result, let rate = r.rateSecondsPerDay, let unc = r.uncertainty else {
                context.draw(Text(result?.beatsTracked ?? 0 > 0
                                  ? "measuring…" : "waiting for the tick")
                    .font(.system(size: 11))
                    .foregroundColor(.secondary),
                             at: CGPoint(x: size.width / 2, y: axisY))
                return
            }

            // A span that always contains zero, the ± and every sub-window, then
            // rounded outward to a whole number of seconds so the labels are
            // readable rather than arbitrary.
            var reach = max(abs(rate) + unc, 5)
            for s in r.rateSamples { reach = max(reach, abs(s.rate)) }
            let step: Double = reach <= 6 ? 2 : (reach <= 15 ? 5 : 10)
            let limit = (reach / step).rounded(.up) * step

            func x(_ value: Double) -> CGFloat {
                inset + CGFloat((value + limit) / (2 * limit)) * plotW
            }

            // Axis.
            var axis = Path()
            axis.move(to: CGPoint(x: inset, y: axisY))
            axis.addLine(to: CGPoint(x: size.width - inset, y: axisY))
            context.stroke(axis, with: .color(.secondary.opacity(0.4)),
                           style: StrokeStyle(lineWidth: 1))

            var value = -limit
            while value <= limit + 0.001 {
                let isZero = abs(value) < 0.001
                var tick = Path()
                tick.move(to: CGPoint(x: x(value), y: axisY - (isZero ? 8 : 4)))
                tick.addLine(to: CGPoint(x: x(value), y: axisY + (isZero ? 8 : 4)))
                context.stroke(tick, with: .color(.secondary.opacity(isZero ? 0.8 : 0.4)),
                               style: StrokeStyle(lineWidth: isZero ? 1.5 : 1))
                context.draw(Text(isZero ? "0" : String(format: "%+.0f", value))
                    .font(.system(size: 9, design: .monospaced))
                    .foregroundColor(.secondary),
                             at: CGPoint(x: x(value), y: axisY + 16), anchor: .top)
                value += step
            }

            context.draw(Text("slow").font(.system(size: 9)).foregroundColor(.secondary),
                         at: CGPoint(x: inset, y: axisY + 32), anchor: .leading)
            context.draw(Text("fast").font(.system(size: 9)).foregroundColor(.secondary),
                         at: CGPoint(x: size.width - inset, y: axisY + 32), anchor: .trailing)

            // Each independent sub-window, as a tick above the axis. The width of
            // this cloud is the raw scatter; the ± below is what it implies about
            // the average.
            for sample in r.rateSamples {
                var mark = Path()
                mark.move(to: CGPoint(x: x(sample.rate), y: axisY - 12))
                mark.addLine(to: CGPoint(x: x(sample.rate), y: axisY - 26))
                context.stroke(mark, with: .color(.secondary.opacity(0.45)),
                               style: StrokeStyle(lineWidth: 1.5, lineCap: .round))
            }
            if !r.rateSamples.isEmpty {
                context.draw(Text("\(r.rateSamples.count) independent windows")
                    .font(.system(size: 9))
                    .foregroundColor(.secondary),
                             at: CGPoint(x: size.width / 2, y: axisY - 34), anchor: .bottom)
            }

            // The reading: a bar the width of the ±, with the marker on it.
            let barY = axisY - 2
            let barRect = CGRect(x: x(rate - unc), y: barY - 3,
                                 width: max(4, x(rate + unc) - x(rate - unc)), height: 6)
            context.fill(Path(roundedRect: barRect, cornerRadius: 3),
                         with: .color(.accentColor.opacity(0.35)))
            let dot = Path(ellipseIn: CGRect(x: x(rate) - 5, y: barY - 5, width: 10, height: 10))
            context.fill(dot, with: .color(.accentColor))
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

#if DEBUG && DIAGNOSTIC_RECORDING
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
                        body: "Rest the watch on your phone's microphone (the bottom edge on most iPhones) with firm, steady contact. The crystal works as well as the caseback — often better, since a bracelet or single-pass strap frequently covers the back — so face-down is usually the easiest way to get metal or glass onto the mic. Keep the strap on; you don't need to open anything."
                    )
                    section(
                        title: "The chart",
                        body: "One axis in seconds per day. The marker is the reading and the bar through it is the ±, so the bar visibly narrows as the measurement tightens. The small ticks above the axis are the independent windows the reading was averaged from — their spread is the raw scatter, the bar is what that implies about the average. There are deliberately no \"good/bad\" zones: what counts as acceptable depends on the watch, and that judgement isn't the app's to make."
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
                        title: "What defeats it",
                        body: "Talking is the one thing that reliably stops a reading — speech puts energy right where the tick lives, and in tests no amount of extra listening recovered it. Stay quiet and the app copes with a lot: a running air conditioner is almost entirely low rumble and gets filtered out before it reaches the tick. The reading is also against your phone's own crystal, good to roughly half a second a day, so the ± never claims better than that."
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
