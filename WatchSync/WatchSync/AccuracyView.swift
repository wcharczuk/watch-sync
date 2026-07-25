import SwiftUI
import UIKit
import Combine

struct AccuracyView: View {
    @StateObject private var viewModel = TimegrapherViewModel()
    @State private var showHelp = false
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
        VStack(spacing: 16) {
            rateReadout
            convergenceBar
            beatStrip
                .frame(height: 130)
            metricsRow
            diagnosticsLine
            beatRatePicker
            signalMeter
            Spacer(minLength: 4)
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

    // MARK: Diagnostics line (raw internals for tuning)

    @ViewBuilder
    private var diagnosticsLine: some View {
        if let r = viewModel.result {
            Text(String(format: "snr %.1f · line %.0f · %.0f–%.0fk · %.0fs",
                        r.amplitudeSNR, r.lineSeparation,
                        r.bandLowHz / 1000, r.bandHighHz / 1000, r.elapsedSeconds))
                .font(.system(size: 11, design: .monospaced))
                .foregroundColor(.secondary)
        } else {
            Color.clear.frame(height: 14)
        }
    }

    // MARK: Rate readout

    private var rateReadout: some View {
        VStack(spacing: 6) {
            // Status pill — always tells the user what's happening.
            HStack(spacing: 6) {
                statusIcon
                Text(statusHeadline)
                    .font(.system(size: 14, weight: .semibold))
            }
            .foregroundColor(statusColor)

            if let r = viewModel.result, r.isCalibrated {
                Text(String(format: "%+.1f", r.rateSecondsPerDay))
                    .font(.system(size: 60, weight: .semibold, design: .rounded))
                    .foregroundColor(rateColor(r.rateSecondsPerDay))
                    .contentTransition(.numericText())
                Text(String(format: "s/day   ± %.1f", displayUncertainty ?? r.rateUncertainty))
                    .font(.system(size: 15, design: .monospaced))
                    .foregroundColor(.secondary)
            } else {
                Text("—")
                    .font(.system(size: 60, weight: .medium, design: .rounded))
                    .foregroundColor(Color(uiColor: .quaternaryLabel))
                // While measuring, the convergence bar below owns the guidance
                // line — only show it here in the idle state to avoid duplication.
                if !viewModel.isMeasuring {
                    Text(statusGuidance)
                        .font(.system(size: 13))
                        .foregroundColor(.secondary)
                        .multilineTextAlignment(.center)
                }
            }
        }
        .frame(height: 128)
    }

    // MARK: Measurement state machine

    private enum MeasurePhase { case idle, acquiring, weak, settling, stable }

    private var phase: MeasurePhase {
        guard viewModel.isMeasuring else { return .idle }
        guard let r = viewModel.result, r.isCalibrated else {
            if let r = viewModel.result, r.elapsedSeconds > 6, r.amplitudeSNR < 1.8 { return .weak }
            return .acquiring
        }
        if isStable(r) { return .stable }
        if r.amplitudeSNR < 2 { return .weak }
        return .settling
    }

    private func isStable(_ r: TimegrapherResult) -> Bool {
        guard r.amplitudeSNR >= 2, r.elapsedSeconds >= 20 else { return false }
        guard let s = viewModel.recentSpread else { return false }
        return s <= 1.5
    }

    /// 0…1 progress toward a trustworthy reading — driven by how settled the
    /// live reading is (it has stopped moving), not the pessimistic ±.
    private var convergence: Double {
        guard let r = viewModel.result, r.isCalibrated else { return 0 }
        return viewModel.stabilityScore
    }

    /// Overall trust = tick signal quality × how settled the reading is.
    private var overallConfidence: Double? {
        guard let r = viewModel.result, r.isCalibrated else { return nil }
        return r.confidence * viewModel.stabilityScore
    }

    /// ± to display: the observed spread of recent readings (how settled),
    /// which matches what the user sees, rather than the sub-window bound.
    private var displayUncertainty: Double? {
        guard let r = viewModel.result, r.isCalibrated else { return nil }
        if let s = viewModel.recentSpread { return Swift.max(0.3, s) }
        return r.rateUncertainty
    }

    private var statusHeadline: String {
        switch phase {
        case .idle: return "Ready"
        case .acquiring: return "Listening for the tick…"
        case .weak: return "Faint signal"
        case .settling: return "Measuring…"
        case .stable: return "Reading stable"
        }
    }

    private var statusGuidance: String {
        switch phase {
        case .idle: return "Tap Start, then press the watch to the mic"
        case .acquiring: return "Press the watch firmly against the microphone"
        case .weak: return "Press firmer, or move somewhere quieter"
        case .settling: return "Keep holding — the reading is tightening"
        case .stable: return "You can stop now"
        }
    }

    private var statusColor: Color {
        switch phase {
        case .stable: return .green
        case .settling: return .primary
        case .weak: return .orange
        case .acquiring, .idle: return .secondary
        }
    }

    @ViewBuilder
    private var statusIcon: some View {
        switch phase {
        case .acquiring:
            ProgressView().controlSize(.small)
        case .weak:
            Image(systemName: "exclamationmark.triangle.fill")
        case .settling:
            Image(systemName: "waveform")
        case .stable:
            Image(systemName: "checkmark.circle.fill")
        case .idle:
            Image(systemName: "mic.fill")
        }
    }

    // MARK: Convergence bar

    @ViewBuilder
    private var convergenceBar: some View {
        if viewModel.isMeasuring {
            VStack(spacing: 4) {
                GeometryReader { geo in
                    ZStack(alignment: .leading) {
                        Capsule().fill(Color(uiColor: .tertiarySystemFill))
                        Capsule()
                            .fill(phase == .stable ? Color.green : Color.accentColor)
                            .frame(width: geo.size.width * CGFloat(convergence))
                            .animation(.easeOut(duration: 0.4), value: convergence)
                    }
                }
                .frame(height: 6)
                Text(statusGuidance)
                    .font(.system(size: 12))
                    .foregroundColor(phase == .stable ? .green : .secondary)
            }
        }
    }

    // MARK: Beat strip (classic timegrapher paper-tape view)

    private var beatStrip: some View {
        DriftTraceView(result: viewModel.result)
            .background(
                RoundedRectangle(cornerRadius: 14)
                    .fill(Color(uiColor: .secondarySystemBackground))
            )
    }

    // MARK: Metrics

    private var metricsRow: some View {
        HStack(spacing: 12) {
            metric(
                label: "Beat rate",
                value: viewModel.result.map { "\($0.beatsPerHour)" } ?? "—",
                color: .primary
            )
            metric(
                label: "Signal",
                value: viewModel.result.map { String(format: "%.1f", $0.amplitudeSNR) } ?? "—",
                color: signalColor(viewModel.result?.amplitudeSNR)
            )
            metric(
                label: "Confidence",
                value: overallConfidence.map { String(format: "%.0f%%", $0 * 100) } ?? "—",
                color: .primary
            )
        }
    }

    private func metric(label: String, value: String, color: Color) -> some View {
        VStack(spacing: 3) {
            Text(value)
                .font(.system(size: 17, weight: .medium, design: .monospaced))
                .foregroundColor(color)
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

    // MARK: Beat-rate picker (Auto + manual override)

    private var beatRatePicker: some View {
        HStack(spacing: 8) {
            Text("Beat rate")
                .font(.system(size: 13))
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
            if viewModel.selectedBPH == nil, let r = viewModel.result, r.isCalibrated {
                Text(String(format: "detected %.2f/s", r.beatsPerSecond))
                    .font(.system(size: 12, design: .monospaced))
                    .foregroundColor(.secondary)
            }
        }
    }

    // MARK: Signal meter

    /// While acquiring, shows raw mic level so the user knows it's live. Once a
    /// tick is detected, shows the tick-signal strength (ampSNR) — the number
    /// that actually predicts a good reading — with an actionable color.
    private var signalMeter: some View {
        let usingSNR = viewModel.result?.isCalibrated ?? false
        let snr = viewModel.result?.amplitudeSNR ?? 0
        let fill = usingSNR
            ? min(1.0, snr / 6.0)                       // ampSNR 6 = full
            : Double(viewModel.audio.inputLevel)
        return VStack(spacing: 4) {
            HStack {
                Text(usingSNR ? "Tick signal" : "Mic level")
                    .font(.system(size: 11)).foregroundColor(.secondary)
                Spacer()
                if usingSNR {
                    Text(snr >= 3 ? "strong" : snr >= 2 ? "ok" : "weak")
                        .font(.system(size: 11, weight: .medium))
                        .foregroundColor(signalColor(snr))
                }
            }
            GeometryReader { geo in
                ZStack(alignment: .leading) {
                    Capsule().fill(Color(uiColor: .tertiarySystemFill))
                    Capsule()
                        .fill(usingSNR ? signalColor(snr) : Color.green)
                        .frame(width: geo.size.width * CGFloat(fill))
                }
            }
            .frame(height: 6)
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

            if !viewModel.isMeasuring, viewModel.audio.lastRecordingURL != nil {
                Button(action: { shareItems = viewModel.exportItems() }) {
                    Image(systemName: "square.and.arrow.up")
                        .font(.system(size: 22, weight: .medium))
                        .frame(width: 50, height: 50)
                }
                .buttonStyle(.bordered)
                .tint(.blue)
            }

            Button(action: { showHelp = true }) {
                Image(systemName: "questionmark.circle")
                    .font(.system(size: 22, weight: .medium))
                    .frame(width: 50, height: 50)
            }
            .buttonStyle(.bordered)
            .tint(.secondary)
        }
    }

    // MARK: Colors

    private func rateColor(_ rate: Double) -> Color {
        let a = abs(rate)
        if a < 6 { return .green }
        if a < 15 { return .yellow }
        return .red
    }

    private func signalColor(_ snr: Double?) -> Color {
        guard let snr else { return .primary }
        if snr >= 3 { return .green }
        if snr >= 1.8 { return .yellow }
        return .red
    }
}

// MARK: - Beat strip

/// The rate-drift trace: cumulative timing offset (ms) vs time. A level line
/// means the watch is on rate; a consistent slope is the rate deviation. The
/// straightness of the line shows how trustworthy the reading is.
private struct DriftTraceView: View {
    let result: TimegrapherResult?

    var body: some View {
        Canvas { context, size in
            let midY = size.height / 2
            var center = Path()
            center.move(to: CGPoint(x: 0, y: midY))
            center.addLine(to: CGPoint(x: size.width, y: midY))
            context.stroke(center, with: .color(.secondary.opacity(0.3)),
                           style: StrokeStyle(lineWidth: 1, dash: [3, 3]))

            guard let r = result, r.trace.count > 2 else { return }
            let ys = r.trace.map { $0.offsetMs }
            let minY = ys.min() ?? -1, maxY = ys.max() ?? 1
            let span = max(0.4, maxY - minY)          // ms, min scale
            let t0 = r.trace.first!.time
            let tSpan = max(0.5, r.trace.last!.time - t0)

            var path = Path()
            for (i, p) in r.trace.enumerated() {
                let x = (p.time - t0) / tSpan * Double(size.width)
                // Center the trace vertically around its own midpoint.
                let mid = (maxY + minY) / 2
                let y = midY - CGFloat((p.offsetMs - mid) / span) * (size.height * 0.42)
                let pt = CGPoint(x: x, y: y)
                if i == 0 { path.move(to: pt) } else { path.addLine(to: pt) }
            }
            context.stroke(path, with: .color(.green),
                           style: StrokeStyle(lineWidth: 2, lineCap: .round, lineJoin: .round))
        }
    }
}

// MARK: - View model

final class TimegrapherViewModel: ObservableObject {
    @Published var result: TimegrapherResult?
    @Published var isMeasuring = false
    /// nil = auto-detect the beat rate.
    @Published var selectedBPH: Int?
    /// Recent calibrated rate readings, for convergence/stability detection.
    @Published private(set) var rateHistory: [Double] = []

    /// Spread (std dev, s/day) of recent readings — small = settled.
    var recentSpread: Double? {
        guard rateHistory.count >= 4 else { return nil }
        let recent = rateHistory.suffix(8)
        let m = recent.reduce(0, +) / Double(recent.count)
        let v = recent.reduce(0) { $0 + ($1 - m) * ($1 - m) } / Double(recent.count - 1)
        return v.squareRoot()
    }

    /// 0…1 how settled the live reading is (has it stopped moving?). This — not
    /// the pessimistic sub-window ± — is what tells us the reading has converged.
    var stabilityScore: Double {
        guard let s = recentSpread else { return 0 }
        return max(0, min(1, (2.5 - s) / 2.5))   // spread 0 → 1, spread ≥2.5 → 0
    }

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

    func toggle() {
        isMeasuring ? stop() : start()
    }

    func setManualBPH(_ bph: Int?) {
        analyzer.setManualBPH(bph)
    }

    func start() {
        guard audio.permission == .granted else {
            audio.requestPermission()
            return
        }
        result = nil
        rateHistory = []
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
                if let r {
                    self.result = r
                    if r.isCalibrated {
                        self.rateHistory.append(r.rateSecondsPerDay)
                        if self.rateHistory.count > 16 { self.rateHistory.removeFirst() }
                    }
                }
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
            "beatsPerSecond": r.beatsPerSecond,
            "rateSecondsPerDay": r.rateSecondsPerDay,
            "rateUncertainty": r.rateUncertainty,
            "amplitudeSNR": r.amplitudeSNR,
            "lineProminence": r.lineSeparation,
            "bandLowHz": r.bandLowHz,
            "bandHighHz": r.bandHighHz,
            "confidence": r.confidence,
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
                        body: "WatchSync listens to your mechanical watch's escapement — the high-frequency \"tick\" each beat makes. By tracking the beat's timing against your phone's audio clock, it measures how fast or slow the watch runs."
                    )
                    section(
                        title: "Set up the shot",
                        body: "Press the watch firmly against your phone's microphone (the bottom edge on most iPhones) so the tick conducts into the phone. Through the crystal, with the strap on, works fine — you don't need the caseback. A silent room matters a lot."
                    )
                    section(
                        title: "Record long enough",
                        body: "Accuracy improves the longer you record. A short clip can be off by several s/day; give it 1–2 minutes for a reading you can trust. Watch the ± figure — it shrinks as the measurement settles."
                    )
                    section(
                        title: "Beat rate",
                        body: "Leave this on Auto — WatchSync detects your movement's beat rate (18000–36000 bph) automatically. Only override it if your watch has an unusual rate the detector can't lock onto."
                    )
                    section(
                        title: "Reading the result",
                        body: "Rate: negative runs slow, positive runs fast; ±5–15 s/day is typical for a mechanical watch. Signal and Confidence show how well it's hearing the tick — if they're low, improve contact or find a quieter spot."
                    )
                    section(
                        title: "The trace",
                        body: "The line is the watch's cumulative timing offset. Level means it's on rate; a steady tilt is the rate deviation. A straight line means a trustworthy reading; a wandering one means keep recording."
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
