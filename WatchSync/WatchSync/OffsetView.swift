import SwiftUI
import UIKit
import AVFoundation
import Combine

/// How many seconds the watch is off, right now.
///
/// The Accuracy screen answers "how fast is it gaining?"; this one answers
/// "where has it got to?". Point the camera at the dial and the app compares
/// where the second hand is with where the phone's clock says it should be.
struct OffsetView: View {
    @StateObject private var viewModel = OffsetViewModel()
    /// Zoom at the start of the pinch in progress.
    @State private var pinchBase: CGFloat?

    var body: some View {
        ZStack {
            switch viewModel.camera.permission {
            case .denied:
                permissionDeniedView
            case .granted:
                content
            case .unknown:
                ProgressView("Requesting camera access…")
            }
        }
        .onAppear { viewModel.camera.requestPermission() }
    }

    // MARK: Permission

    private var permissionDeniedView: some View {
        VStack(spacing: 16) {
            Spacer()
            Image(systemName: "video.slash.fill")
                .font(.system(size: 48))
                .foregroundColor(.secondary)
            Text("Camera Access Required")
                .font(.headline)
            Text("WatchSync watches the second hand to compare it with the phone's clock. Open Settings to grant camera permission.")
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
        GeometryReader { geo in
            // Decided by the shape of the space, not by device or orientation:
            // a phone on its side and a phone unfolded are both wider than
            // tall, and stacking there squeezes the viewfinder — the one thing
            // that wants every point it can get — between its own controls.
            Group {
                if geo.size.width > geo.size.height * 1.15 {
                    HStack(spacing: 28) {
                        viewfinder
                        VStack(spacing: 14) {
                            Spacer(minLength: 0)
                            readout
                            progressBar
                            footnoteText
                            Spacer(minLength: 0)
                            measureButton
                        }
                        .frame(maxWidth: 340)
                    }
                } else {
                    VStack(spacing: 14) {
                        readout
                        progressBar
                        viewfinder
                        footnoteText
                        Spacer(minLength: 0)
                        measureButton
                    }
                    .frame(maxWidth: 600)
                }
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
        }
        .padding(.horizontal)
        .padding(.top, 12)
        .padding(.bottom, 6)
        .onAppear { viewModel.camera.start() }
        .onDisappear {
            viewModel.stop()
            viewModel.camera.stop()
        }
    }

    private var footnoteText: some View {
        Text(footnote)
            .font(.system(size: 11))
            .foregroundColor(.secondary)
            .multilineTextAlignment(.center)
            .fixedSize(horizontal: false, vertical: true)
    }

    private var measureButton: some View {
        Button(action: { viewModel.toggle() }) {
            Label(viewModel.isMeasuring ? "Stop" : "Measure",
                  systemImage: viewModel.isMeasuring ? "stop.fill" : "camera.fill")
                .font(.system(size: 17, weight: .semibold))
                .lineLimit(1)
                .frame(maxWidth: .infinity)
                .frame(height: 50)
        }
        .buttonStyle(.borderedProminent)
        .tint(viewModel.isMeasuring ? .red : .green)
        .disabled(!viewModel.camera.isAvailable)
    }

    // MARK: Readout

    private var readout: some View {
        VStack(spacing: 6) {
            HStack(spacing: 6) {
                stageIcon
                Text(headline)
                    .font(.system(size: 14, weight: .semibold))
            }
            .foregroundColor(stageColor)
            .padding(.horizontal, 10)
            .padding(.vertical, 4)
            .background(Capsule().fill(stageColor.opacity(isComplete ? 0.14 : 0.0)))
            .animation(.easeInOut(duration: 0.3), value: isComplete)

            if let r = viewModel.reading, let offset = r.offsetSeconds, let unc = r.uncertainty {
                Text(formattedOffset(offset, motion: r.motion))
                    .font(.system(size: 60, weight: .semibold, design: .rounded))
                    .foregroundColor(.primary)
                    .contentTransition(.numericText())
                Text(String(format: "seconds %@   ± %@",
                            offset >= 0 ? "fast" : "slow", formattedUncertainty(unc)))
                    .font(.system(size: 15, design: .monospaced))
                    .foregroundColor(.secondary)
            } else {
                Text("—")
                    .font(.system(size: 60, weight: .medium, design: .rounded))
                    .foregroundColor(Color(uiColor: .tertiaryLabel))
                Text(guidance)
                    .font(.system(size: 13))
                    .foregroundColor(.secondary)
                    .multilineTextAlignment(.center)
            }
        }
        .frame(height: 126)
    }

    /// A ticking hand is timed by the instant it jumps, which is good to a few
    /// hundredths; a sweeping one by its angle, which is good to about a fifth
    /// of a second. Each gets the digits it has earned.
    private func formattedOffset(_ offset: Double, motion: OffsetReading.HandMotion) -> String {
        String(format: motion == .tick ? "%+.2f" : "%+.1f", offset)
    }

    private func formattedUncertainty(_ unc: Double) -> String {
        String(format: unc < 0.1 ? "%.2f" : "%.1f", unc)
    }

    private var isComplete: Bool { viewModel.reading?.stage == .done }

    private var hasOffset: Bool { viewModel.reading?.offsetSeconds != nil }

    private var headline: String {
        if isComplete { return "Reading complete" }
        guard viewModel.isMeasuring else { return hasOffset ? "Stopped early" : "Ready" }
        return hasOffset ? "Reading the second hand" : "Looking for the second hand…"
    }

    private var guidance: String {
        guard viewModel.camera.isAvailable else { return "This device has no camera to read the dial with" }
        guard viewModel.isMeasuring else {
            return "Fill the circle with the dial, 12 at the top, then tap Measure"
        }
        // The first reading needs three seconds of frames whatever happens; only
        // after that is silence worth explaining.
        return viewModel.searchingTooLong
            ? "Nothing in the circle is moving like a second hand. Centre the dial, fill the circle, and hold steady."
            : "Hold steady with the dial in the circle"
    }

    private var footnote: String {
        let minute = "Only the second hand is read, so this is the error to the nearest minute: a watch 58 s slow reads as 2 s fast."
        guard let twist = viewModel.reading?.dialTwist else { return minute }
        return String(format: "Dial read as turned %.1f° %@ and corrected for. ", abs(twist),
                      twist >= 0 ? "clockwise" : "anticlockwise") + minute
    }

    private var stageColor: Color {
        if isComplete { return .green }
        if !viewModel.isMeasuring { return .secondary }
        return hasOffset ? .primary : (viewModel.searchingTooLong ? .orange : .secondary)
    }

    @ViewBuilder
    private var stageIcon: some View {
        if isComplete {
            Image(systemName: "checkmark.circle.fill")
        } else if !viewModel.isMeasuring {
            Image(systemName: "camera.fill")
        } else if hasOffset {
            Image(systemName: "scope")
        } else if viewModel.searchingTooLong {
            Image(systemName: "exclamationmark.triangle.fill")
        } else {
            ProgressView().controlSize(.small)
        }
    }

    @ViewBuilder
    private var progressBar: some View {
        if viewModel.isMeasuring {
            GeometryReader { geo in
                ZStack(alignment: .leading) {
                    Capsule().fill(.clear).instrumentGroove(cornerRadius: 4)
                    Capsule()
                        .fill(Instrument.band)
                        .overlay(Capsule().fill(Instrument.gloss).opacity(0.7))
                        .frame(width: geo.size.width * CGFloat(viewModel.reading?.progress ?? 0))
                        .animation(.easeOut(duration: 0.4), value: viewModel.reading?.progress ?? 0)
                }
            }
            .frame(height: 8)
        }
    }

    // MARK: Viewfinder

    private var viewfinder: some View {
        ZStack {
            if viewModel.camera.isAvailable {
                CameraPreview(camera: viewModel.camera)
            } else {
                Rectangle().fill(Instrument.face)
                Image(systemName: "video.slash")
                    .font(.system(size: 36, weight: .light))
                    .foregroundColor(.secondary)
            }
            GuideOverlay(reading: viewModel.reading)
            Text(String(format: "%.1f×", viewModel.camera.zoom))
                .font(.system(size: 11, weight: .medium, design: .monospaced))
                .foregroundColor(.white.opacity(0.8))
                .padding(.horizontal, 7)
                .padding(.vertical, 3)
                .background(Capsule().fill(.black.opacity(0.45)))
                .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .bottomTrailing)
                .padding(8)
                .accessibilityLabel("Zoom, pinch to change")
        }
        .aspectRatio(1, contentMode: .fit)
        .clipShape(RoundedRectangle(cornerRadius: 16, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: 16, style: .continuous)
                .strokeBorder(Instrument.hairline.opacity(0.6), lineWidth: 1)
        )
        .gesture(
            MagnificationGesture()
                .onChanged { scale in
                    let base = pinchBase ?? viewModel.camera.zoom
                    pinchBase = base
                    viewModel.camera.setZoom(base * scale)
                }
                .onEnded { _ in pinchBase = nil }
        )
    }
}

// MARK: - Guide overlay

/// The circle the dial goes in, and — once there's a reading — a bead riding
/// the rim where the app believes the second hand is pointing.
///
/// The circle is not decoration: the reader samples exactly this circle, with
/// 12 at the top, so lining the dial up with it *is* the calibration. The bead
/// is the check on the result. If it tracks the real hand around the dial the
/// reading is right, and if the app has locked onto something else, that is
/// obvious at a glance in a way no confidence figure could make it.
private struct GuideOverlay: View {
    let reading: OffsetReading?

    var body: some View {
        TimelineView(.animation(minimumInterval: 1.0 / 30, paused: reading?.offsetSeconds == nil)) { timeline in
            Canvas { context, size in
                let center = CGPoint(x: size.width / 2, y: size.height / 2)
                let radius = CGFloat(SecondHandReader.guideRadiusFraction) * min(size.width, size.height) / 2
                let circle = Path(ellipseIn: CGRect(x: center.x - radius, y: center.y - radius,
                                                    width: radius * 2, height: radius * 2))

                // Dim everything that isn't the dial.
                var surround = Path(CGRect(origin: .zero, size: size))
                surround.addPath(circle)
                context.fill(surround, with: .color(.black.opacity(0.45)), style: FillStyle(eoFill: true))
                context.stroke(circle, with: .color(.white.opacity(0.85)),
                               style: StrokeStyle(lineWidth: 1.5))

                // Once the reader has found the dial's markers, the ticks turn
                // to sit on them: the twist is measured, not trusted, and
                // this is where to see that it found the right thing.
                let twist = reading?.dialTwist
                func point(_ degrees: Double, _ r: CGFloat) -> CGPoint {
                    let a = (degrees + (twist ?? 0)) * .pi / 180
                    return CGPoint(x: center.x + r * CGFloat(sin(a)), y: center.y - r * CGFloat(cos(a)))
                }

                // Hour ticks outside the rim, to line the dial's markers up
                // against. A degree of twist is a sixth of a second.
                let tickColor: Color = twist == nil ? .white : Instrument.amber
                for hour in 1..<12 {
                    let quarter = hour % 3 == 0
                    var tick = Path()
                    tick.move(to: point(Double(hour) * 30, radius + 3))
                    tick.addLine(to: point(Double(hour) * 30, radius + (quarter ? 11 : 7)))
                    context.stroke(tick, with: .color(tickColor.opacity(quarter ? 0.9 : 0.55)),
                                   style: StrokeStyle(lineWidth: quarter ? 2 : 1, lineCap: .round))
                }
                var twelve = Path()
                twelve.move(to: point(0, radius + 2))
                twelve.addLine(to: point(-2.2, radius + 13))
                twelve.addLine(to: point(2.2, radius + 13))
                twelve.closeSubpath()
                context.fill(twelve, with: .color(Instrument.amber))

                // Where the hands' pivot belongs.
                let dot = CGRect(x: center.x - 3, y: center.y - 3, width: 6, height: 6)
                context.stroke(Path(ellipseIn: dot), with: .color(Instrument.amber.opacity(0.9)),
                               style: StrokeStyle(lineWidth: 1.5))

                if let r = reading, let offset = r.offsetSeconds {
                    var watch = (timeline.date.timeIntervalSince1970 + offset)
                        .truncatingRemainder(dividingBy: 60)
                    if r.motion == .tick { watch = watch.rounded(.down) }
                    let at = point(watch * 6, radius)
                    let bead = CGRect(x: at.x - 5, y: at.y - 5, width: 10, height: 10)
                    context.fill(Path(ellipseIn: bead.insetBy(dx: -1.5, dy: -1.5)),
                                 with: .color(.black.opacity(0.5)))
                    context.fill(Path(ellipseIn: bead), with: .color(Instrument.amber))
                }
            }
        }
        .allowsHitTesting(false)
    }
}

// MARK: - Camera preview

private struct CameraPreview: UIViewRepresentable {
    let camera: CameraCaptureManager

    final class PreviewView: UIView {
        override class var layerClass: AnyClass { AVCaptureVideoPreviewLayer.self }
        var previewLayer: AVCaptureVideoPreviewLayer { layer as! AVCaptureVideoPreviewLayer }
    }

    func makeUIView(context: Context) -> PreviewView {
        let view = PreviewView()
        view.previewLayer.session = camera.session
        // Fill, centred: with a square view this shows the middle square of the
        // frame, which is the square the reader's guide circle is defined in.
        view.previewLayer.videoGravity = .resizeAspectFill
        camera.attach(previewLayer: view.previewLayer)
        return view
    }

    func updateUIView(_ view: PreviewView, context: Context) {}
}

// MARK: - View model

final class OffsetViewModel: ObservableObject {
    @Published var reading: OffsetReading?
    @Published var isMeasuring = false
    /// Measuring for a while with nothing found: time to say what might be wrong.
    @Published private(set) var searchingTooLong = false

    let camera = CameraCaptureManager()
    private let reader = SecondHandReader()
    private var cancellables = Set<AnyCancellable>()

    // Touched only on the camera's video queue.
    private var active = false
    private var framesSinceAnalysis = 0
    private var firstFrameTime: Double?

    init() {
        camera.onFrame = { [weak self] luma, width, height, bytesPerRow, time in
            self?.handle(luma, width: width, height: height, bytesPerRow: bytesPerRow, time: time)
        }
        // Nested ObservableObject: re-broadcast, as the timegrapher does for audio.
        camera.objectWillChange
            .sink { [weak self] in self?.objectWillChange.send() }
            .store(in: &cancellables)
    }

    func toggle() { isMeasuring ? stop() : start() }

    func start() {
        reading = nil
        searchingTooLong = false
        isMeasuring = true
        camera.videoQueue.async { [weak self] in
            guard let self else { return }
            self.reader.reset()
            self.framesSinceAnalysis = 0
            self.firstFrameTime = nil
            self.active = true
        }
    }

    func stop() {
        isMeasuring = false
        camera.videoQueue.async { [weak self] in self?.active = false }
    }

    private func handle(_ luma: UnsafePointer<UInt8>, width: Int, height: Int,
                        bytesPerRow: Int, time: Double) {
        guard active else { return }
        reader.process(luma: luma, width: width, height: height,
                       bytesPerRow: bytesPerRow, time: time)
        let started = firstFrameTime ?? time
        firstFrameTime = started
        framesSinceAnalysis += 1
        // About three times a second: the analysis takes a frame or two, and
        // frames dropped meanwhile cost nothing since each has its own time.
        guard framesSinceAnalysis >= 10 else { return }
        framesSinceAnalysis = 0
        let result = reader.analyze()
        if result.stage == .done { active = false }
        let slow = result.offsetSeconds == nil && time - started > 7
        DispatchQueue.main.async { [weak self] in
            guard let self, self.isMeasuring else { return }
            self.reading = result
            self.searchingTooLong = slow
            if result.stage == .done {
                // The phone is held over the watch, and the eye is on keeping
                // the dial in the circle — finish with something that's felt.
                self.isMeasuring = false
                UINotificationFeedbackGenerator().notificationOccurred(.success)
            }
        }
    }
}
