import Foundation
import AVFoundation
import Combine

/// Captures microphone audio for the timegrapher.
///
/// Uses `.measurement` mode so the OS applies no automatic gain control or other
/// signal conditioning — the raw transient timing of each escapement beat is
/// exactly what we need to preserve.
///
/// Audio is analysed in memory and discarded. Writing a WAV to disk exists only
/// in a developer's own debug build: it requires both `DEBUG` and an opt-in the
/// repository cannot carry — see `Config/Diagnostics.xcconfig`. Requiring `DEBUG`
/// as well means a Release build cannot contain the recording code even if the
/// setting were switched on by mistake, so a shipped app has no path to it.
///
/// All `AVAudioSession` / `AVAudioEngine` calls run on a dedicated serial queue,
/// never the main thread: those synchronous AVFoundation calls otherwise trip the
/// "unsafeForcedSync from a Swift Concurrent context" runtime check, and audio
/// session activation can block.
final class AudioCaptureManager: ObservableObject {
    enum PermissionState { case unknown, granted, denied }

    @Published var permission: PermissionState = .unknown
    /// Smoothed input level (0…1) for the live meter.
    @Published var inputLevel: Float = 0
#if DEBUG && DIAGNOSTIC_RECORDING
    /// URL of the most recently finished raw recording (diagnostic builds only).
    @Published var lastRecordingURL: URL?
#endif

    /// Called on the audio thread with each mono buffer. Keep it cheap.
    var onBuffer: ((UnsafePointer<Float>, Int) -> Void)?

    // Built on first use, not at init: constructing the engine reaches for the
    // input hardware, which is enough to make iOS raise the microphone prompt
    // before the user has asked to measure anything.
    private lazy var engine = AVAudioEngine()
    private let engineQueue = DispatchQueue(label: "audio.engine", qos: .userInitiated)
    private var running = false

#if DEBUG && DIAGNOSTIC_RECORDING
    // Raw-audio recording for offline DSP tuning.
    private var recordFile: AVAudioFile?
    private var recordURL: URL?
#endif

    // MARK: Permission

    func requestPermission() {
        switch AVAudioApplication.shared.recordPermission {
        case .granted: permission = .granted
        case .denied: permission = .denied
        case .undetermined:
            AVAudioApplication.requestRecordPermission { [weak self] granted in
                DispatchQueue.main.async {
                    self?.permission = granted ? .granted : .denied
                }
            }
        @unknown default:
            permission = .denied
        }
    }

    // MARK: Engine

    /// Start capture. `prepare` runs on the audio queue with the hardware sample
    /// rate *before* the tap is installed (so the analyzer is reset before any
    /// samples flow); `onStarted` runs on the main queue once audio is live.
    func start(prepare: @escaping (Double) -> Void, onStarted: @escaping () -> Void) {
        engineQueue.async { [weak self] in
            guard let self, !self.running else { return }
            let session = AVAudioSession.sharedInstance()
            do {
                try session.setCategory(.record, mode: .measurement, options: [])
                try session.setActive(true)
            } catch {
                print("AudioCapture: session error \(error)")
                return
            }

            let input = self.engine.inputNode
            let format = input.outputFormat(forBus: 0)

            // Reset the analyzer before samples start flowing — no tap yet, so
            // there's no race with process().
            prepare(format.sampleRate)

#if DEBUG && DIAGNOSTIC_RECORDING
            // Diagnostic builds only: a raw WAV alongside the live analysis, so
            // the exact audio can be replayed and the DSP tuned offline. Written
            // to Documents (UIFileSharingEnabled) so it can be pulled off the
            // device via the Files app / Finder / `xcrun devicectl`.
            self.recordFile = nil
            self.recordURL = nil
            let ts = Int(Date().timeIntervalSince1970)
            let docs = FileManager.default.urls(for: .documentDirectory,
                                                in: .userDomainMask)[0]
            // A fresh container has no Documents directory — iOS reports the
            // path but only creates it on demand — and AVAudioFile won't make
            // one, it just throws. Recordings vanished silently until this.
            try? FileManager.default.createDirectory(at: docs,
                                                     withIntermediateDirectories: true)
            let url = docs.appendingPathComponent("watch-\(ts).wav")
            do {
                self.recordFile = try AVAudioFile(forWriting: url,
                                                  settings: format.settings)
                self.recordURL = url
            } catch {
                print("AudioCapture: could not open recording \(error)")
            }
#endif

            input.installTap(onBus: 0, bufferSize: 4096, format: format) { [weak self] buffer, _ in
                self?.handle(buffer)
            }
            self.engine.prepare()
            do {
                try self.engine.start()
                self.running = true
            } catch {
                print("AudioCapture: engine start error \(error)")
                input.removeTap(onBus: 0)
                return
            }
            DispatchQueue.main.async { onStarted() }
        }
    }

    func stop() {
        engineQueue.async { [weak self] in
            guard let self, self.running else { return }
            self.engine.inputNode.removeTap(onBus: 0)
            self.engine.stop()
            self.running = false
            try? AVAudioSession.sharedInstance().setActive(false)
#if DEBUG && DIAGNOSTIC_RECORDING
            // Releasing the AVAudioFile finalizes the WAV header on disk.
            let url = self.recordURL
            self.recordFile = nil
            self.recordURL = nil
#endif
            DispatchQueue.main.async {
                self.inputLevel = 0
#if DEBUG && DIAGNOSTIC_RECORDING
                self.lastRecordingURL = url
#endif
            }
        }
    }

    private func handle(_ buffer: AVAudioPCMBuffer) {
        guard let channel = buffer.floatChannelData?[0] else { return }
        let count = Int(buffer.frameLength)
        guard count > 0 else { return }

#if DEBUG && DIAGNOSTIC_RECORDING
        if let file = recordFile { try? file.write(from: buffer) }
#endif

        onBuffer?(channel, count)

        // RMS level for the meter (log-scaled to feel responsive).
        var sum: Float = 0
        for i in 0..<count { let s = channel[i]; sum += s * s }
        let rms = sqrtf(sum / Float(count))
        let db = 20 * log10f(max(rms, 1e-7))
        let level = max(0, min(1, (db + 60) / 60))   // -60 dB → 0, 0 dB → 1
        DispatchQueue.main.async { [weak self] in
            guard let self else { return }
            // Simple attack/decay smoothing.
            self.inputLevel = level > self.inputLevel
                ? level
                : self.inputLevel * 0.8 + level * 0.2
        }
    }
}
