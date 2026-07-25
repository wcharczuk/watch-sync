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
/// in builds compiled with the `DIAGNOSTIC_RECORDING` condition (set the
/// `WATCHSYNC_DIAGNOSTIC_RECORDING` build setting to `DIAGNOSTIC_RECORDING`).
/// In a normal build the recording code isn't compiled at all, so a measurement
/// cannot leave anything behind.
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
#if DIAGNOSTIC_RECORDING
    /// URL of the most recently finished raw recording (diagnostic builds only).
    @Published var lastRecordingURL: URL?
#endif

    /// Called on the audio thread with each mono buffer. Keep it cheap.
    var onBuffer: ((UnsafePointer<Float>, Int) -> Void)?

    private let engine = AVAudioEngine()
    private let engineQueue = DispatchQueue(label: "audio.engine", qos: .userInitiated)
    private var running = false

#if DIAGNOSTIC_RECORDING
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

#if DIAGNOSTIC_RECORDING
            // Diagnostic builds only: a raw WAV alongside the live analysis, so
            // the exact audio can be replayed and the DSP tuned offline. Written
            // to Documents (UIFileSharingEnabled) so it can be pulled off the
            // device via the Files app / Finder / `xcrun devicectl`.
            self.recordFile = nil
            self.recordURL = nil
            let ts = Int(Date().timeIntervalSince1970)
            let docs = FileManager.default.urls(for: .documentDirectory,
                                                in: .userDomainMask)[0]
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
#if DIAGNOSTIC_RECORDING
            // Releasing the AVAudioFile finalizes the WAV header on disk.
            let url = self.recordURL
            self.recordFile = nil
            self.recordURL = nil
#endif
            DispatchQueue.main.async {
                self.inputLevel = 0
#if DIAGNOSTIC_RECORDING
                self.lastRecordingURL = url
#endif
            }
        }
    }

    private func handle(_ buffer: AVAudioPCMBuffer) {
        guard let channel = buffer.floatChannelData?[0] else { return }
        let count = Int(buffer.frameLength)
        guard count > 0 else { return }

#if DIAGNOSTIC_RECORDING
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
