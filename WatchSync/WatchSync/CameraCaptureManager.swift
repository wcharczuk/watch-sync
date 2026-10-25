import Foundation
import AVFoundation
import Combine

/// Captures timestamped camera frames for the second-hand reader.
///
/// What matters here is *when* each frame was taken, not when it reached us:
/// delivery lags capture by a variable few tens of milliseconds, and the whole
/// measurement is a comparison against the clock. So every frame is handed over
/// with its own capture time, converted from the sample buffer's presentation
/// timestamp to wall-clock time.
///
/// Nothing is recorded: frames are reduced to two rings of brightness samples
/// as they arrive and then released.
///
/// As with audio, every session call runs on a dedicated queue —
/// `startRunning()` blocks, and must not be on the main thread.
final class CameraCaptureManager: NSObject, ObservableObject {
    enum PermissionState { case unknown, granted, denied }

    @Published var permission: PermissionState = .unknown
    /// False where there is no back camera at all (the simulator).
    @Published var isAvailable = true
    @Published private(set) var zoom: CGFloat = CameraCaptureManager.defaultZoom

    let session = AVCaptureSession()
    /// Frames are delivered here; anything that touches the reader should too.
    let videoQueue = DispatchQueue(label: "camera.frames", qos: .userInitiated)

    /// Called on `videoQueue` with an upright luma plane (pointer, width,
    /// height, bytes per row) and its capture time in Unix seconds.
    var onFrame: ((UnsafePointer<UInt8>, Int, Int, Int, Double) -> Void)?

    /// Enough that a dial fills the guide with the phone a hand's width above
    /// it. The first version used the main camera at 4×, because that lens
    /// can't focus inside 20 cm — which meant holding the phone at arm's length
    /// and steadying a telephoto view. The ultra-wide focuses down to a couple
    /// of centimetres, so the phone can sit close, where it is easy to hold
    /// still and every wobble moves the picture a quarter as far.
    static let defaultZoom: CGFloat = 2.5
    static let zoomRange: ClosedRange<CGFloat> = 1...8

    private let sessionQueue = DispatchQueue(label: "camera.session", qos: .userInitiated)
    private let output = AVCaptureVideoDataOutput()
    private var device: AVCaptureDevice?
    private var configured = false

    // Main thread only.
    private weak var previewLayer: AVCaptureVideoPreviewLayer?
    private var rotationCoordinator: AVCaptureDevice.RotationCoordinator?
    private var rotationObservation: NSKeyValueObservation?

    // MARK: Permission

    func requestPermission() {
        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .authorized: permission = .granted
        case .denied, .restricted: permission = .denied
        case .notDetermined:
            AVCaptureDevice.requestAccess(for: .video) { [weak self] granted in
                DispatchQueue.main.async {
                    self?.permission = granted ? .granted : .denied
                }
            }
        @unknown default:
            permission = .denied
        }
    }

    // MARK: Session

    func start() {
        sessionQueue.async { [weak self] in
            guard let self else { return }
            if !self.configured { self.configure() }
            guard self.configured, !self.session.isRunning else { return }
            self.session.startRunning()
        }
    }

    func stop() {
        sessionQueue.async { [weak self] in
            guard let self, self.session.isRunning else { return }
            self.session.stopRunning()
        }
    }

    func setZoom(_ factor: CGFloat) {
        let wanted = min(max(factor, Self.zoomRange.lowerBound), Self.zoomRange.upperBound)
        zoom = wanted
        sessionQueue.async { [weak self] in
            guard let device = self?.device else { return }
            self?.applyZoom(wanted, to: device)
        }
    }

    // MARK: Rotation

    /// The layer the preview is shown in. Which way up the sensor's picture
    /// goes depends on how the window sits relative to the camera, and that is
    /// no longer one fixed answer per device: a phone that unfolds has its
    /// window turn without the camera moving. The coordinator knows; a
    /// hard-coded 90° only knew about a phone held upright.
    func attach(previewLayer: AVCaptureVideoPreviewLayer) {
        guard self.previewLayer !== previewLayer else { return }
        self.previewLayer = previewLayer
        sessionQueue.async { [weak self] in
            guard let device = self?.device else { return }
            DispatchQueue.main.async { self?.followRotation(of: device) }
        }
    }

    private func followRotation(of device: AVCaptureDevice) {
        guard let previewLayer, rotationCoordinator == nil else { return }
        let coordinator = AVCaptureDevice.RotationCoordinator(device: device, previewLayer: previewLayer)
        rotationCoordinator = coordinator
        rotationObservation = coordinator.observe(\.videoRotationAngleForHorizonLevelPreview,
                                                  options: [.initial, .new]) { [weak self] coordinator, _ in
            let angle = coordinator.videoRotationAngleForHorizonLevelPreview
            DispatchQueue.main.async { self?.applyRotation(angle) }
        }
    }

    /// The same angle to the preview and to the frames: the guide circle on
    /// screen and the circle the reader samples are then the same circle, with
    /// 12 at the same top.
    private func applyRotation(_ angle: CGFloat) {
        if let connection = previewLayer?.connection, connection.isVideoRotationAngleSupported(angle) {
            connection.videoRotationAngle = angle
        }
        sessionQueue.async { [weak self] in
            guard let connection = self?.output.connection(with: .video),
                  connection.isVideoRotationAngleSupported(angle),
                  connection.videoRotationAngle != angle else { return }
            connection.videoRotationAngle = angle
        }
    }

    private func applyZoom(_ factor: CGFloat, to device: AVCaptureDevice) {
        guard (try? device.lockForConfiguration()) != nil else { return }
        device.videoZoomFactor = min(max(factor, device.minAvailableVideoZoomFactor),
                                     device.maxAvailableVideoZoomFactor)
        device.unlockForConfiguration()
    }

    private func configure() {
        // The ultra-wide where there is one, for its close focus; the main
        // camera otherwise. Always a single physical lens, never a multi-lens
        // virtual device: those switch lenses on their own when the subject is
        // close, and a lens change mid-measurement is a jump in every angle on
        // the dial.
        let found = AVCaptureDevice.default(.builtInUltraWideCamera, for: .video, position: .back)
            ?? AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back)
        guard let device = found,
              let input = try? AVCaptureDeviceInput(device: device) else {
            DispatchQueue.main.async { self.isAvailable = false }
            return
        }
        session.beginConfiguration()
        session.sessionPreset = .hd1920x1080
        guard session.canAddInput(input), session.canAddOutput(output) else {
            session.commitConfiguration()
            DispatchQueue.main.async { self.isAvailable = false }
            return
        }
        session.addInput(input)

        // Bi-planar so the luma plane can be read in place — the reader only
        // wants brightness, and this way no frame is ever converted or copied.
        output.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String:
                                    kCVPixelFormatType_420YpCbCr8BiPlanarFullRange]
        // A late frame is worthless and a dropped one is harmless: each frame
        // carries its own timestamp, so gaps cost nothing but a sample.
        output.alwaysDiscardsLateVideoFrames = true
        output.setSampleBufferDelegate(self, queue: videoQueue)
        session.addOutput(output)
        session.commitConfiguration()

        if (try? device.lockForConfiguration()) != nil {
            if device.isFocusModeSupported(.continuousAutoFocus) {
                device.focusMode = .continuousAutoFocus
            }
            if device.isAutoFocusRangeRestrictionSupported {
                device.autoFocusRangeRestriction = .near
            }
            device.unlockForConfiguration()
        }
        applyZoom(zoom, to: device)
        self.device = device
        configured = true
        DispatchQueue.main.async { self.followRotation(of: device) }
    }
}

extension CameraCaptureManager: AVCaptureVideoDataOutputSampleBufferDelegate {
    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer,
                       from connection: AVCaptureConnection) {
        guard let onFrame, let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else {
            return
        }

        // Capture time on the wall clock. The presentation timestamp is on the
        // session's clock; move it to the host clock, see how long ago that
        // was, and take that off the current date.
        let host = CMClockGetHostTimeClock()
        var captured = CMSampleBufferGetPresentationTimeStamp(sampleBuffer)
        if let clock = session.synchronizationClock {
            captured = CMSyncConvertTime(captured, from: clock, to: host)
        }
        let age = CMTimeGetSeconds(CMTimeSubtract(CMClockGetTime(host), captured))
        // Anything else means the clocks didn't convert; a wrong timestamp is
        // worse than a missing frame.
        guard age >= 0, age < 1 else { return }
        let time = Date().timeIntervalSince1970 - age

        CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }
        guard let base = CVPixelBufferGetBaseAddressOfPlane(pixelBuffer, 0) else { return }
        onFrame(base.assumingMemoryBound(to: UInt8.self),
                CVPixelBufferGetWidthOfPlane(pixelBuffer, 0),
                CVPixelBufferGetHeightOfPlane(pixelBuffer, 0),
                CVPixelBufferGetBytesPerRowOfPlane(pixelBuffer, 0),
                time)
    }
}
