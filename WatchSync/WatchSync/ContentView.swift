import SwiftUI

enum AppMode: String, CaseIterable {
    case sync = "Sync"
    case accuracy = "Accuracy"
}

struct ContentView: View {
    @State private var mode: AppMode = .sync
    @State private var hashMarkSeconds: Double = 0
    @State private var lastAppliedDelta: Int = 0

    var body: some View {
        VStack(spacing: 0) {
            Picker("Mode", selection: $mode) {
                ForEach(AppMode.allCases, id: \.self) { m in
                    Text(m.rawValue).tag(m)
                }
            }
            .pickerStyle(.segmented)
            .padding(.horizontal)
            .padding(.top, 8)

            switch mode {
            case .sync:
                syncView
            case .accuracy:
                AccuracyView()
            }
        }
    }

    private var syncView: some View {
        TimelineView(.animation(minimumInterval: 0.01)) { timeline in
            let currentTime = timeline.date
            let countdown = calculateCountdown(currentTime: currentTime)
            // Calculate flash opacity:
            // - Full opacity (1.0) during 250ms before reaching hash mark
            // - Fade out from 1.0 to 0.0 during 100ms after passing hash mark
            let fadeOutFrom = 60.0
            let flashOpacity: Double = {
                if countdown <= 0.25 {
                    return 1.0 // 250ms before: full flash
                } else if countdown >= fadeOutFrom {
                    return (countdown - fadeOutFrom) * 10
                } else {
                    return 0.0
                }
            }()

            ZStack {
                // Red flash overlay with fade
                Color.white
                    .ignoresSafeArea()
                    .opacity(flashOpacity)

                VStack(spacing: 40) {
                Spacer()

                AnalogClockView(
                    currentTime: currentTime,
                    hashMarkSeconds: hashMarkSeconds
                )
                .frame(width: 300, height: 300)

                VStack(spacing: 8) {
                    Text("Time until hash mark")
                        .font(.headline)
                        .foregroundColor(.secondary)

                    Text(formatCountdown(countdown))
                        .font(.system(size: 48, weight: .light, design: .monospaced))
                        .foregroundColor(.primary)
                }

                Spacer()

                Text("Drag up/down anywhere to move hash mark")
                    .font(.caption)
                    .foregroundColor(.secondary)
                    .padding(.bottom, 20)
                }
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .contentShape(Rectangle())
        .gesture(
            DragGesture()
                .onChanged { value in
                    // 10 points of drag = 1 second
                    let dragDelta = -value.translation.height
                    let totalSecondsDelta = Int(dragDelta / 10)

                    // Only apply the difference from last update (relative movement)
                    let deltaToApply = totalSecondsDelta - lastAppliedDelta

                    if deltaToApply != 0 {
                        var newPosition = Int(hashMarkSeconds) + deltaToApply

                        // Wrap around 0-59
                        while newPosition < 0 { newPosition += 60 }
                        while newPosition >= 60 { newPosition -= 60 }

                        hashMarkSeconds = Double(newPosition)
                        lastAppliedDelta = totalSecondsDelta
                    }
                }
                .onEnded { _ in
                    lastAppliedDelta = 0
                }
        )
    }

    private func calculateCountdown(currentTime: Date) -> Double {
        let calendar = Calendar.current
        let components = calendar.dateComponents([.second, .nanosecond], from: currentTime)
        let currentSeconds = Double(components.second ?? 0) + Double(components.nanosecond ?? 0) / 1_000_000_000

        var diff = hashMarkSeconds - currentSeconds
        if diff <= 0 {
            diff += 60.0
        }

        return diff
    }

    private func formatCountdown(_ seconds: Double) -> String {
        let totalMilliseconds = Int(seconds * 1000)
        let secs = totalMilliseconds / 1000
        let millis = totalMilliseconds % 1000
        return String(format: "%02d.%03d", secs, millis)
    }
}

struct AnalogClockView: View {
    let currentTime: Date
    let hashMarkSeconds: Double

    var body: some View {
        GeometryReader { geometry in
            let size = min(geometry.size.width, geometry.size.height)
            let radius = size / 2 - 10

            ZStack {
                // Clock face
                Circle()
                    .stroke(Color.primary.opacity(0.3), lineWidth: 2)

                // Hour markers
                ForEach(0..<12) { hour in
                    Rectangle()
                        .fill(Color.primary)
                        .frame(width: hour % 3 == 0 ? 3 : 1.5, height: hour % 3 == 0 ? 15 : 10)
                        .offset(y: -radius + (hour % 3 == 0 ? 7.5 : 5))
                        .rotationEffect(.degrees(Double(hour) * 30))
                }

                // Minute markers
                ForEach(0..<60) { minute in
                    if minute % 5 != 0 {
                        Rectangle()
                            .fill(Color.primary.opacity(0.5))
                            .frame(width: 1, height: 5)
                            .offset(y: -radius + 2.5)
                            .rotationEffect(.degrees(Double(minute) * 6))
                    }
                }

                // 24-hour GMT bezel (0/24 at top, one step every 15°)
                ForEach(0..<24) { h in
                    Rectangle()
                        .fill(Color.blue.opacity(h % 6 == 0 ? 0.9 : 0.4))
                        .frame(width: h % 6 == 0 ? 2 : 1, height: h % 6 == 0 ? 8 : 5)
                        .offset(y: -radius + 20)
                        .rotationEffect(.degrees(Double(h) * 15))
                }
                ForEach(0..<24) { h in
                    let angle = Double(h) * 15 - 90
                    let r = radius - 34
                    Text("\(h)")
                        .font(.system(size: 8, weight: h % 6 == 0 ? .semibold : .regular))
                        .foregroundColor(.blue.opacity(h % 6 == 0 ? 0.9 : 0.6))
                        .offset(
                            x: CGFloat(cos(angle * .pi / 180)) * r,
                            y: CGFloat(sin(angle * .pi / 180)) * r
                        )
                }

                // Red hash mark
                Rectangle()
                    .fill(Color.red)
                    .frame(width: 4, height: 20)
                    .offset(y: -radius + 10)
                    .rotationEffect(.degrees(hashMarkSeconds * 6))

                // Hour hand
                ClockHand(
                    length: radius * 0.5,
                    width: 6,
                    color: .primary
                )
                .rotationEffect(hourAngle)

                // Minute hand
                ClockHand(
                    length: radius * 0.7,
                    width: 4,
                    color: .primary
                )
                .rotationEffect(minuteAngle)

                // GMT hand (UTC, one revolution per 24 hours)
                ClockHand(
                    length: radius * 0.65,
                    width: 4,
                    color: .blue
                )
                .rotationEffect(gmtAngle)

                // Second hand
                ClockHand(
                    length: radius * 0.85,
                    width: 2,
                    color: .orange
                )
                .rotationEffect(secondAngle)

                // Center dot
                Circle()
                    .fill(Color.orange)
                    .frame(width: 12, height: 12)
            }
            .frame(width: size, height: size)
        }
    }

    private var hourAngle: Angle {
        let calendar = Calendar.current
        let hour = calendar.component(.hour, from: currentTime) % 12
        let minute = calendar.component(.minute, from: currentTime)
        let hourDegrees = Double(hour) * 30 + Double(minute) * 0.5
        return .degrees(hourDegrees)
    }

    private var minuteAngle: Angle {
        let calendar = Calendar.current
        let minute = calendar.component(.minute, from: currentTime)
        let second = calendar.component(.second, from: currentTime)
        let minuteDegrees = Double(minute) * 6 + Double(second) * 0.1
        return .degrees(minuteDegrees)
    }

    private var gmtAngle: Angle {
        var calendar = Calendar.current
        calendar.timeZone = TimeZone(identifier: "UTC")!
        let components = calendar.dateComponents([.hour, .minute, .second], from: currentTime)
        let hour = Double(components.hour ?? 0)
        let minute = Double(components.minute ?? 0)
        let second = Double(components.second ?? 0)
        // 360° / 24h = 15° per hour; 0/24h UTC points up (12 o'clock)
        let gmtDegrees = (hour + minute / 60 + second / 3600) * 15
        return .degrees(gmtDegrees)
    }

    private var secondAngle: Angle {
        let calendar = Calendar.current
        let components = calendar.dateComponents([.second, .nanosecond], from: currentTime)
        let second = components.second ?? 0
        let nanosecond = components.nanosecond ?? 0
        let secondDegrees = Double(second) * 6 + Double(nanosecond) / 1_000_000_000 * 6
        return .degrees(secondDegrees)
    }
}

struct ClockHand: View {
    let length: CGFloat
    let width: CGFloat
    let color: Color

    var body: some View {
        RoundedRectangle(cornerRadius: width / 2)
            .fill(color)
            .frame(width: width, height: length)
            .offset(y: -length / 2)
    }
}

#Preview {
    ContentView()
}
