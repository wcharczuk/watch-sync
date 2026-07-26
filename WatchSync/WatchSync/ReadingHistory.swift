import Foundation
import SwiftUI

/// A saved measurement.
///
/// Everything needed to interpret the reading later is stored with it: the rate
/// is meaningless without its ±, and comparing readings across time only works if
/// you know the position the watch was in — positional variation is often larger
/// than the drift you're trying to track.
struct Reading: Codable, Identifiable {
    var id = UUID()
    var date = Date()
    /// User's label for the watch. Empty means unassigned.
    var watch: String = ""
    /// Dial up, crown down, and so on. Empty means unrecorded.
    var position: String = ""

    var rateSecondsPerDay: Double
    var uncertainty: Double
    var beatsPerHour: Int
    var beatErrorMs: Double?
    var jitterMs: Double
    var detectionRate: Double
    var bandLowHz: Double
    var bandHighHz: Double
    var elapsedSeconds: Double
    /// Acoustic signature of the tick, for recognising the watch on later runs.
    var signature: [Double] = []

    static let positions = ["Dial up", "Dial down", "Crown down",
                            "Crown up", "Crown left", "Worn"]
}

/// Local store of saved readings.
///
/// Kept in Application Support rather than Documents: these are the app's own
/// records, not files the user should have to see or tidy up in the Files app.
@MainActor
final class ReadingStore: ObservableObject {
    @Published private(set) var readings: [Reading] = []

    private let url: URL = {
        let base = FileManager.default.urls(for: .applicationSupportDirectory,
                                            in: .userDomainMask)[0]
        try? FileManager.default.createDirectory(at: base, withIntermediateDirectories: true)
        return base.appendingPathComponent("readings.json")
    }()

    init() { load() }

    /// Watch names seen so far, most recently used first — so the common case
    /// (the watch you just measured yesterday) is the first thing offered.
    var knownWatches: [String] {
        var seen: [String] = []
        for r in readings.sorted(by: { $0.date > $1.date })
        where !r.watch.isEmpty && !seen.contains(r.watch) {
            seen.append(r.watch)
        }
        return seen
    }

    func readings(for watch: String) -> [Reading] {
        readings.filter { $0.watch == watch }.sorted { $0.date < $1.date }
    }

    func add(_ reading: Reading) {
        readings.append(reading)
        save()
    }

    func update(_ reading: Reading) {
        guard let i = readings.firstIndex(where: { $0.id == reading.id }) else { return }
        readings[i] = reading
        save()
    }

    func delete(_ reading: Reading) {
        readings.removeAll { $0.id == reading.id }
        save()
    }

    /// Best guess at which known watch a signature belongs to.
    ///
    /// Deliberately a suggestion, never an assignment. Measured across recorded
    /// sessions, two readings of the same watch score about 0.5–0.7 — the tick
    /// shape moves with contact and position — so a confident-looking threshold
    /// would be false confidence. The beat rate is used as a hard filter first,
    /// since a 21600 movement is never a 28800 one.
    func suggestWatch(signature: [Double], beatsPerHour: Int) -> (watch: String, score: Double)? {
        guard signature.count > 8 else { return nil }
        var best: (String, Double)?
        for name in knownWatches {
            var top = -1.0
            for r in readings(for: name)
            where r.beatsPerHour == beatsPerHour && r.signature.count == signature.count {
                var dot = 0.0
                for i in 0..<signature.count { dot += signature[i] * r.signature[i] }
                top = max(top, dot)
            }
            if top > (best?.1 ?? 0.35) { best = (name, top) }
        }
        return best.map { (watch: $0.0, score: $0.1) }
    }

    private func load() {
        guard let data = try? Data(contentsOf: url),
              let decoded = try? JSONDecoder().decode([Reading].self, from: data) else { return }
        readings = decoded
    }

    private func save() {
        guard let data = try? JSONEncoder().encode(readings) else { return }
        try? data.write(to: url, options: .atomic)
    }
}

// MARK: - Save sheet

/// Asks which watch this reading belongs to, offering a guess when the tick
/// looks familiar.
struct SaveReadingView: View {
    @ObservedObject var store: ReadingStore
    let result: TimegrapherResult
    @Environment(\.dismiss) private var dismiss

    @State private var watch = ""
    @State private var position = ""
    @State private var suggestion: (watch: String, score: Double)?

    var body: some View {
        NavigationStack {
            Form {
                Section("Reading") {
                    LabeledContent("Rate") {
                        Text(String(format: "%+.1f ± %.1f s/day",
                                    result.rateSecondsPerDay ?? 0, result.uncertainty ?? 0))
                            .font(.system(.body, design: .monospaced))
                    }
                    LabeledContent("Beat rate") { Text("\(result.beatsPerHour) bph") }
                    if let be = result.beatErrorMs {
                        LabeledContent("Beat error") { Text(String(format: "%.1f ms", be)) }
                    }
                }

                Section("Watch") {
                    if let s = suggestion, watch.isEmpty {
                        Button {
                            watch = s.watch
                        } label: {
                            HStack {
                                Image(systemName: "wand.and.stars")
                                VStack(alignment: .leading, spacing: 2) {
                                    Text("Sounds like \(s.watch)")
                                    Text("Tap to use — the tick matches a previous reading")
                                        .font(.caption)
                                        .foregroundColor(.secondary)
                                }
                            }
                        }
                    }
                    TextField("Watch name", text: $watch)
                        .textInputAutocapitalization(.words)
                    if !store.knownWatches.isEmpty {
                        ScrollView(.horizontal, showsIndicators: false) {
                            HStack(spacing: 8) {
                                ForEach(store.knownWatches, id: \.self) { name in
                                    Button(name) { watch = name }
                                        .buttonStyle(.bordered)
                                        .tint(watch == name ? .accentColor : .secondary)
                                }
                            }
                        }
                    }
                }

                Section {
                    Picker("Position", selection: $position) {
                        Text("Not recorded").tag("")
                        ForEach(Reading.positions, id: \.self) { Text($0).tag($0) }
                    }
                } header: {
                    Text("Position")
                } footer: {
                    Text("A mechanical watch runs at different rates in different positions — often by more than it drifts over weeks. Recording it makes readings comparable.")
                }
            }
            .navigationTitle("Save reading")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Cancel") { dismiss() }
                }
                ToolbarItem(placement: .confirmationAction) {
                    Button("Save") { save() }
                        .fontWeight(.semibold)
                        .disabled(watch.trimmingCharacters(in: .whitespaces).isEmpty)
                }
            }
            .onAppear {
                suggestion = store.suggestWatch(signature: result.signature,
                                                beatsPerHour: result.beatsPerHour)
            }
        }
    }

    private func save() {
        guard let rate = result.rateSecondsPerDay, let unc = result.uncertainty else { return }
        store.add(Reading(watch: watch.trimmingCharacters(in: .whitespaces),
                          position: position,
                          rateSecondsPerDay: rate,
                          uncertainty: unc,
                          beatsPerHour: result.beatsPerHour,
                          beatErrorMs: result.beatErrorMs,
                          jitterMs: result.jitterMs,
                          detectionRate: result.detectionRate,
                          bandLowHz: result.bandLowHz,
                          bandHighHz: result.bandHighHz,
                          elapsedSeconds: result.elapsedSeconds,
                          signature: result.signature))
        dismiss()
    }
}

// MARK: - History

struct HistoryView: View {
    @ObservedObject var store: ReadingStore
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            Group {
                if store.readings.isEmpty {
                    ContentUnavailableView("No readings yet",
                                           systemImage: "clock.arrow.circlepath",
                                           description: Text("Save a reading and it will appear here, so you can watch a rate change over weeks rather than guess from one measurement."))
                } else {
                    List {
                        ForEach(store.knownWatches, id: \.self) { name in
                            Section(name) {
                                WatchTrendView(readings: store.readings(for: name))
                                    .frame(height: 90)
                                    .listRowInsets(EdgeInsets(top: 8, leading: 12,
                                                              bottom: 8, trailing: 12))
                                ForEach(store.readings(for: name).reversed()) { r in
                                    ReadingRow(reading: r)
                                }
                                .onDelete { offsets in
                                    let items = store.readings(for: name).reversed().map { $0 }
                                    for i in offsets { store.delete(items[i]) }
                                }
                            }
                        }
                    }
                }
            }
            .navigationTitle("History")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Button("Done") { dismiss() }.fontWeight(.semibold)
                }
            }
        }
    }
}

private struct ReadingRow: View {
    let reading: Reading

    var body: some View {
        HStack {
            VStack(alignment: .leading, spacing: 3) {
                Text(reading.date, format: .dateTime.month().day().hour().minute())
                    .font(.system(size: 14))
                if !reading.position.isEmpty {
                    Text(reading.position)
                        .font(.system(size: 11))
                        .foregroundColor(.secondary)
                }
            }
            Spacer()
            Text(String(format: "%+.1f", reading.rateSecondsPerDay))
                .font(.system(size: 17, weight: .medium, design: .monospaced))
            Text(String(format: "± %.1f", reading.uncertainty))
                .font(.system(size: 12, design: .monospaced))
                .foregroundColor(.secondary)
        }
    }
}

/// Rate over time for one watch, with each reading's ± drawn as a bar.
///
/// The point of keeping history: one reading tells you the rate, a series tells
/// you whether it's *changing* — and the ± bars show whether an apparent change
/// is real or just two measurements disagreeing within their own uncertainty.
private struct WatchTrendView: View {
    let readings: [Reading]

    var body: some View {
        Canvas { context, size in
            guard readings.count >= 1 else { return }
            let inset: CGFloat = 8
            var lo = 0.0, hi = 0.0
            for r in readings {
                lo = min(lo, r.rateSecondsPerDay - r.uncertainty)
                hi = max(hi, r.rateSecondsPerDay + r.uncertainty)
            }
            let pad = max(1.0, (hi - lo) * 0.2)
            lo -= pad; hi += pad

            func pt(_ i: Int, _ value: Double) -> CGPoint {
                let x = readings.count == 1
                    ? size.width / 2
                    : inset + CGFloat(i) / CGFloat(readings.count - 1) * (size.width - 2 * inset)
                return CGPoint(x: x, y: inset + (hi - value) / (hi - lo) * (size.height - 2 * inset))
            }

            var zero = Path()
            zero.move(to: CGPoint(x: 0, y: pt(0, 0).y))
            zero.addLine(to: CGPoint(x: size.width, y: pt(0, 0).y))
            context.stroke(zero, with: .color(.secondary.opacity(0.35)),
                           style: StrokeStyle(lineWidth: 1, dash: [3, 3]))

            if readings.count > 1 {
                var line = Path()
                for (i, r) in readings.enumerated() {
                    let p = pt(i, r.rateSecondsPerDay)
                    if i == 0 { line.move(to: p) } else { line.addLine(to: p) }
                }
                context.stroke(line, with: .color(.accentColor.opacity(0.5)),
                               style: StrokeStyle(lineWidth: 1.5))
            }

            for (i, r) in readings.enumerated() {
                let top = pt(i, r.rateSecondsPerDay + r.uncertainty)
                let bottom = pt(i, r.rateSecondsPerDay - r.uncertainty)
                var bar = Path()
                bar.move(to: top)
                bar.addLine(to: bottom)
                context.stroke(bar, with: .color(.accentColor.opacity(0.35)),
                               style: StrokeStyle(lineWidth: 4, lineCap: .round))
                let p = pt(i, r.rateSecondsPerDay)
                context.fill(Path(ellipseIn: CGRect(x: p.x - 3, y: p.y - 3, width: 6, height: 6)),
                             with: .color(.accentColor))
            }

            for value in [hi, lo] {
                context.draw(Text(String(format: "%+.0f", value))
                    .font(.system(size: 9, design: .monospaced))
                    .foregroundColor(.secondary),
                             at: CGPoint(x: 2, y: pt(0, value).y), anchor: .leading)
            }
        }
    }
}
