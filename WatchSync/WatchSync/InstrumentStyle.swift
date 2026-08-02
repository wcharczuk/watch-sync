import SwiftUI

/// The visual language of the app icon, made reusable.
///
/// The icon is rendered rather than flat: a radial "ground" behind everything, a
/// darker radial dial face, a six-stop steel gradient for the bezel, polished
/// hands, and a gloss pass over the crystal. The app was flat SwiftUI defaults,
/// so opening it after tapping the icon felt like arriving somewhere else.
///
/// Every value here is lifted from `tools/icon/svg/watch-accuracy-default.svg`
/// rather than eyeballed, so the two stay in step. If the icon is re-cut, these
/// are the numbers to update.
enum Instrument {

    // MARK: Palette

    static let ink = Color(hex: 0x0B0C15)
    static let amber = Color(hex: 0xF2A54A)        // the measured-rate band
    static let steelLight = Color(hex: 0xE3E5EF)
    static let steelMid = Color(hex: 0x83879C)
    static let steelDark = Color(hex: 0x31344A)
    static let hairline = Color(hex: 0x4A4E63)
    static let dialText = Color(hex: 0xECEEF5)

    // MARK: Gradients (SVG ids in brackets)

    /// Behind everything [ground].
    static let ground = RadialGradient(
        stops: [.init(color: Color(hex: 0x2B2E48), location: 0),
                .init(color: Color(hex: 0x191B2C), location: 0.58),
                .init(color: Color(hex: 0x0B0C15), location: 1)],
        center: .init(x: 0.5, y: 0.34), startRadius: 0, endRadius: 620)

    /// Panels and cards — the dial face [face].
    static let face = RadialGradient(
        stops: [.init(color: Color(hex: 0x31344F), location: 0),
                .init(color: Color(hex: 0x1B1D2E), location: 0.62),
                .init(color: Color(hex: 0x0D0E18), location: 1)],
        center: .init(x: 0.32, y: 0.2), startRadius: 0, endRadius: 420)

    /// Bezel and rails [steel]. The alternating light/dark stops are what make
    /// it read as turned metal instead of a grey fill.
    static let steel = LinearGradient(
        stops: [.init(color: Color(hex: 0x8E91A8), location: 0),
                .init(color: Color(hex: 0xE3E5EF), location: 0.18),
                .init(color: Color(hex: 0x767A92), location: 0.38),
                .init(color: Color(hex: 0xC9CCDB), location: 0.55),
                .init(color: Color(hex: 0x5F6379), location: 0.74),
                .init(color: Color(hex: 0x9EA2B8), location: 1)],
        startPoint: .topLeading, endPoint: .bottomTrailing)

    /// Polished white metal — hands, markers [hand].
    static let polished = LinearGradient(
        stops: [.init(color: Color(hex: 0xF2F3F8), location: 0),
                .init(color: Color(hex: 0xC6C9D8), location: 0.52),
                .init(color: Color(hex: 0x8B8FA4), location: 1)],
        startPoint: .top, endPoint: .bottom)

    /// Light catching the crystal [gloss]. Sits over a surface, never under it.
    static let gloss = LinearGradient(
        stops: [.init(color: .white.opacity(0.30), location: 0),
                .init(color: .white.opacity(0.06), location: 0.60),
                .init(color: .white.opacity(0), location: 1)],
        startPoint: .top, endPoint: .bottom)

    /// The amber band, with the falloff it has on the dial.
    static let band = LinearGradient(
        stops: [.init(color: amber.opacity(0.95), location: 0),
                .init(color: amber.opacity(0.65), location: 1)],
        startPoint: .top, endPoint: .bottom)
}

extension Color {
    init(hex: UInt32) {
        self.init(.sRGB,
                  red: Double((hex >> 16) & 0xFF) / 255,
                  green: Double((hex >> 8) & 0xFF) / 255,
                  blue: Double(hex & 0xFF) / 255,
                  opacity: 1)
    }
}

// MARK: - Surfaces

/// A raised panel: dial face, a hairline that catches light along the top edge,
/// and a shadow beneath. The hairline is what sells it — a gradient fill on its
/// own still reads as flat.
struct InstrumentPanel: ViewModifier {
    var cornerRadius: CGFloat = 16

    func body(content: Content) -> some View {
        content
            .background(
                RoundedRectangle(cornerRadius: cornerRadius, style: .continuous)
                    .fill(Instrument.face)
            )
            .overlay(
                RoundedRectangle(cornerRadius: cornerRadius, style: .continuous)
                    .fill(Instrument.gloss)
                    .blendMode(.plusLighter)
                    .opacity(0.5)
                    .allowsHitTesting(false)
            )
            .overlay(
                RoundedRectangle(cornerRadius: cornerRadius, style: .continuous)
                    .strokeBorder(
                        LinearGradient(
                            stops: [.init(color: .white.opacity(0.22), location: 0),
                                    .init(color: Instrument.hairline.opacity(0.5), location: 0.5),
                                    .init(color: .black.opacity(0.35), location: 1)],
                            startPoint: .top, endPoint: .bottom),
                        lineWidth: 1)
            )
            .shadow(color: .black.opacity(0.45), radius: 10, x: 0, y: 4)
    }
}

/// A recessed groove — the opposite lighting to a panel, for tracks and rails.
struct InstrumentGroove: ViewModifier {
    var cornerRadius: CGFloat = 8

    func body(content: Content) -> some View {
        content
            .background(
                RoundedRectangle(cornerRadius: cornerRadius, style: .continuous)
                    .fill(Color.black.opacity(0.35))
            )
            .overlay(
                RoundedRectangle(cornerRadius: cornerRadius, style: .continuous)
                    .strokeBorder(
                        LinearGradient(
                            stops: [.init(color: .black.opacity(0.6), location: 0),
                                    .init(color: .white.opacity(0.14), location: 1)],
                            startPoint: .top, endPoint: .bottom),
                        lineWidth: 1)
            )
    }
}

extension View {
    func instrumentPanel(cornerRadius: CGFloat = 16) -> some View {
        modifier(InstrumentPanel(cornerRadius: cornerRadius))
    }

    func instrumentGroove(cornerRadius: CGFloat = 8) -> some View {
        modifier(InstrumentGroove(cornerRadius: cornerRadius))
    }
}

/// A polished bead, the marker used on the icon's dial for the true rate.
///
/// Drawn rather than filled flat: a highlight above centre and a darker rim
/// below is the whole trick to making a small circle look like a machined part.
struct Bead: View {
    var size: CGFloat = 16
    var tint: Color = .white

    var body: some View {
        Circle()
            .fill(
                RadialGradient(
                    stops: [.init(color: .white, location: 0),
                            .init(color: tint, location: 0.55),
                            .init(color: tint.opacity(0.75), location: 1)],
                    center: .init(x: 0.35, y: 0.3),
                    startRadius: 0, endRadius: size * 0.75)
            )
            .overlay(Circle().strokeBorder(.black.opacity(0.35), lineWidth: 0.5))
            .frame(width: size, height: size)
            .shadow(color: .black.opacity(0.5), radius: size * 0.18, y: size * 0.08)
    }
}
