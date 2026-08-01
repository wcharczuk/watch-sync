#!/usr/bin/env python3
"""
Render app icon candidates.

The old icon was a finely detailed dive-watch dial. Two problems: it read as a
generic clock app, and at the size an icon is actually seen (40-60 px) the tick
marks, lume plots and hands collapsed into grey mush. An icon gets one idea and
has to survive being shrunk.

So each candidate here commits to a single silhouette, drawn at 4x and
downsampled for clean edges. What distinguishes this app from a clock is that it
*measures* a mechanical watch by listening to it — so the vocabulary is the
escapement and the measurement, not the dial.

  ./venv/bin/python ../tools/make_icon.py            # write candidates
  ./venv/bin/python ../tools/make_icon.py --install trace
"""
import argparse
import math
import os
import sys

from PIL import Image, ImageDraw

SIZE = 1024
SS = 4                      # supersample factor
S = SIZE * SS

INK = (14, 16, 20)          # near-black, warmer than pure black
INK_2 = (32, 36, 44)
GREEN = (52, 211, 122)      # the "on rate" green used in the app
BLUE = (10, 132, 255)       # system accent
WHITE = (245, 247, 250)
DIM = (92, 100, 112)


def canvas(bg_top=INK_2, bg_bottom=INK):
    """Square with a subtle vertical gradient — flat black looks dead on a
    home screen, but a gradient keeps it calm."""
    img = Image.new("RGB", (S, S), bg_bottom)
    d = ImageDraw.Draw(img)
    for y in range(S):
        t = y / S
        d.line([(0, y), (S, y)],
               fill=tuple(int(bg_top[i] + (bg_bottom[i] - bg_top[i]) * t) for i in range(3)))
    return img


def finish(img, name, outdir):
    img = img.resize((SIZE, SIZE), Image.LANCZOS)
    path = os.path.join(outdir, f"icon-{name}.png")
    img.save(path)
    print(f"wrote {path}")
    return path


# ---------------------------------------------------------------------------
# drawing helpers
# ---------------------------------------------------------------------------

def radial_canvas(inner, outer, focus=0.42):
    """Radial gradient — light behind the subject, falling off to the corners.

    A flat field is what made the first pass look inert at icon size; a little
    depth behind the mark is most of the difference between "flat clip art" and
    something that looks lit.
    """
    img = Image.new("RGB", (S, S), outer)
    d = ImageDraw.Draw(img)
    cx, cy = S / 2, S * focus
    maxd = math.hypot(S, S) * 0.62
    steps = 260
    for i in range(steps, 0, -1):
        t = i / steps
        r = maxd * t
        col = tuple(int(inner[j] + (outer[j] - inner[j]) * t) for j in range(3))
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=col)
    return img


def glow_arc(img, box, start, end, color, width, glow=3.0, passes=5):
    """Stroke an arc with a soft halo, by compositing widening translucent passes."""
    layer = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    ld = ImageDraw.Draw(layer)
    for i in range(passes, 0, -1):
        t = i / passes
        w = int(width * (1 + glow * t))
        alpha = int(38 * (1 - t) ** 1.6)
        ld.arc(box, start=start, end=end, fill=color + (alpha,), width=max(1, w))
    ld.arc(box, start=start, end=end, fill=color + (255,), width=int(width))
    img.alpha_composite(layer) if img.mode == "RGBA" else img.paste(
        Image.alpha_composite(img.convert("RGBA"), layer).convert("RGB"), (0, 0))


# ---------------------------------------------------------------------------
# candidates
# ---------------------------------------------------------------------------

def icon_rings(outdir):
    """A wristwatch, and the sound coming off it — the app's own radar motif.

    Says "listening to a watch" rather than "telling the time". The crown on the
    case side is what stops the circle reading as a generic dot, and the hand is
    angled rather than vertical so it doesn't look like a power symbol.
    """
    img = canvas()
    d = ImageDraw.Draw(img)
    cx, cy = S // 2, int(S * 0.605)

    for i, r in enumerate([0.30, 0.40, 0.50]):
        radius = S * r
        width = int(S * (0.017 - i * 0.003))
        alpha = [1.0, 0.55, 0.28][i]
        col = tuple(int(INK[j] + (GREEN[j] - INK[j]) * alpha) for j in range(3))
        d.arc([cx - radius, cy - radius, cx + radius, cy + radius],
              start=198, end=342, fill=col, width=width)

    r = S * 0.175
    ring = int(S * 0.032)
    # Crown first, so the case ring is drawn over its inner end.
    d.rounded_rectangle([cx + r - ring * 0.2, cy - S * 0.030,
                         cx + r + S * 0.052, cy + S * 0.030],
                        radius=S * 0.012, fill=WHITE)
    d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=WHITE, width=ring)

    # A hand, angled so it reads as a watch rather than a power glyph.
    a = math.radians(-58)
    d.line([cx, cy, cx + r * 0.70 * math.cos(a), cy + r * 0.70 * math.sin(a)],
           fill=GREEN, width=int(S * 0.028))
    d.ellipse([cx - S * 0.016, cy - S * 0.016, cx + S * 0.016, cy + S * 0.016], fill=WHITE)
    return finish(img, "rings", outdir)


def icon_balance(outdir):
    """A balance wheel — the part of the watch this app is actually listening to.

    Unmistakably a *mechanical movement* rather than a clock face. Spokes are
    kept light so the hairspring, which is what makes it read as an oscillator
    rather than a steering wheel, stays the loudest thing in the mark.
    """
    img = canvas()
    d = ImageDraw.Draw(img)
    cx, cy = S // 2, S // 2

    r = S * 0.355
    d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=WHITE, width=int(S * 0.052))

    for angle in (90, 210, 330):
        a = math.radians(angle)
        d.line([cx, cy, cx + r * math.cos(a), cy + r * math.sin(a)],
               fill=DIM, width=int(S * 0.024))

    pts = []
    turns, steps = 2.4, 500
    for i in range(steps):
        t = i / steps
        a = t * turns * 2 * math.pi - math.pi / 2
        rad = S * 0.045 + t * S * 0.135
        pts.append((cx + rad * math.cos(a), cy + rad * math.sin(a)))
    d.line(pts, fill=GREEN, width=int(S * 0.030), joint="curve")

    d.ellipse([cx - S * 0.028, cy - S * 0.028, cx + S * 0.028, cy + S * 0.028], fill=WHITE)
    return finish(img, "balance", outdir)


def icon_trace(outdir):
    """The timegrapher paper tape: beats marching across, drifting off level.

    The most domain-specific option — anyone who has used a timegrapher reads it
    instantly, and it looks like nothing else on a home screen.
    """
    img = canvas()
    d = ImageDraw.Draw(img)
    rows, cols = 7, 9
    margin = S * 0.17
    span = S - 2 * margin
    dot = S * 0.028
    drift = S * 0.055          # how far the line leans per row

    for row in range(rows):
        y = margin + span * row / (rows - 1)
        offset = (row - (rows - 1) / 2) * drift
        for col in range(cols):
            x = margin + span * col / (cols - 1) + offset
            if x < margin * 0.4 or x > S - margin * 0.4:
                continue
            # The beats themselves are bright; the drift is the whole story.
            d.ellipse([x - dot, y - dot, x + dot, y + dot], fill=GREEN)
    return finish(img, "trace", outdir)


def icon_beat(outdir):
    """One beat: a sharp tick spike, with the rate written as a level line.

    Reads as an instrument — the thing that shows you a measurement.
    """
    img = canvas()
    d = ImageDraw.Draw(img)
    mid = S * 0.52
    left, right = S * 0.14, S * 0.86

    d.line([left, mid, right, mid], fill=DIM, width=int(S * 0.012))

    # Two escapement transients, the second smaller — tick and tock.
    for x0, height in ((S * 0.36, 0.30), (S * 0.63, 0.20)):
        w = S * 0.030
        d.polygon([(x0 - w, mid), (x0, mid - S * height), (x0 + w, mid)], fill=GREEN)
        d.line([x0, mid, x0, mid + S * 0.055], fill=GREEN, width=int(S * 0.022))

    # A ring at the top-left corner hinting at the microphone listening.
    for i, rr in enumerate([0.10, 0.145]):
        radius = S * rr
        col = GREEN if i == 0 else tuple(int(INK[j] + (GREEN[j] - INK[j]) * 0.4) for j in range(3))
        d.arc([left - radius, mid - radius, left + radius, mid + radius],
              start=250, end=110, fill=col, width=int(S * 0.014))
    return finish(img, "beat", outdir)



def _rings_mark(inner_bg, outer_bg, ring_col, case_col, hand_col, name, outdir):
    """The chosen direction, built to carry at 60 px.

    Concentric arcs centred on the watch rather than stacked above it: the sound
    comes *off* the watch, and a centred composition survives shrinking far
    better than a stack. Everything is heavier than the first pass — thin strokes
    are the main thing that disappears when an icon is scaled down.
    """
    img = radial_canvas(inner_bg, outer_bg, focus=0.5).convert("RGBA")
    # Optical centre: the arcs are top-heavy, so the case sits a little low to
    # put the mark's visual weight in the middle of the square.
    cx, cy = S / 2, S * 0.545

    # Sound, radiating off the watch. A gap at the bottom stops the rings
    # reading as a target, and the arcs sweep wide enough to fill the square.
    for i, rr in enumerate([0.255, 0.350, 0.445]):
        radius = S * rr
        width = S * (0.032 - i * 0.006)
        fade = [1.0, 0.60, 0.28][i]
        col = tuple(int(outer_bg[j] + (ring_col[j] - outer_bg[j]) * fade) for j in range(3))
        glow_arc(img, [cx - radius, cy - radius, cx + radius, cy + radius],
                 192, 348, col, width, glow=2.6)

    # The watch: heavy case, crown, one hand. No cast shadow — at icon size a
    # drop shadow behind a ring just reads as a smudge.
    r = S * 0.180
    ring = S * 0.054
    d = ImageDraw.Draw(img)
    d.rounded_rectangle([cx + r - ring * 0.35, cy - S * 0.036,
                         cx + r + S * 0.060, cy + S * 0.036],
                        radius=S * 0.015, fill=case_col + (255,))
    # Fill the dial so the arcs behind can't show through the case.
    d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=outer_bg + (255,))
    d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=case_col + (255,), width=int(ring))

    a = math.radians(-62)
    d.line([cx, cy, cx + r * 0.60 * math.cos(a), cy + r * 0.60 * math.sin(a)],
           fill=hand_col + (255,), width=int(S * 0.036))
    d.ellipse([cx - S * 0.022, cy - S * 0.022, cx + S * 0.022, cy + S * 0.022],
              fill=case_col + (255,))
    return finish(img.convert("RGB"), name, outdir)


def icon_rings2_dark(outdir):
    """Green on graphite — closest to the app's own palette."""
    return _rings_mark((38, 44, 54), (11, 13, 17), GREEN, WHITE, GREEN,
                       "rings2-dark", outdir)


def icon_rings2_vivid(outdir):
    """White on green. Loud, and unmissable in a folder of grey utilities."""
    return _rings_mark((64, 226, 140), (16, 122, 78), WHITE, WHITE, (12, 40, 28),
                       "rings2-vivid", outdir)


def icon_rings2_night(outdir):
    """Green on deep blue — instrument-panel feel, warmer than pure black."""
    return _rings_mark((24, 44, 78), (8, 14, 28), GREEN, WHITE, GREEN,
                       "rings2-night", outdir)


CANDIDATES = {
    "rings2-dark": icon_rings2_dark,
    "rings2-vivid": icon_rings2_vivid,
    "rings2-night": icon_rings2_night,
    "rings": icon_rings,
    "trace": icon_trace,
    "balance": icon_balance,
    "beat": icon_beat,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=".")
    ap.add_argument("--install", choices=sorted(CANDIDATES),
                    help="also copy this candidate into the asset catalogue")
    ap.add_argument("--preview", action="store_true",
                    help="also emit 120px versions, which is how an icon is really seen")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    written = {name: fn(args.outdir) for name, fn in CANDIDATES.items()}

    if args.preview:
        for name, path in written.items():
            small = Image.open(path).resize((120, 120), Image.LANCZOS)
            small.save(os.path.join(args.outdir, f"small-{name}.png"))
        print("wrote 120px previews")

    if args.install:
        here = os.path.dirname(os.path.abspath(__file__))
        dest = os.path.join(here, "..", "WatchSync", "WatchSync", "Assets.xcassets",
                            "AppIcon.appiconset", "AppIcon.png")
        Image.open(written[args.install]).save(os.path.abspath(dest))
        print(f"installed '{args.install}' as the app icon")


if __name__ == "__main__":
    sys.exit(main())
