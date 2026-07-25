#!/usr/bin/env python3
"""Synthetic watch face renderer for second hand angle detection training data.

V2: Realistic rendering to bridge the domain gap with real camera images.
Adds bezels, straps, glass reflections, perspective transforms, textured
backgrounds, lume dots, subdials, date windows, and watch crowns.
"""

import csv
import math
import os
import random
from io import BytesIO

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "data")
TRAIN_DIR = os.path.join(OUTPUT_DIR, "train")
VAL_DIR = os.path.join(OUTPUT_DIR, "val")
IMG_SIZE = 224
RENDER_SIZE = 448  # 2x for anti-aliasing
SCALE = RENDER_SIZE / IMG_SIZE
NUM_TRAIN = 50000
NUM_VAL = 5000


# ---------- Background generators ----------

def generate_wood_background(size):
    """Generate a wood-grain-like background texture."""
    base_rgb = random.choice([
        [180, 140, 80], [150, 110, 60], [120, 80, 40],
        [100, 65, 30], [160, 130, 90], [140, 100, 55],
        [90, 55, 25], [170, 135, 75], [130, 95, 50],
    ])
    base = np.array(base_rgb, dtype=np.float32)
    arr = np.zeros((size, size, 3), dtype=np.float32)

    # Horizontal grain bands
    freq1 = random.uniform(0.015, 0.06)
    freq2 = random.uniform(0.08, 0.25)
    phase1 = random.uniform(0, 2 * math.pi)
    phase2 = random.uniform(0, 2 * math.pi)
    strength1 = random.uniform(10, 25)
    strength2 = random.uniform(3, 10)

    ys = np.arange(size, dtype=np.float32)
    grain = np.sin(ys * freq1 + phase1) * strength1
    grain += np.sin(ys * freq2 + phase2) * strength2

    for y in range(size):
        arr[y, :] = base + grain[y]

    # Per-pixel noise
    noise = np.random.normal(0, 5, (size, size, 3))
    arr = np.clip(arr + noise, 0, 255)

    # Slight horizontal streaks for grain
    for _ in range(random.randint(3, 8)):
        y = random.randint(0, size - 1)
        h = random.randint(1, 3)
        shift = random.uniform(-12, 12)
        arr[max(0, y):min(size, y + h), :] += shift

    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def generate_fabric_background(size):
    """Generate a fabric/leather-like background texture."""
    base_rgb = random.choice([
        [60, 60, 60], [40, 35, 30], [30, 30, 35],
        [80, 70, 60], [50, 50, 50], [70, 65, 55],
        [45, 40, 35], [55, 50, 45],
    ])
    base = np.array(base_rgb, dtype=np.float32)
    arr = np.tile(base, (size, size, 1))

    # Fine texture noise
    noise = np.random.normal(0, 8, (size, size, 3))
    arr = np.clip(arr + noise, 0, 255)

    # Slight weave pattern
    xs = np.arange(size)
    ys = np.arange(size)
    xg, yg = np.meshgrid(xs, ys)
    weave = (np.sin(xg * 0.5) * np.sin(yg * 0.5) * 3).reshape(size, size, 1)
    arr = np.clip(arr + weave, 0, 255)

    return Image.fromarray(arr.astype(np.uint8))


def generate_background(size):
    """Generate a random background image."""
    bg_type = random.choices(
        ["wood", "fabric", "solid", "gradient"],
        weights=[0.50, 0.15, 0.15, 0.20],
    )[0]

    if bg_type == "wood":
        return generate_wood_background(size)
    elif bg_type == "fabric":
        return generate_fabric_background(size)
    elif bg_type == "solid":
        c = tuple(random.randint(20, 140) for _ in range(3))
        return Image.new("RGB", (size, size), c)
    else:
        img = Image.new("RGB", (size, size))
        draw = ImageDraw.Draw(img)
        c1 = tuple(random.randint(20, 120) for _ in range(3))
        c2 = tuple(random.randint(20, 120) for _ in range(3))
        for y in range(size):
            t = y / size
            c = tuple(int(c1[i] * (1 - t) + c2[i] * t) for i in range(3))
            draw.line([(0, y), (size - 1, y)], fill=c)
        return img


# ---------- Watch component drawing ----------

def pick_metal_color():
    """Pick a random metal finish color."""
    return random.choice([
        (185, 185, 190), (165, 165, 170), (200, 200, 205),  # steel
        (150, 150, 155), (175, 175, 180), (210, 210, 215),  # polished steel
        (175, 165, 140), (195, 185, 160), (165, 155, 130),  # gold-ish
        (110, 110, 115), (130, 130, 135),  # dark steel
    ])


def draw_strap(draw, cx, cy, case_radius, size, metal_color):
    """Draw watch strap/bracelet visible at top and bottom."""
    strap_type = random.choices(
        ["bracelet", "nato", "leather"],
        weights=[0.50, 0.25, 0.25],
    )[0]
    strap_width = case_radius * random.uniform(0.50, 0.70)
    half_w = strap_width / 2

    if strap_type == "bracelet":
        dark = tuple(max(0, c - 35) for c in metal_color)
        light = tuple(min(255, c + 20) for c in metal_color)
        link_h = random.uniform(8, 14) * SCALE

        # Top
        y = cy - case_radius - 3
        while y > -link_h:
            w_var = strap_width * random.uniform(0.96, 1.02)
            draw.rectangle(
                [cx - w_var / 2, y - link_h, cx + w_var / 2, y],
                fill=metal_color, outline=dark, width=1,
            )
            # Center link highlight
            draw.rectangle(
                [cx - w_var / 6, y - link_h + 1, cx + w_var / 6, y - 1],
                fill=light,
            )
            y -= link_h + 1

        # Bottom
        y = cy + case_radius + 3
        while y < size + link_h:
            w_var = strap_width * random.uniform(0.96, 1.02)
            draw.rectangle(
                [cx - w_var / 2, y, cx + w_var / 2, y + link_h],
                fill=metal_color, outline=dark, width=1,
            )
            draw.rectangle(
                [cx - w_var / 6, y + 1, cx + w_var / 6, y + link_h - 1],
                fill=light,
            )
            y += link_h + 1

    elif strap_type == "nato":
        nato_color = random.choice([
            (100, 100, 80), (60, 80, 60), (80, 80, 80),
            (50, 50, 70), (120, 110, 90), (70, 70, 60),
            (85, 90, 75), (65, 65, 60),
        ])
        nato_dark = tuple(max(0, c - 15) for c in nato_color)
        # Top
        draw.rectangle([cx - half_w, 0, cx + half_w, cy - case_radius + 2], fill=nato_color)
        # Edge stitching
        draw.line([(cx - half_w + 2, 0), (cx - half_w + 2, cy - case_radius)],
                  fill=nato_dark, width=1)
        draw.line([(cx + half_w - 2, 0), (cx + half_w - 2, cy - case_radius)],
                  fill=nato_dark, width=1)
        # Bottom
        draw.rectangle([cx - half_w, cy + case_radius - 2, cx + half_w, size], fill=nato_color)
        draw.line([(cx - half_w + 2, cy + case_radius), (cx - half_w + 2, size)],
                  fill=nato_dark, width=1)
        draw.line([(cx + half_w - 2, cy + case_radius), (cx + half_w - 2, size)],
                  fill=nato_dark, width=1)

    else:  # leather
        leather_color = random.choice([
            (60, 40, 25), (80, 50, 30), (40, 30, 20),
            (100, 70, 40), (50, 35, 20), (70, 45, 25),
        ])
        leather_dark = tuple(max(0, c - 20) for c in leather_color)
        draw.rectangle([cx - half_w, 0, cx + half_w, cy - case_radius + 2], fill=leather_color)
        draw.rectangle([cx - half_w, cy + case_radius - 2, cx + half_w, size], fill=leather_color)
        # Stitching
        for side in [-1, 1]:
            x = cx + side * (half_w - 3)
            for sy in range(0, size, int(6 * SCALE)):
                draw.line([(x, sy), (x, sy + int(3 * SCALE))], fill=leather_dark, width=1)


def draw_crown(draw, cx, cy, case_radius, metal_color):
    """Draw watch crown (winding knob) at 3 o'clock position."""
    if random.random() < 0.3:
        return  # Some watches don't show crown prominently

    crown_width = case_radius * random.uniform(0.08, 0.14)
    crown_length = case_radius * random.uniform(0.12, 0.20)
    dark = tuple(max(0, c - 30) for c in metal_color)

    # Crown position (3 o'clock = right side)
    crown_x = cx + case_radius
    crown_y = cy

    draw.rectangle(
        [crown_x - 2, crown_y - crown_width,
         crown_x + crown_length, crown_y + crown_width],
        fill=metal_color, outline=dark, width=1,
    )
    # Knurling lines
    for i in range(3):
        lx = crown_x + crown_length * (0.3 + i * 0.2)
        draw.line(
            [(lx, crown_y - crown_width + 1), (lx, crown_y + crown_width - 1)],
            fill=dark, width=1,
        )


def draw_case_and_bezel(draw, cx, cy, case_radius, bezel_width):
    """Draw the watch case, bezel, and return dial radius."""
    metal_color = pick_metal_color()
    metal_highlight = tuple(min(255, c + 35) for c in metal_color)
    metal_shadow = tuple(max(0, c - 45) for c in metal_color)

    # Outer case ring
    draw.ellipse(
        [cx - case_radius, cy - case_radius, cx + case_radius, cy + case_radius],
        fill=metal_color, outline=metal_shadow, width=2,
    )

    # Highlight arc (top-left) for 3D effect
    for offset in range(3):
        r = case_radius - offset
        draw.arc(
            [cx - r, cy - r, cx + r, cy + r],
            start=200, end=340,
            fill=metal_highlight, width=1,
        )

    bezel_inner = case_radius - bezel_width

    # Bezel type
    bezel_type = random.choices(
        ["numbered", "coin_edge", "plain", "ticks"],
        weights=[0.30, 0.25, 0.25, 0.20],
    )[0]

    bezel_color = random.choice([
        (20, 20, 25), (30, 30, 35), (15, 15, 20),  # black
        metal_color,  # matching case
        (35, 35, 75), (25, 25, 55),  # dark blue
    ])

    # Fill bezel ring
    draw.ellipse(
        [cx - case_radius + 3, cy - case_radius + 3,
         cx + case_radius - 3, cy + case_radius - 3],
        fill=bezel_color,
    )

    text_color = (220, 220, 220) if sum(bezel_color) < 350 else (30, 30, 30)

    if bezel_type == "numbered":
        # Dive/GMT style numbered bezel
        num_r = (case_radius + bezel_inner) / 2
        for i in range(0, 24, 2):
            angle_rad = math.radians(i * 15 - 90)
            tx = cx + num_r * math.cos(angle_rad)
            ty = cy + num_r * math.sin(angle_rad)
            # Draw number as small rectangles (simplified text)
            digit_str = str(i)
            char_w = bezel_width * 0.22
            total_w = len(digit_str) * char_w
            for j, ch in enumerate(digit_str):
                dx = tx - total_w / 2 + j * char_w + char_w / 2
                draw.rectangle(
                    [dx - char_w / 2.5, ty - bezel_width * 0.2,
                     dx + char_w / 2.5, ty + bezel_width * 0.2],
                    fill=text_color,
                )
        # Pip at 12
        pip_r = bezel_width * 0.15
        pip_y = cy - (case_radius + bezel_inner) / 2
        draw.polygon(
            [(cx, pip_y - pip_r * 1.5), (cx - pip_r, pip_y + pip_r), (cx + pip_r, pip_y + pip_r)],
            fill=text_color,
        )

    elif bezel_type == "coin_edge":
        for i in range(120):
            angle_rad = math.radians(i * 3)
            cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
            r1 = case_radius - 2
            r2 = case_radius - bezel_width * 0.35
            x1, y1 = cx + r1 * cos_a, cy + r1 * sin_a
            x2, y2 = cx + r2 * cos_a, cy + r2 * sin_a
            line_color = metal_highlight if i % 2 == 0 else metal_shadow
            draw.line([(x1, y1), (x2, y2)], fill=line_color, width=1)

    elif bezel_type == "ticks":
        for i in range(60):
            angle_rad = math.radians(i * 6 - 90)
            cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
            r1 = case_radius - 3
            r2 = bezel_inner + 3 if i % 5 == 0 else bezel_inner + bezel_width * 0.45
            x1, y1 = cx + r1 * cos_a, cy + r1 * sin_a
            x2, y2 = cx + r2 * cos_a, cy + r2 * sin_a
            w = 2 if i % 5 == 0 else 1
            draw.line([(x1, y1), (x2, y2)], fill=text_color, width=w)

    # else: plain — just the filled bezel color

    return bezel_inner, metal_color


def draw_dial(draw, cx, cy, dial_radius, dial_color):
    """Draw the dial (watch face background)."""
    draw.ellipse(
        [cx - dial_radius, cy - dial_radius, cx + dial_radius, cy + dial_radius],
        fill=dial_color,
    )

    # Optional subtle chapter ring (inner rehaut)
    if random.random() < 0.5:
        rehaut_r = dial_radius * 0.96
        rehaut_color = tuple(
            max(0, min(255, c + random.randint(-20, 20))) for c in dial_color
        )
        draw.ellipse(
            [cx - rehaut_r, cy - rehaut_r, cx + rehaut_r, cy + rehaut_r],
            outline=rehaut_color, width=max(1, int(dial_radius * 0.03)),
        )


def draw_subdials(draw, cx, cy, dial_radius, dial_color, marker_color):
    """Draw optional subdials (like chronograph watches)."""
    if random.random() < 0.65:
        return  # Most watches don't have subdials

    num_subdials = random.choice([2, 3])
    subdial_r = dial_radius * random.uniform(0.14, 0.20)

    # Common subdial positions: 3, 6, 9 o'clock or 3, 9 o'clock
    if num_subdials == 3:
        positions = [90, 180, 270]  # 3, 6, 9 o'clock
    else:
        positions = [90, 270]  # 3, 9 o'clock

    subdial_bg = tuple(
        max(0, min(255, c + random.choice([-30, -20, 20, 30]))) for c in dial_color
    )

    for pos_deg in positions:
        angle_rad = math.radians(pos_deg - 90)
        dist = dial_radius * random.uniform(0.42, 0.55)
        sx = cx + dist * math.cos(angle_rad)
        sy = cy + dist * math.sin(angle_rad)

        draw.ellipse(
            [sx - subdial_r, sy - subdial_r, sx + subdial_r, sy + subdial_r],
            fill=subdial_bg, outline=marker_color, width=1,
        )

        # Small hand in subdial (random angle)
        hand_angle = random.uniform(0, 360)
        hand_len = subdial_r * 0.7
        angle_rad2 = math.radians(hand_angle - 90)
        hx = sx + hand_len * math.cos(angle_rad2)
        hy = sy + hand_len * math.sin(angle_rad2)
        draw.line([(sx, sy), (hx, hy)], fill=marker_color, width=1)


def draw_date_window(draw, cx, cy, dial_radius, dial_color):
    """Draw optional date window."""
    if random.random() < 0.55:
        return

    # Common position: 3 o'clock
    positions = [(0.75, 0), (0, 0.75), (-0.75, 0), (0, -0.55)]
    dx_frac, dy_frac = random.choice(positions)
    wx = cx + dial_radius * dx_frac
    wy = cy + dial_radius * dy_frac

    win_w = dial_radius * random.uniform(0.10, 0.16)
    win_h = dial_radius * random.uniform(0.08, 0.14)

    # White/cream window
    win_bg = random.choice([(245, 245, 245), (250, 245, 235), (255, 255, 255)])
    draw.rectangle(
        [wx - win_w, wy - win_h, wx + win_w, wy + win_h],
        fill=win_bg, outline=(100, 100, 100), width=1,
    )

    # Number
    num_color = (30, 30, 30)
    # Simple digit representation
    draw.rectangle(
        [wx - win_w * 0.4, wy - win_h * 0.5, wx + win_w * 0.4, wy + win_h * 0.5],
        fill=num_color,
    )


def draw_markers(draw, cx, cy, radius, style, color):
    """Draw hour/minute markers in various styles."""
    is_dark_dial = sum(color) < 384  # Used for lume color choice

    if style == "none":
        return

    if style in ("batons", "mixed", "lume_batons"):
        for h in range(12):
            angle_rad = math.radians(h * 30 - 90)
            cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
            inner = radius * 0.80
            outer = radius * 0.93
            w = random.uniform(2.0, 4.5) * SCALE

            if style == "lume_batons":
                # Wider, filled batons with lume (like real dive watches)
                w = random.uniform(3.0, 5.5) * SCALE
                inner = radius * 0.78

            x1 = cx + inner * cos_a
            y1 = cy + inner * sin_a
            x2 = cx + outer * cos_a
            y2 = cy + outer * sin_a

            # Draw as thick line
            draw.line([(x1, y1), (x2, y2)], fill=color, width=max(1, int(w)))

    if style == "lume_dots" or style == "dots":
        for h in range(12):
            angle_rad = math.radians(h * 30 - 90)
            r_dot = random.uniform(3, 6) * SCALE
            pos = radius * 0.85
            dx = cx + pos * math.cos(angle_rad)
            dy = cy + pos * math.sin(angle_rad)

            if style == "lume_dots":
                # Lume (phosphorescent) dots - slightly glowing appearance
                lume_color = random.choice([
                    (220, 230, 210), (200, 215, 195), (230, 240, 220),
                ]) if is_dark_dial else color
                draw.ellipse(
                    [dx - r_dot, dy - r_dot, dx + r_dot, dy + r_dot],
                    fill=lume_color,
                )
                # Subtle glow ring
                draw.ellipse(
                    [dx - r_dot - 1, dy - r_dot - 1, dx + r_dot + 1, dy + r_dot + 1],
                    outline=lume_color,
                )
            else:
                draw.ellipse(
                    [dx - r_dot, dy - r_dot, dx + r_dot, dy + r_dot],
                    fill=color,
                )

    if style in ("ticks", "mixed"):
        for m in range(60):
            if style == "mixed" and m % 5 == 0:
                continue
            angle_rad = math.radians(m * 6 - 90)
            cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
            inner = radius * 0.91
            outer = radius * 0.96
            x1 = cx + inner * cos_a
            y1 = cy + inner * sin_a
            x2 = cx + outer * cos_a
            y2 = cy + outer * sin_a
            draw.line([(x1, y1), (x2, y2)], fill=color, width=1)

    # Draw minute track (fine ticks at edge) for some styles
    if style not in ("none", "ticks") and random.random() < 0.6:
        track_color = tuple(
            max(0, min(255, c + random.choice([-40, -30, 30, 40]))) for c in color
        )
        for m in range(60):
            angle_rad = math.radians(m * 6 - 90)
            cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
            inner = radius * 0.95
            outer = radius * 0.98
            x1, y1 = cx + inner * cos_a, cy + inner * sin_a
            x2, y2 = cx + outer * cos_a, cy + outer * sin_a
            draw.line([(x1, y1), (x2, y2)], fill=track_color, width=1)


def draw_hand(draw, cx, cy, angle_deg, length, width, color, hand_type="simple"):
    """Draw a clock hand."""
    angle_rad = math.radians(angle_deg - 90)  # 0 deg = 12 o'clock
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)

    ex = cx + length * cos_a
    ey = cy + length * sin_a

    hw = width / 2
    px = -sin_a * hw
    py = cos_a * hw

    if hand_type == "arrow" and width > 2:
        # Arrow-shaped hand (wider near center, pointed tip)
        mid_x = cx + length * 0.6 * cos_a
        mid_y = cy + length * 0.6 * sin_a
        pw = width * 0.8
        ppx = -sin_a * pw
        ppy = cos_a * pw

        polygon = [
            (cx - px * 0.5, cy - py * 0.5),
            (cx + px * 0.5, cy + py * 0.5),
            (mid_x + ppx, mid_y + ppy),
            (ex, ey),
            (mid_x - ppx, mid_y - ppy),
        ]
        draw.polygon(polygon, fill=color)
    elif hand_type == "lume":
        # Hand with lume fill (rectangular with lighter center)
        polygon = [
            (cx - px, cy - py),
            (cx + px, cy + py),
            (ex + px * 0.3, ey + py * 0.3),
            (ex - px * 0.3, ey - py * 0.3),
        ]
        draw.polygon(polygon, fill=color)
        # Lume stripe
        lume = (220, 230, 210)
        inner_hw = hw * 0.5
        ipx = -sin_a * inner_hw
        ipy = cos_a * inner_hw
        lume_end_x = cx + length * 0.85 * cos_a
        lume_end_y = cy + length * 0.85 * sin_a
        lume_start_x = cx + length * 0.15 * cos_a
        lume_start_y = cy + length * 0.15 * sin_a
        lp = [
            (lume_start_x - ipx, lume_start_y - ipy),
            (lume_start_x + ipx, lume_start_y + ipy),
            (lume_end_x + ipx, lume_end_y + ipy),
            (lume_end_x - ipx, lume_end_y - ipy),
        ]
        draw.polygon(lp, fill=lume)
    else:
        # Simple rectangular hand
        polygon = [
            (cx - px, cy - py),
            (cx + px, cy + py),
            (ex + px, ey + py),
            (ex - px, ey - py),
        ]
        draw.polygon(polygon, fill=color)


def draw_brand_text(draw, cx, cy, dial_radius, text_color):
    """Draw simplified brand text on the dial."""
    if random.random() < 0.4:
        return

    # Brand text area (below 12 o'clock)
    text_y = cy - dial_radius * random.uniform(0.30, 0.45)
    text_w = dial_radius * random.uniform(0.25, 0.45)
    text_h = dial_radius * random.uniform(0.03, 0.06)

    # Draw as thin rectangles (simulating text without fonts)
    num_chars = random.randint(3, 8)
    char_gap = text_w * 2 / num_chars
    start_x = cx - text_w
    for i in range(num_chars):
        cw = char_gap * random.uniform(0.3, 0.7)
        ch = text_h * random.uniform(0.7, 1.0)
        x = start_x + i * char_gap + char_gap * 0.2
        draw.rectangle(
            [x, text_y - ch, x + cw, text_y + ch],
            fill=text_color,
        )

    # Optional second line (model name)
    if random.random() < 0.5:
        text_y2 = text_y + dial_radius * 0.08
        text_h2 = text_h * 0.7
        num_chars2 = random.randint(4, 10)
        char_gap2 = text_w * 1.5 / num_chars2
        start_x2 = cx - text_w * 0.75
        for i in range(num_chars2):
            cw = char_gap2 * random.uniform(0.3, 0.6)
            ch = text_h2 * random.uniform(0.7, 1.0)
            x = start_x2 + i * char_gap2 + char_gap2 * 0.2
            draw.rectangle(
                [x, text_y2 - ch, x + cw, text_y2 + ch],
                fill=text_color,
            )


def add_glass_reflection(img, cx, cy, dial_radius):
    """Add glass crystal reflection overlay."""
    if random.random() < 0.3:
        return img

    size = img.size[0]
    overlay = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    od = ImageDraw.Draw(overlay)

    # Large diagonal gradient reflection
    reflection_type = random.choice(["gradient", "spot", "edge"])

    if reflection_type == "gradient":
        # Diagonal gradient across the dial
        angle = random.uniform(20, 70)
        alpha_max = random.randint(25, 70)

        for i in range(int(dial_radius * 1.5)):
            t = i / (dial_radius * 1.5)
            alpha = int(alpha_max * (1 - abs(2 * t - 1)) ** 2)
            if alpha < 2:
                continue

            angle_rad = math.radians(angle)
            dx = math.cos(angle_rad) * dial_radius * (t - 0.5) * 2
            dy = math.sin(angle_rad) * dial_radius * (t - 0.5) * 2

            # Draw line perpendicular to angle
            perp = angle_rad + math.pi / 2
            length = dial_radius * 0.8
            x1 = cx + dx - math.cos(perp) * length
            y1 = cy + dy - math.sin(perp) * length
            x2 = cx + dx + math.cos(perp) * length
            y2 = cy + dy + math.sin(perp) * length
            od.line([(x1, y1), (x2, y2)], fill=(255, 255, 255, alpha), width=2)

    elif reflection_type == "spot":
        # Circular spot reflection
        gx = cx + random.uniform(-dial_radius * 0.3, dial_radius * 0.3)
        gy = cy + random.uniform(-dial_radius * 0.3, dial_radius * 0.3)
        gr = random.uniform(dial_radius * 0.15, dial_radius * 0.35)
        alpha = random.randint(30, 80)
        od.ellipse([gx - gr, gy - gr, gx + gr, gy + gr], fill=(255, 255, 255, alpha))

    else:
        # Edge reflection (like light catching the crystal edge)
        start_angle = random.uniform(0, 360)
        span = random.uniform(60, 140)
        alpha = random.randint(20, 50)
        r = dial_radius * random.uniform(0.85, 1.0)
        od.arc(
            [cx - r, cy - r, cx + r, cy + r],
            start=start_angle, end=start_angle + span,
            fill=(255, 255, 255, alpha), width=max(1, int(dial_radius * 0.08)),
        )

    # Blur the reflection for realism
    overlay = overlay.filter(ImageFilter.GaussianBlur(radius=dial_radius * 0.05))

    img = img.convert("RGBA")
    img = Image.alpha_composite(img, overlay)
    return img.convert("RGB")


def apply_perspective(img, strength=0.0004):
    """Apply mild perspective transform to simulate camera viewing angle."""
    if random.random() < 0.4:
        return img

    w, h = img.size
    s = strength * random.uniform(0.5, 1.0)

    # Random perspective: slight tilt
    g = random.uniform(-s, s)  # perspective x
    p = random.uniform(-s, s)  # perspective y

    coeffs = (
        1 + random.uniform(-0.015, 0.015),  # a: x scale
        random.uniform(-0.015, 0.015),       # b: x shear
        random.uniform(-2, 2),               # c: x translate
        random.uniform(-0.015, 0.015),       # d: y shear
        1 + random.uniform(-0.015, 0.015),   # e: y scale
        random.uniform(-2, 2),               # f: y translate
        g,                                    # g: perspective x
        p,                                    # h: perspective y
    )

    return img.transform(img.size, Image.PERSPECTIVE, coeffs, Image.BILINEAR,
                         fillcolor=(random.randint(20, 60),) * 3)


def random_dial_color():
    """Pick a random dial color weighted towards what real watches look like."""
    return random.choice([
        (20, 20, 25), (25, 25, 30), (15, 15, 20), (30, 30, 35),  # black (most common)
        (35, 35, 65), (25, 30, 55), (20, 25, 50),  # dark navy
        (250, 250, 248), (245, 242, 235), (240, 238, 232),  # white/cream
        (80, 80, 85), (100, 100, 105), (65, 65, 70),  # grey
        (180, 178, 172), (195, 193, 188),  # silver/light grey
        (40, 55, 40), (30, 45, 35),  # dark green
    ])


def random_marker_color(dial_color):
    """Pick marker color that contrasts with dial."""
    is_dark = sum(dial_color) < 300
    if is_dark:
        return random.choice([
            (220, 220, 220), (200, 200, 200), (240, 240, 240),
            (180, 180, 180), (210, 215, 205),  # lume-ish
        ])
    else:
        return random.choice([
            (30, 30, 30), (50, 50, 50), (20, 20, 20),
            (60, 55, 50), (40, 40, 45),
        ])


def render_watch_face(second_hand_angle_deg):
    """Render a synthetic watch face. Returns 224x224 PIL Image."""
    img = generate_background(RENDER_SIZE)
    draw = ImageDraw.Draw(img)

    # Watch geometry
    cx = RENDER_SIZE / 2 + random.uniform(-10, 10) * SCALE
    cy = RENDER_SIZE / 2 + random.uniform(-10, 10) * SCALE

    # Case radius — watch fills 60-85% of frame (matching real camera framing)
    case_radius = RENDER_SIZE * random.uniform(0.30, 0.42)
    bezel_width = case_radius * random.uniform(0.07, 0.16)
    dial_radius_approx = case_radius - bezel_width

    # Colors
    dial_color = random_dial_color()
    marker_color = random_marker_color(dial_color)
    metal_color = pick_metal_color()

    # Draw strap behind case
    draw_strap(draw, cx, cy, case_radius, RENDER_SIZE, metal_color)

    # Draw crown
    draw_crown(draw, cx, cy, case_radius, metal_color)

    # Draw case and bezel
    dial_radius, _ = draw_case_and_bezel(draw, cx, cy, case_radius, bezel_width)

    # Draw dial
    draw_dial(draw, cx, cy, dial_radius, dial_color)

    # Draw subdials
    draw_subdials(draw, cx, cy, dial_radius, dial_color, marker_color)

    # Draw date window
    draw_date_window(draw, cx, cy, dial_radius, dial_color)

    # Draw brand text
    draw_brand_text(draw, cx, cy, dial_radius, marker_color)

    # Draw markers
    marker_style = random.choices(
        ["lume_batons", "lume_dots", "batons", "dots", "ticks", "mixed", "none"],
        weights=[0.25, 0.25, 0.15, 0.10, 0.10, 0.10, 0.05],
    )[0]
    draw_markers(draw, cx, cy, dial_radius, marker_style, marker_color)

    # Hand colors
    is_dark_dial = sum(dial_color) < 300
    if is_dark_dial:
        hand_color = random.choice([
            (220, 220, 220), (200, 200, 200), (240, 240, 240),
            (180, 180, 185), (160, 160, 165),
        ])
    else:
        hand_color = random.choice([
            (25, 25, 25), (40, 40, 40), (50, 50, 55),
            (20, 20, 20), (60, 60, 60),
        ])

    hand_style = random.choice(["simple", "arrow", "lume"])

    # Hour hand
    hour_angle = random.uniform(0, 360)
    hour_length = dial_radius * random.uniform(0.38, 0.52)
    hour_width = random.uniform(4, 7) * SCALE
    draw_hand(draw, cx, cy, hour_angle, hour_length, hour_width, hand_color, hand_style)

    # Minute hand
    minute_angle = random.uniform(0, 360)
    minute_length = dial_radius * random.uniform(0.58, 0.78)
    minute_width = random.uniform(2.5, 4.5) * SCALE
    draw_hand(draw, cx, cy, minute_angle, minute_length, minute_width, hand_color, hand_style)

    # Second hand (target) — always thinnest, with more aggressive variation
    second_length = dial_radius * random.uniform(0.65, 0.95)
    second_width = random.uniform(0.5, 2.5) * SCALE  # Wider range: very thin to medium
    second_color = random_second_hand_color()
    # Occasionally make the hand semi-transparent (faint)
    if random.random() < 0.15:
        # Blend hand color toward dial for a faint appearance
        blend = random.uniform(0.3, 0.6)
        second_color = tuple(
            int(second_color[i] * blend + dial_color[i] * (1 - blend))
            for i in range(3)
        )
    draw_hand(draw, cx, cy, second_hand_angle_deg, second_length, second_width, second_color)

    # Optional counterweight tail
    if random.random() < 0.5:
        tail_length = dial_radius * random.uniform(0.15, 0.28)
        tail_angle = (second_hand_angle_deg + 180) % 360
        draw_hand(draw, cx, cy, tail_angle, tail_length, second_width * 1.5, second_color)

    # Center pin
    pin_r = random.uniform(2.5, 5) * SCALE
    pin_color = random.choice([second_color, hand_color, metal_color])
    draw.ellipse([cx - pin_r, cy - pin_r, cx + pin_r, cy + pin_r], fill=pin_color)

    # Glass reflection (before perspective)
    img = add_glass_reflection(img, cx, cy, dial_radius)

    # Perspective transform
    img = apply_perspective(img)

    # Downsample to 224x224 with anti-aliasing
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)

    # Post-processing augmentations
    img = apply_augmentations(img)

    return img


def random_second_hand_color():
    """Pick second hand color."""
    r = random.random()
    if r < 0.35:
        return random.choice([(220, 35, 30), (200, 25, 20), (185, 15, 15), (240, 50, 40)])
    elif r < 0.45:
        return random.choice([(35, 65, 200), (25, 55, 185), (45, 80, 220)])
    elif r < 0.55:
        return random.choice([(230, 120, 20), (220, 140, 30), (240, 100, 15)])
    else:
        return random.choice([
            (20, 20, 20), (30, 30, 30), (40, 40, 40), (50, 50, 50),
            (200, 200, 200), (180, 180, 180), (220, 220, 220),
        ])


def apply_motion_blur(img):
    """Apply directional motion blur to simulate camera/hand shake."""
    if random.random() < 0.7:
        return img  # Only apply 30% of the time

    size = random.choice([3, 5, 7])
    angle = random.uniform(0, 180)

    # Create directional kernel
    kernel = np.zeros((size, size), dtype=np.float32)
    center = size // 2
    angle_rad = math.radians(angle)
    dx = math.cos(angle_rad)
    dy = math.sin(angle_rad)

    for i in range(size):
        t = (i - center) / max(center, 1)
        x = int(round(center + t * center * dx))
        y = int(round(center + t * center * dy))
        if 0 <= x < size and 0 <= y < size:
            kernel[y, x] = 1.0

    kernel_sum = kernel.sum()
    if kernel_sum > 0:
        kernel /= kernel_sum

    kernel_img = ImageFilter.Kernel(
        size=(size, size),
        kernel=kernel.flatten().tolist(),
        scale=1,
        offset=0,
    )
    return img.filter(kernel_img)


def apply_augmentations(img):
    """Apply post-render augmentations."""
    # Motion blur (directional)
    img = apply_motion_blur(img)

    # Gaussian blur
    blur_sigma = random.uniform(0, 1.2)
    if blur_sigma > 0.3:
        img = img.filter(ImageFilter.GaussianBlur(radius=blur_sigma))

    arr = np.array(img, dtype=np.float32)

    # Noise
    noise_sigma = random.uniform(0, 12)
    if noise_sigma > 1:
        noise = np.random.normal(0, noise_sigma, arr.shape).astype(np.float32)
        arr = np.clip(arr + noise, 0, 255)

    # Brightness
    brightness = random.uniform(0.65, 1.35)
    arr = np.clip(arr * brightness, 0, 255)

    # Contrast
    contrast = random.uniform(0.75, 1.25)
    mean_val = arr.mean()
    arr = np.clip((arr - mean_val) * contrast + mean_val, 0, 255)

    img = Image.fromarray(arr.astype(np.uint8))

    # JPEG artifacts
    quality = random.randint(55, 95)
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    img = Image.open(buf).convert("RGB")

    return img


def generate_dataset(out_dir, num_images, label_file):
    """Generate a dataset of synthetic watch images with labels."""
    os.makedirs(out_dir, exist_ok=True)

    with open(label_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "angle_degrees", "sin_theta", "cos_theta"])

        for i in range(num_images):
            angle_deg = random.uniform(0, 360)
            angle_rad = math.radians(angle_deg)
            sin_theta = math.sin(angle_rad)
            cos_theta = math.cos(angle_rad)

            img = render_watch_face(angle_deg)
            fname = f"{i:06d}.png"
            img.save(os.path.join(out_dir, fname))

            writer.writerow([fname, f"{angle_deg:.4f}", f"{sin_theta:.6f}", f"{cos_theta:.6f}"])

            if (i + 1) % 1000 == 0:
                print(f"  {i + 1}/{num_images}")


def main():
    print("Generating training data (v2 — realistic rendering)...")
    generate_dataset(TRAIN_DIR, NUM_TRAIN, os.path.join(OUTPUT_DIR, "train_labels.csv"))
    print("Generating validation data...")
    generate_dataset(VAL_DIR, NUM_VAL, os.path.join(OUTPUT_DIR, "val_labels.csv"))
    print("Done!")


if __name__ == "__main__":
    main()
