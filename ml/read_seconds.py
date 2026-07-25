#!/usr/bin/env python3
"""Single-frame second-hand reader.

Given a still image of an analog watch, return the angle of the second hand in
degrees clockwise from 12 (= 0°). No temporal tracking, no background model —
each frame is processed independently.

Strategy:
  1. Find watch face (Hough circles).
  2. Build a 1D angular profile in the outer ring (where the second hand tip
     lives but the minute/hour hands usually don't reach). Each angle bin gets
     the peak intensity along a thin radial line.
  3. Subtract a smoothed (low-pass) version of that profile. This kills the
     dial background and slow modulations from lighting.
  4. Find candidate peaks. The second hand is identified by being THIN
     (small FWHM) compared with the minute/hour hands.
  5. Return the chosen angle in degrees and a confidence score.

Run:
  read_seconds.py path/to/frame.jpg [path2 ...]
"""

import os
import sys
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


N_ANGLES = 1440  # 0.25° resolution


# -----------------------------------------------------------------------------
# Watch face detection
# -----------------------------------------------------------------------------

def detect_watch_face(bgr_frame):
    """Find dial (cx, cy, r) in pixels. Returns None if not found."""
    gray = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    scale = 600.0 / max(h, w)
    small = cv2.resize(gray, None, fx=scale, fy=scale,
                       interpolation=cv2.INTER_AREA)
    small = cv2.medianBlur(small, 5)
    sh, sw = small.shape
    min_r = int(min(sh, sw) * 0.12)
    max_r = int(min(sh, sw) * 0.45)
    circles = cv2.HoughCircles(
        small, cv2.HOUGH_GRADIENT, dp=1.0, minDist=min(sh, sw) // 2,
        param1=100, param2=30, minRadius=min_r, maxRadius=max_r,
    )
    if circles is None:
        return None
    cx, cy, r = circles[0][0]
    return float(cx / scale), float(cy / scale), float(r / scale)


# -----------------------------------------------------------------------------
# Polar sampling (precomputed LUT)
# -----------------------------------------------------------------------------

def polar_lut(side, cx, cy, r_inner, r_outer, n_angles, n_radial):
    angles = np.arange(n_angles) * (2 * np.pi / n_angles)
    sin_a = np.sin(angles)
    cos_a = -np.cos(angles)
    radii = r_inner + np.arange(n_radial) * (r_outer - r_inner) / max(1, n_radial - 1)
    px = cx + np.outer(sin_a, radii)
    py = cy + np.outer(cos_a, radii)
    ix = np.floor(px).astype(np.int32)
    iy = np.floor(py).astype(np.int32)
    fx = (px - ix).astype(np.float32)
    fy = (py - iy).astype(np.float32)
    valid = (ix >= 0) & (ix < side - 1) & (iy >= 0) & (iy < side - 1)
    ix = np.clip(ix, 0, side - 2)
    iy = np.clip(iy, 0, side - 2)
    return ix, iy, fx, fy, valid


def sample_polar_max(image, lut):
    """Return per-angle MAX of pixels along the radial samples (instead of
    mean). The second hand tip is bright, so max highlights it cleanly."""
    ix, iy, fx, fy, valid = lut
    img = image.astype(np.float32)
    v00 = img[iy, ix]
    v01 = img[iy, ix + 1]
    v10 = img[iy + 1, ix]
    v11 = img[iy + 1, ix + 1]
    samples = (v00 * (1 - fx) * (1 - fy) + v01 * fx * (1 - fy) +
               v10 * (1 - fx) * fy + v11 * fx * fy)
    samples = np.where(valid, samples, -np.inf)
    return samples.max(axis=1)


def sample_polar_mean(image, lut):
    ix, iy, fx, fy, valid = lut
    img = image.astype(np.float32)
    v00 = img[iy, ix]
    v01 = img[iy, ix + 1]
    v10 = img[iy + 1, ix]
    v11 = img[iy + 1, ix + 1]
    samples = (v00 * (1 - fx) * (1 - fy) + v01 * fx * (1 - fy) +
               v10 * (1 - fx) * fy + v11 * fx * fy)
    samples *= valid
    counts = valid.sum(axis=1).clip(min=1)
    return samples.sum(axis=1) / counts


# -----------------------------------------------------------------------------
# Peak finding
# -----------------------------------------------------------------------------

def smooth_circular(signal, kernel_bins):
    """Box-filter (mean) along a circular array."""
    n = len(signal)
    k = kernel_bins
    if k <= 1:
        return signal.copy()
    # Use FFT convolution for speed
    kern = np.zeros(n)
    half = k // 2
    kern[:half + 1] = 1.0 / k
    kern[-half:] = 1.0 / k if half > 0 else 0.0
    return np.real(np.fft.ifft(np.fft.fft(signal) * np.fft.fft(kern)))


def find_peaks_with_width(signal, min_prominence=1.0):
    """Find local maxima with FWHM. Returns list of (idx, height, fwhm_bins)."""
    n = len(signal)
    peaks = []
    for i in range(n):
        l = signal[(i - 1) % n]
        r = signal[(i + 1) % n]
        if signal[i] > l and signal[i] > r:
            peaks.append(i)
    out = []
    for idx in peaks:
        height = signal[idx]
        if height < min_prominence:
            continue
        half = height / 2.0
        # Walk left
        i = idx
        while i > idx - n // 2:
            if signal[(i - 1) % n] < half:
                break
            i -= 1
        left = i
        # Walk right
        i = idx
        while i < idx + n // 2:
            if signal[(i + 1) % n] < half:
                break
            i += 1
        right = i
        fwhm = right - left + 1
        out.append((idx, height, fwhm))
    return out


def parabolic_subbin(signal, idx):
    n = len(signal)
    a = signal[(idx - 1) % n]
    b = signal[idx % n]
    c = signal[(idx + 1) % n]
    denom = (a - 2 * b + c)
    if abs(denom) < 1e-9:
        return float(idx)
    return idx + 0.5 * (a - c) / denom


# -----------------------------------------------------------------------------
# Second-hand reader
# -----------------------------------------------------------------------------

@dataclass
class Reading:
    angle_deg: float
    confidence: float
    fwhm_deg: float
    height: float
    n_candidates: int
    debug: dict


def read_second_hand(bgr_frame, save_diag=None):
    """Detect the second hand in a single frame.

    Algorithm:
      - Detect dial center.
      - Compute per-angle MEAN intensity in a NARROW ring [0.20, 0.70]R that's
        safely inside the bezel (bezel markings excluded).
      - Subtract a 30°-smoothed angular background. This kills the dial's
        dark/light variation and any slow lighting gradient.
      - The remaining signal has peaks at hand positions (signed: positive
        for bright hands on dark dial, negative for dark hands on light dial).
      - Use BOTH polarities, take |residual|.
      - Each hand has a width: thin (~1°) for the second hand, wide (~3-6°)
        for hour/minute hands. Score = height / width. Best score wins.
    """
    det = detect_watch_face(bgr_frame)
    if det is None:
        return None
    cx, cy, r = det

    side = int(r * 2.1)
    H, W = bgr_frame.shape[:2]
    x0 = int(round(cx - side / 2))
    y0 = int(round(cy - side / 2))
    x0 = max(0, min(W - side, x0))
    y0 = max(0, min(H - side, y0))
    side = min(side, W - x0, H - y0)

    crop_bgr = bgr_frame[y0:y0 + side, x0:x0 + side].copy()
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)

    cx_local = cx - x0
    cy_local = cy - y0
    r_local = r

    # Narrow ring inside the dial, well clear of the bezel (which sits at
    # ~0.85R+) and the central pinion area.
    r_inner = r_local * 0.20
    r_outer = r_local * 0.70
    n_radial = 50
    lut = polar_lut(side, cx_local, cy_local, r_inner, r_outer,
                    N_ANGLES, n_radial)

    profile = sample_polar_mean(gray, lut)

    # Subtract a 30°-smoothed background. Hands appear as sharp deviations.
    bg = smooth_circular(profile, kernel_bins=int(N_ANGLES * 30 / 360))
    signed = profile - bg
    residual = np.abs(signed)

    # Candidate peaks with width (FWHM measured on the |residual| signal).
    peaks = find_peaks_with_width(residual, min_prominence=1.0)
    if not peaks:
        return None

    bins_per_deg = N_ANGLES / 360.0

    # Reject implausibly wide peaks (>8° = could be the hour hand body or
    # overlapping minute+hour). Reject implausibly narrow peaks (<0.5° =
    # noise spikes / lume markers / single-bin artifacts).
    min_fwhm_bins = max(2, int(0.5 * bins_per_deg))
    max_fwhm_bins = int(8 * bins_per_deg)
    candidates = [(idx, h, w) for (idx, h, w) in peaks
                  if min_fwhm_bins <= w <= max_fwhm_bins]
    if not candidates:
        return None

    # Score: height × thinness. Thin=high-frequency, but not so thin it's
    # noise. We prefer the LONGEST thin hand → use both height and width.
    # Score = height / sqrt(fwhm) emphasizes tall+thin.
    scored = sorted(
        candidates,
        key=lambda p: -p[1] / max(np.sqrt(p[2]), 1.0),
    )
    best_idx, best_h, best_w = scored[0]

    sub = parabolic_subbin(residual, best_idx)
    angle_deg = (sub * 360.0 / N_ANGLES) % 360.0

    if len(scored) >= 2:
        s0 = scored[0][1] / max(np.sqrt(scored[0][2]), 1.0)
        s1 = scored[1][1] / max(np.sqrt(scored[1][2]), 1.0)
        confidence = float(min(s0 / max(s1, 1e-6), 10.0))
    else:
        confidence = 5.0

    if save_diag is not None:
        diag = crop_bgr.copy()
        cv2.circle(diag, (int(cx_local), int(cy_local)), int(r_inner),
                   (60, 60, 60), 1)
        cv2.circle(diag, (int(cx_local), int(cy_local)), int(r_outer),
                   (60, 60, 60), 1)
        rad = np.radians(angle_deg)
        tx = int(cx_local + np.sin(rad) * r_outer * 1.05)
        ty = int(cy_local - np.cos(rad) * r_outer * 1.05)
        cv2.line(diag, (int(cx_local), int(cy_local)), (tx, ty), (0, 0, 255), 2)
        for idx, h, w in scored[:3]:
            ang = idx * 360.0 / N_ANGLES
            rad = np.radians(ang)
            tx = int(cx_local + np.sin(rad) * r_outer)
            ty = int(cy_local - np.cos(rad) * r_outer)
            cv2.line(diag, (int(cx_local), int(cy_local)), (tx, ty),
                     (0, 200, 255), 1)
        cv2.putText(diag, f"sec={angle_deg:.1f}d conf={confidence:.1f} "
                    f"fwhm={best_w / bins_per_deg:.1f}d",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imwrite(save_diag, diag)

        plot_h = 240
        plot_w = N_ANGLES
        plot = np.zeros((plot_h, plot_w, 3), dtype=np.uint8)
        rmin = residual.min()
        rmax = residual.max()
        scale_y = (plot_h - 10) / max(rmax - rmin, 1e-6)
        for x in range(plot_w):
            y = int(plot_h - 5 - (residual[x] - rmin) * scale_y)
            cv2.line(plot, (x, plot_h - 5), (x, y), (180, 180, 180), 1)
        bx = int(best_idx)
        cv2.line(plot, (bx, 0), (bx, plot_h), (0, 0, 255), 1)
        for idx, _, _ in scored[1:5]:
            cv2.line(plot, (int(idx), 0), (int(idx), plot_h), (0, 200, 255), 1)
        cv2.imwrite(save_diag.replace(".jpg", "_profile.png"), plot)

    return Reading(
        angle_deg=angle_deg,
        confidence=confidence,
        fwhm_deg=best_w / bins_per_deg,
        height=best_h,
        n_candidates=len(candidates),
        debug={"top_candidates": [
            (float(idx) * 360.0 / N_ANGLES, float(h), float(w / bins_per_deg))
            for idx, h, w in scored[:5]
        ]},
    )


def main(argv):
    if len(argv) < 2:
        print("usage: read_seconds.py <image1> [image2 ...]")
        return
    out_dir = os.path.join(os.path.dirname(__file__), "diagnostics_singleframe")
    os.makedirs(out_dir, exist_ok=True)
    for path in argv[1:]:
        name = os.path.splitext(os.path.basename(path))[0]
        bgr = cv2.imread(path)
        if bgr is None:
            print(f"{name}: cannot read")
            continue
        diag = os.path.join(out_dir, f"{name}_detect.jpg")
        r = read_second_hand(bgr, save_diag=diag)
        if r is None:
            print(f"{name}: detection failed")
            continue
        print(f"{name}: angle={r.angle_deg:6.2f}°  conf={r.confidence:.2f}  "
              f"fwhm={r.fwhm_deg:.2f}°  cands={r.n_candidates}")
        for a, h, w in r.debug["top_candidates"]:
            print(f"    cand: angle={a:6.2f}°  height={h:5.1f}  fwhm={w:.2f}°")


if __name__ == "__main__":
    main(sys.argv)
