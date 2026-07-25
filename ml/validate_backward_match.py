#!/usr/bin/env python3
"""Test backward-matching approach for second hand detection.

For each peak in the current frame, verify it's the second hand by
checking if peaks existed at the predicted positions in past frames:
- 0.5s ago: peak at angle ± 3° (for ±6°/s)
- 1.0s ago: peak at angle ± 6°
- 1.5s ago: peak at angle ± 9°

A static feature won't have matching peaks at the offset positions
(unless by coincidence). The second hand consistently matches.
"""

import math
import os
import subprocess
import tempfile
from collections import deque

import numpy as np
from PIL import Image

VIDEOS_DIR = os.path.join(os.path.dirname(__file__), "videos")
FPS = 30
BUFFER_SIZE = 300
NUM_RAYS = 720
SAMPLES_PER_RAY = 50
INNER_R_FRAC = 0.2
OUTER_R_FRAC = 0.9
BG_WINDOW = 20

# Backward matching parameters — use many lookback distances including
# non-multiples of 6° to distinguish from minute markers (spaced 6° apart)
LOOKBACK_FRAMES = [10, 20, 35, 50, 60]  # 0.33s, 0.67s, 1.17s, 1.67s, 2.0s
# Expected offsets at 6°/s: 2°, 4°, 7°, 10°, 12° — NOT all multiples of 6°
EXPECTED_VEL = 6.0  # °/s
MATCH_TOLERANCE = 2.5  # degrees
MIN_MATCHES_REQUIRED = 4  # must match at least 4 out of 5 lookbacks


def extract_frames(video_path, output_dir, fps=30):
    os.makedirs(output_dir, exist_ok=True)
    cmd = [
        "ffmpeg", "-i", video_path,
        "-vf", f"fps={fps}",
        "-q:v", "2",
        os.path.join(output_dir, "frame_%06d.jpg"),
        "-y", "-loglevel", "error",
    ]
    subprocess.run(cmd, check=True)
    frames = sorted([f for f in os.listdir(output_dir) if f.endswith(".jpg")])
    return [os.path.join(output_dir, f) for f in frames]


def crop_center_square(img):
    w, h = img.size
    side = int(min(w, h) * 0.55)
    left = (w - side) // 2
    top = (h - side) // 2
    return img.crop((left, top, left + side, top + side))


def compute_radial_profile(buf):
    center = BUFFER_SIZE / 2.0
    inner_r = center * INNER_R_FRAC
    outer_r = center * OUTER_R_FRAC
    step_r = (outer_r - inner_r) / (SAMPLES_PER_RAY - 1)
    profile = np.zeros(NUM_RAYS)
    for i in range(NUM_RAYS):
        angle_rad = math.radians(i * 360.0 / NUM_RAYS)
        dx, dy = math.sin(angle_rad), -math.cos(angle_rad)
        total = 0.0
        for s in range(SAMPLES_PER_RAY):
            r = inner_r + s * step_r
            px, py = center + dx * r, center + dy * r
            fx = max(0, min(px, BUFFER_SIZE - 2))
            fy = max(0, min(py, BUFFER_SIZE - 2))
            ix, iy = int(fx), int(fy)
            ddx, ddy = fx - ix, fy - iy
            total += (buf[iy, ix] * (1 - ddx) * (1 - ddy) +
                      buf[iy, ix + 1] * ddx * (1 - ddy) +
                      buf[iy + 1, ix] * (1 - ddx) * ddy +
                      buf[iy + 1, ix + 1] * ddx * ddy)
        profile[i] = total / SAMPLES_PER_RAY
    return profile


def compute_hand_scores(profile):
    n = NUM_RAYS
    half_win = BG_WINDOW // 2
    scores = np.zeros(n)
    for i in range(n):
        bg_sum, bg_count = 0.0, 0
        for j in range(-half_win, half_win + 1):
            if abs(j) <= 2: continue
            bg_sum += profile[(i + j) % n]
            bg_count += 1
        scores[i] = abs(profile[i] - bg_sum / bg_count)
    return scores


def find_peaks(scores):
    n = NUM_RAYS
    median_score = np.median(scores)
    min_prominence = max(median_score * 0.5, 3.0)
    raw = []
    for i in range(n):
        if (scores[i] > scores[(i - 1) % n] and
                scores[i] > scores[(i + 1) % n] and
                scores[i] > min_prominence):
            raw.append((i, scores[i]))
    raw.sort(key=lambda x: -x[1])
    kept = []
    for idx, score in raw:
        if any(min(abs(idx - ki), n - abs(idx - ki)) < 10 for ki, _ in kept):
            continue
        kept.append((idx, score))
        if len(kept) >= 10:
            break
    return kept  # list of (index, score)


def angle_dist(a, b):
    """Circular distance in degrees."""
    d = abs(a - b) % 360
    return min(d, 360 - d)


def backward_match(current_peaks, past_peaks_list, direction):
    """For each current peak, check backward matches at expected offsets.

    direction: +1 for clockwise (positive velocity), -1 for counter-clockwise.
    Returns list of (peak_angle, match_count, avg_score) for peaks that match.
    """
    results = []

    for idx, score in current_peaks:
        angle = idx * 360.0 / NUM_RAYS
        matches = 0

        for li, (past_peaks, lookback_frames) in enumerate(past_peaks_list):
            if past_peaks is None:
                continue
            # Expected position in the past
            dt = lookback_frames / FPS
            expected_past_angle = (angle - direction * EXPECTED_VEL * dt) % 360

            # Check if any past peak is near expected position
            found = False
            for pidx, pscore in past_peaks:
                past_angle = pidx * 360.0 / NUM_RAYS
                if angle_dist(past_angle, expected_past_angle) < MATCH_TOLERANCE:
                    found = True
                    break

            if found:
                matches += 1

        if matches > 0:
            results.append((angle, matches, score))

    return results


def main():
    videos = sorted([f for f in os.listdir(VIDEOS_DIR)
                     if f.endswith((".MOV", ".mov", ".mp4"))])
    if not videos:
        print("No videos found!")
        return

    print(f"Found {len(videos)} videos")
    print(f"Lookbacks: {LOOKBACK_FRAMES} frames, tolerance: ±{MATCH_TOLERANCE}°")
    print(f"Testing both +6°/s and -6°/s directions\n")

    results = []

    for video_name in videos:
        video_path = os.path.join(VIDEOS_DIR, video_name)
        print(f"=== {video_name} ===")

        with tempfile.TemporaryDirectory() as tmp_dir:
            frame_paths = extract_frames(video_path, tmp_dir, fps=FPS)
            max_frames = FPS * 5
            frame_paths = frame_paths[:max_frames]

            if len(frame_paths) < max(LOOKBACK_FRAMES) + 10:
                print(f"  Only {len(frame_paths)} frames, skipping\n")
                continue

            print(f"  Processing {len(frame_paths)} frames ({len(frame_paths)/FPS:.1f}s)...")

            # Store peaks for each frame
            all_peaks = []
            for fi, fp in enumerate(frame_paths):
                img = Image.open(fp).convert("RGB")
                img = crop_center_square(img)
                buf = np.array(img.convert("L").resize((BUFFER_SIZE, BUFFER_SIZE), Image.LANCZOS), dtype=np.float64)
                profile = compute_radial_profile(buf)
                scores = compute_hand_scores(profile)
                peaks = find_peaks(scores)
                all_peaks.append(peaks)

            # Now do backward matching for each frame
            detected_cw = []  # (time, angle) for clockwise matches
            detected_ccw = []  # for counter-clockwise

            for fi in range(max(LOOKBACK_FRAMES), len(all_peaks)):
                time_s = fi / FPS
                current = all_peaks[fi]

                # Get past peaks at each lookback distance
                past_list = []
                for lb in LOOKBACK_FRAMES:
                    past_fi = fi - lb
                    if past_fi >= 0:
                        past_list.append((all_peaks[past_fi], lb))
                    else:
                        past_list.append((None, lb))

                # Try both directions
                for direction, det_list, label in [
                    (+1, detected_cw, "CW"),
                    (-1, detected_ccw, "CCW"),
                ]:
                    matches = backward_match(current, past_list, direction)
                    # Keep peaks matching most lookback distances
                    full_matches = [(a, m, s) for a, m, s in matches if m >= MIN_MATCHES_REQUIRED]
                    if full_matches:
                        # Pick the one with highest score
                        best = max(full_matches, key=lambda x: x[2])
                        det_list.append((time_s, best[0]))

                if fi % 30 == 0:
                    cw_angle = detected_cw[-1][1] if detected_cw and detected_cw[-1][0] == time_s else None
                    ccw_angle = detected_ccw[-1][1] if detected_ccw and detected_ccw[-1][0] == time_s else None
                    print(f"  t={time_s:.1f}s: CW={f'{cw_angle:.1f}°' if cw_angle else 'none':>8s}, "
                          f"CCW={f'{ccw_angle:.1f}°' if ccw_angle else 'none':>8s}")

            # Analyze both directions, pick the better one
            best_result = None
            for det_list, label in [(detected_cw, "CW(+6)"), (detected_ccw, "CCW(-6)")]:
                if len(det_list) < 10:
                    print(f"\n  {label}: only {len(det_list)} detections")
                    continue

                times = np.array([t for t, a in det_list])
                angles = np.array([a for t, a in det_list])

                # Unwrap
                for i in range(1, len(angles)):
                    diff = angles[i] - angles[i - 1]
                    if diff > 180: diff -= 360
                    if diff < -180: diff += 360
                    angles[i] = angles[i - 1] + diff

                mean_t = np.mean(times)
                mean_a = np.mean(angles)
                slope = np.sum((times - mean_t) * (angles - mean_a)) / max(np.sum((times - mean_t) ** 2), 1e-10)
                predicted = mean_a + slope * (times - mean_t)
                rms = np.sqrt(np.mean((angles - predicted) ** 2))

                vel_err = abs(abs(slope) - 6)
                coverage = len(det_list) / (len(all_peaks) - max(LOOKBACK_FRAMES))

                print(f"\n  {label}: {len(det_list)} detections ({coverage:.0%} coverage)")
                print(f"    Velocity: {slope:.2f}°/s, RMS: {rms:.2f}°")

                if best_result is None or vel_err < best_result[0]:
                    best_result = (vel_err, slope, rms, label, coverage)

            if best_result:
                vel_err, slope, rms, label, coverage = best_result
                if vel_err < 2 and coverage > 0.3:
                    print(f"\n  RESULT: GOOD — {label} at {slope:.1f}°/s ({coverage:.0%} coverage)")
                    results.append(("GOOD", video_name, slope))
                elif vel_err < 4 and coverage > 0.2:
                    print(f"\n  RESULT: PARTIAL — {label} at {slope:.1f}°/s ({coverage:.0%} coverage)")
                    results.append(("PARTIAL", video_name, slope))
                else:
                    print(f"\n  RESULT: POOR — best was {label} at {slope:.1f}°/s ({coverage:.0%} coverage)")
                    results.append(("POOR", video_name, slope))
            else:
                print(f"\n  RESULT: FAILED — no detections")
                results.append(("FAILED", video_name, 0))

        print()

    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for status, name, vel in results:
        print(f"  {status:10s} {name:20s} vel={vel:+.1f}°/s")
    good = sum(1 for s, _, _ in results if s in ("GOOD", "PARTIAL"))
    print(f"\n  {good}/{len(results)} videos: GOOD or PARTIAL")


if __name__ == "__main__":
    main()
