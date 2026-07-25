#!/usr/bin/env python3
"""Validate the radial-scan classical CV approach against real watch videos.

Mirrors the Swift SecondHandAnalyzer algorithm:
1. Extract frames, crop center square, convert to 300x300 grayscale
2. Radial scan: 720 rays, 50 samples per ray, 0.2R–0.9R
3. Hand scoring: absolute deviation from local background
4. Peak detection with NMS
5. Temporal tracking with velocity estimation
6. Second hand selection by |velocity| ≈ 6°/s (handles both +6 and -6)

Reports per-video: detected peaks, tracked hands, velocity estimates,
and whether the second hand is successfully identified.
"""

import math
import os
import subprocess
import sys
import tempfile

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

# Tracking parameters (matching Swift)
MIN_FRAMES_LOCK = 45        # ~1.5s at 30fps
EXPECTED_VEL = 6.0           # °/s
VEL_TOLERANCE = 2.5          # ±2.5 from 6
MAX_VEL_RESIDUAL = 3.0       # max RMS residual for lock
UNLOCK_VEL_THRESHOLD = 4.0   # unlock if |vel| deviates this much


def extract_frames(video_path, output_dir, fps=30):
    """Extract frames from video using ffmpeg."""
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
    """Crop center square (matching app's 0.55 factor)."""
    w, h = img.size
    side = int(min(w, h) * 0.55)
    left = (w - side) // 2
    top = (h - side) // 2
    return img.crop((left, top, left + side, top + side))


def to_grayscale_buffer(img):
    """Convert PIL image to 300x300 grayscale numpy array."""
    gray = img.convert("L").resize((BUFFER_SIZE, BUFFER_SIZE), Image.LANCZOS)
    return np.array(gray, dtype=np.float64)


def sample_pixel_bilinear(buf, x, y):
    """Bilinear interpolation sample."""
    h, w = buf.shape
    fx = max(0, min(x, w - 2))
    fy = max(0, min(y, h - 2))
    ix = int(fx)
    iy = int(fy)
    dx = fx - ix
    dy = fy - iy
    return (buf[iy, ix] * (1 - dx) * (1 - dy) +
            buf[iy, ix + 1] * dx * (1 - dy) +
            buf[iy + 1, ix] * (1 - dx) * dy +
            buf[iy + 1, ix + 1] * dx * dy)


def compute_radial_profile(buf):
    """Compute average brightness along each of 720 radial rays."""
    center = BUFFER_SIZE / 2.0
    radius = center
    inner_r = radius * INNER_R_FRAC
    outer_r = radius * OUTER_R_FRAC
    step_r = (outer_r - inner_r) / (SAMPLES_PER_RAY - 1)

    profile = np.zeros(NUM_RAYS)

    for i in range(NUM_RAYS):
        angle_deg = i * 360.0 / NUM_RAYS
        angle_rad = math.radians(angle_deg)
        dx = math.sin(angle_rad)
        dy = -math.cos(angle_rad)

        total = 0.0
        for s in range(SAMPLES_PER_RAY):
            r = inner_r + s * step_r
            px = center + dx * r
            py = center + dy * r
            total += sample_pixel_bilinear(buf, px, py)
        profile[i] = total / SAMPLES_PER_RAY

    return profile


def compute_hand_scores(profile):
    """Score each angle by deviation from local background."""
    n = NUM_RAYS
    half_win = BG_WINDOW // 2
    scores = np.zeros(n)

    for i in range(n):
        bg_sum = 0.0
        bg_count = 0
        for j in range(-half_win, half_win + 1):
            if abs(j) <= 2:
                continue
            idx = (i + j) % n
            bg_sum += profile[idx]
            bg_count += 1
        bg = bg_sum / bg_count
        scores[i] = abs(profile[i] - bg)

    return scores


def find_peaks(scores):
    """Find peaks with NMS."""
    n = NUM_RAYS
    median_score = np.median(scores)
    min_prominence = max(median_score * 0.5, 3.0)

    raw_peaks = []
    for i in range(n):
        prev_val = scores[(i - 1) % n]
        curr = scores[i]
        next_val = scores[(i + 1) % n]
        if curr > prev_val and curr > next_val and curr > min_prominence:
            raw_peaks.append((i, curr))

    raw_peaks.sort(key=lambda x: -x[1])

    nms_dist = 10
    kept = []
    for idx, score in raw_peaks:
        too_close = False
        for k_idx, _, _, _ in kept:
            d = abs(idx - k_idx)
            dist = min(d, n - d)
            if dist < nms_dist:
                too_close = True
                break
        if too_close:
            continue

        half_max = score / 2
        wl = 0
        for j in range(1, n // 2):
            if scores[(idx - j) % n] < half_max:
                wl = j
                break
        wr = 0
        for j in range(1, n // 2):
            if scores[(idx + j) % n] < half_max:
                wr = j
                break

        angle_deg = idx * 360.0 / NUM_RAYS
        kept.append((idx, angle_deg, score, wl + wr))

        if len(kept) >= 10:
            break

    return kept


class HandTrack:
    _next_id = 0

    def __init__(self, angle_deg, score, time_s):
        self.id = HandTrack._next_id
        HandTrack._next_id += 1
        self.angle_deg = angle_deg
        self.velocity_dps = 0.0
        self.velocity_residual = float('inf')
        self.last_seen_time = time_s
        self.frame_count = 1
        self.score = score
        self.recent_angles = [(time_s, angle_deg)]

    def update(self, angle_deg, score, time_s):
        new_angle = angle_deg
        if self.recent_angles:
            last_angle = self.recent_angles[-1][1]
            diff = new_angle - (last_angle % 360)
            if diff > 180: diff -= 360
            if diff < -180: diff += 360
            new_angle = last_angle + diff

        self.angle_deg = angle_deg
        self.last_seen_time = time_s
        self.frame_count += 1
        self.score = score
        self.recent_angles.append((time_s, new_angle))
        if len(self.recent_angles) > 90:
            self.recent_angles.pop(0)
        self._estimate_velocity()

    def _estimate_velocity(self):
        if len(self.recent_angles) < 5:
            self.velocity_dps = 0
            self.velocity_residual = float('inf')
            return
        times = np.array([a[0] for a in self.recent_angles])
        angles = np.array([a[1] for a in self.recent_angles])
        mean_t = np.mean(times)
        mean_a = np.mean(angles)
        num = np.sum((times - mean_t) * (angles - mean_a))
        den = np.sum((times - mean_t) ** 2)
        if den < 1e-10:
            self.velocity_dps = 0
            self.velocity_residual = float('inf')
        else:
            self.velocity_dps = num / den
            predicted = mean_a + self.velocity_dps * (times - mean_t)
            residuals = angles - predicted
            self.velocity_residual = np.sqrt(np.mean(residuals**2))


def match_and_update_tracks(tracks, peaks, time_s):
    """Match detected peaks to existing tracks, create new tracks for unmatched."""
    match_threshold = 15.0
    used = [False] * len(peaks)

    for track in tracks:
        dt = time_s - track.last_seen_time
        predicted = (track.angle_deg + track.velocity_dps * dt) % 360

        best_idx = -1
        best_dist = match_threshold
        for pi, (_, angle_deg, _, _) in enumerate(peaks):
            if used[pi]:
                continue
            diff = angle_deg - predicted
            if diff > 180: diff -= 360
            if diff < -180: diff += 360
            dist = abs(diff)
            if dist < best_dist:
                best_dist = dist
                best_idx = pi

        if best_idx >= 0:
            used[best_idx] = True
            _, angle_deg, score, _ = peaks[best_idx]
            track.update(angle_deg, score, time_s)

    for pi, (idx, angle_deg, score, width) in enumerate(peaks):
        if not used[pi]:
            tracks.append(HandTrack(angle_deg, score, time_s))

    tracks[:] = [t for t in tracks if time_s - t.last_seen_time <= 2.0]


def main():
    videos = sorted([f for f in os.listdir(VIDEOS_DIR)
                     if f.endswith((".MOV", ".mov", ".mp4"))])
    if not videos:
        print("No videos found!")
        return

    print(f"Found {len(videos)} videos")
    print(f"Parameters: lock_frames={MIN_FRAMES_LOCK}, vel_tol=±{VEL_TOLERANCE}°/s, "
          f"max_residual={MAX_VEL_RESIDUAL}°, checking |vel|≈6°/s\n")

    results = []

    for video_name in videos:
        video_path = os.path.join(VIDEOS_DIR, video_name)
        print(f"=== {video_name} ===")

        with tempfile.TemporaryDirectory() as tmp_dir:
            frames = extract_frames(video_path, tmp_dir, fps=FPS)
            max_frames = FPS * 5
            frames = frames[:max_frames]

            if len(frames) < 10:
                print(f"  Only {len(frames)} frames, skipping\n")
                continue

            print(f"  Processing {len(frames)} frames ({len(frames)/FPS:.1f}s)...")

            HandTrack._next_id = 0
            tracks = []
            locked_id = None
            locked_direction = 1.0

            all_peaks_per_frame = []
            locked_angles = []

            for fi, frame_path in enumerate(frames):
                time_s = fi / FPS

                img = Image.open(frame_path).convert("RGB")
                img = crop_center_square(img)
                buf = to_grayscale_buffer(img)

                profile = compute_radial_profile(buf)
                scores = compute_hand_scores(profile)
                peaks = find_peaks(scores)

                all_peaks_per_frame.append(peaks)
                match_and_update_tracks(tracks, peaks, time_s)

                # Locked track validation
                if locked_id is not None:
                    locked_track = next((t for t in tracks if t.id == locked_id), None)
                    if locked_track:
                        # Check velocity hasn't drifted
                        vel_err = abs(abs(locked_track.velocity_dps) - EXPECTED_VEL)
                        if vel_err > UNLOCK_VEL_THRESHOLD and locked_track.frame_count > MIN_FRAMES_LOCK * 2:
                            locked_id = None
                        else:
                            locked_angles.append((time_s, locked_track.angle_deg))
                    else:
                        locked_id = None

                # Try to lock
                if locked_id is None:
                    best = None
                    best_err = VEL_TOLERANCE
                    for t in tracks:
                        if t.frame_count < MIN_FRAMES_LOCK:
                            continue
                        if t.velocity_residual > MAX_VEL_RESIDUAL:
                            continue
                        # Check |velocity| ≈ 6°/s (handles both +6 and -6)
                        err = abs(abs(t.velocity_dps) - EXPECTED_VEL)
                        if err < best_err:
                            best_err = err
                            best = t
                    if best:
                        locked_id = best.id
                        locked_direction = 1.0 if best.velocity_dps >= 0 else -1.0
                        locked_angles.append((time_s, best.angle_deg))
                        print(f"  ** LOCKED track {best.id} at t={time_s:.1f}s, "
                              f"vel={best.velocity_dps:.1f}°/s, residual={best.velocity_residual:.1f}°, "
                              f"dir={'CW' if locked_direction > 0 else 'CCW'}")

                if fi % 30 == 0:
                    n_peaks = len(peaks)
                    n_tracks = len(tracks)
                    # Show tracks with |vel| near 6
                    near6 = [(t.id, t.velocity_dps, t.velocity_residual, t.frame_count)
                             for t in tracks
                             if t.frame_count >= 5 and abs(abs(t.velocity_dps) - 6) < 5]
                    print(f"  t={time_s:.1f}s: {n_peaks} peaks, {n_tracks} tracks"
                          f"{' LOCKED='+str(locked_id) if locked_id is not None else ''}"
                          f"  near-6: {[(tid, f'{v:.1f}°/s', f'res={r:.1f}', fc) for tid, v, r, fc in near6]}")

            # Summary
            print(f"\n  Final: {len(tracks)} tracks alive")
            for t in sorted(tracks, key=lambda x: -x.frame_count)[:10]:
                marker = " <-- LOCKED" if t.id == locked_id else ""
                near6 = " [~6°/s!]" if abs(abs(t.velocity_dps) - 6) < 3 else ""
                print(f"    Track {t.id}: vel={t.velocity_dps:.1f}°/s, res={t.velocity_residual:.1f}°, "
                      f"frames={t.frame_count}, angle={t.angle_deg:.1f}°{near6}{marker}")

            if locked_angles:
                times = np.array([a[0] for a in locked_angles])
                angles = np.array([a[1] for a in locked_angles])

                if len(times) > 1:
                    for i in range(1, len(angles)):
                        diff = angles[i] - angles[i-1]
                        if diff > 180: diff -= 360
                        if diff < -180: diff += 360
                        angles[i] = angles[i-1] + diff

                    mean_t = np.mean(times)
                    mean_a = np.mean(angles)
                    slope = np.sum((times - mean_t) * (angles - mean_a)) / max(np.sum((times - mean_t)**2), 1e-10)

                    residuals_arr = angles - (mean_a + slope * (times - mean_t))
                    rms_error = np.sqrt(np.mean(residuals_arr**2))

                    print(f"\n  Locked track velocity: {slope:.2f}°/s (expected ±6°/s)")
                    print(f"  Locked for {len(locked_angles)} frames ({times[-1]-times[0]:.1f}s)")
                    print(f"  RMS residual: {rms_error:.2f}°")

                    if abs(abs(slope) - 6) < 2:
                        print(f"  RESULT: GOOD — tracks second hand at {slope:.1f}°/s")
                        results.append(("GOOD", video_name, slope))
                    elif abs(abs(slope) - 6) < 4:
                        print(f"  RESULT: PARTIAL — velocity {slope:.1f}°/s is close")
                        results.append(("PARTIAL", video_name, slope))
                    else:
                        print(f"  RESULT: POOR — velocity {slope:.1f}°/s doesn't match second hand")
                        results.append(("POOR", video_name, slope))
            else:
                print(f"\n  RESULT: NO LOCK — never achieved lock in {len(frames)/FPS:.1f}s")
                results.append(("NO LOCK", video_name, 0))

        print()

    # Overall summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for status, name, vel in results:
        print(f"  {status:10s} {name:20s} vel={vel:+.1f}°/s")
    good = sum(1 for s, _, _ in results if s == "GOOD")
    print(f"\n  {good}/{len(results)} videos: GOOD")


if __name__ == "__main__":
    main()
