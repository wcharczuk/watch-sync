#!/usr/bin/env python3
"""Refined median subtraction + tracking approach.

Improvements over v1:
- Fewer peaks (top 5) with wider NMS separation to reduce noise
- Tighter matching threshold (8°) with velocity-guided prediction
- Longer lock requirement (45 frames = 1.5s)
- Stricter residual requirement (3.0°)
- Better velocity estimation using only recent history for lock decision
- After lock, use very tight velocity-guided matching
"""

import math
import os
import subprocess
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

MEDIAN_WINDOW = 90
MIN_FRAMES_LOCK = 45
EXPECTED_VEL = 6.0
VEL_TOLERANCE = 2.5
MAX_RESIDUAL = 3.0


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


def find_residual_peaks(residual, max_peaks=5):
    """Find top peaks in median-subtracted residual with wide NMS."""
    n = len(residual)
    abs_res = np.abs(residual)

    # Smooth
    kernel = 3
    smoothed = np.zeros(n)
    for i in range(n):
        total = 0
        for k in range(-kernel, kernel + 1):
            total += abs_res[(i + k) % n]
        smoothed[i] = total / (2 * kernel + 1)

    # Threshold
    median_val = np.median(smoothed)
    mad = np.median(np.abs(smoothed - median_val))
    threshold = median_val + 2.0 * max(mad, 0.5)

    # Find local maxima
    raw = []
    for i in range(n):
        if (smoothed[i] > smoothed[(i - 1) % n] and
                smoothed[i] > smoothed[(i + 1) % n] and
                smoothed[i] > threshold):
            raw.append((i, smoothed[i]))

    raw.sort(key=lambda x: -x[1])

    # NMS with wider separation (15 indices = 7.5°)
    kept = []
    for idx, val in raw:
        if any(min(abs(idx - ki), n - abs(idx - ki)) < 15 for ki, _ in kept):
            continue
        angle = idx * 360.0 / n
        kept.append((idx, val))
        if len(kept) >= max_peaks:
            break

    return [(idx, idx * 360.0 / n, val) for idx, val in kept]


class Track:
    _next_id = 0

    def __init__(self, angle_deg, score, time_s):
        self.id = Track._next_id
        Track._next_id += 1
        self.angle_deg = angle_deg
        self.velocity = 0.0
        self.residual = float('inf')
        self.last_time = time_s
        self.frames = 1
        self.score = score
        self.history = [(time_s, angle_deg)]
        self.miss_count = 0  # consecutive frames without a match

    def update(self, angle_deg, score, time_s):
        if self.history:
            last = self.history[-1][1]
            diff = angle_deg - (last % 360)
            if diff > 180: diff -= 360
            if diff < -180: diff += 360
            unwrapped = last + diff
        else:
            unwrapped = angle_deg

        self.angle_deg = angle_deg
        self.last_time = time_s
        self.frames += 1
        self.score = score
        self.miss_count = 0
        self.history.append((time_s, unwrapped))
        if len(self.history) > 120:
            self.history.pop(0)
        self._fit_velocity()

    def predict(self, time_s):
        """Predict angle at given time using current velocity."""
        dt = time_s - self.last_time
        return (self.angle_deg + self.velocity * dt) % 360

    def _fit_velocity(self):
        if len(self.history) < 5:
            self.velocity = 0
            self.residual = float('inf')
            return
        # Use recent history for velocity estimation
        recent = self.history[-min(60, len(self.history)):]
        times = np.array([h[0] for h in recent])
        angles = np.array([h[1] for h in recent])
        mt = np.mean(times)
        ma = np.mean(angles)
        den = np.sum((times - mt) ** 2)
        if den < 1e-10:
            self.velocity = 0
            self.residual = float('inf')
            return
        self.velocity = np.sum((times - mt) * (angles - ma)) / den
        pred = ma + self.velocity * (times - mt)
        self.residual = np.sqrt(np.mean((angles - pred) ** 2))


def angle_diff(a, b):
    """Signed circular difference a - b in [-180, 180]."""
    d = a - b
    while d > 180: d -= 360
    while d < -180: d += 360
    return d


def match_tracks(tracks, peaks, time_s, locked_id=None):
    """Match peaks to tracks. Locked track gets priority with tight threshold."""
    used = [False] * len(peaks)

    # First pass: match locked track with tight threshold
    if locked_id is not None:
        lt = next((t for t in tracks if t.id == locked_id), None)
        if lt:
            predicted = lt.predict(time_s)
            threshold = 5.0  # tighter for locked track
            best_i = -1
            best_d = threshold
            for pi, (_, angle, score) in enumerate(peaks):
                d = abs(angle_diff(angle, predicted))
                if d < best_d:
                    best_d = d
                    best_i = pi
            if best_i >= 0:
                used[best_i] = True
                _, angle, score = peaks[best_i]
                lt.update(angle, score, time_s)
            else:
                lt.miss_count += 1

    # Second pass: match other tracks
    for track in tracks:
        if track.id == locked_id:
            continue
        predicted = track.predict(time_s)
        threshold = 10.0

        best_i = -1
        best_d = threshold
        for pi, (_, angle, score) in enumerate(peaks):
            if used[pi]:
                continue
            d = abs(angle_diff(angle, predicted))
            if d < best_d:
                best_d = d
                best_i = pi

        if best_i >= 0:
            used[best_i] = True
            _, angle, score = peaks[best_i]
            track.update(angle, score, time_s)
        else:
            track.miss_count += 1

    # Create new tracks for unmatched peaks
    for pi, (_, angle, score) in enumerate(peaks):
        if not used[pi]:
            tracks.append(Track(angle, score, time_s))

    # Prune tracks that haven't been seen recently
    tracks[:] = [t for t in tracks if t.miss_count < 30]  # ~1s without match


def main():
    videos = sorted([f for f in os.listdir(VIDEOS_DIR)
                     if f.endswith((".MOV", ".mov", ".mp4"))])
    if not videos:
        print("No videos found!")
        return

    print(f"Found {len(videos)} videos")
    print(f"Median window: {MEDIAN_WINDOW} frames ({MEDIAN_WINDOW / FPS:.1f}s)")
    print(f"Lock: {MIN_FRAMES_LOCK} frames, |vel-6| < {VEL_TOLERANCE}°/s, residual < {MAX_RESIDUAL}°\n")

    results = []

    for video_name in videos:
        video_path = os.path.join(VIDEOS_DIR, video_name)
        print(f"=== {video_name} ===")

        with tempfile.TemporaryDirectory() as tmp_dir:
            frame_paths = extract_frames(video_path, tmp_dir, fps=FPS)
            max_frames = FPS * 6
            frame_paths = frame_paths[:max_frames]

            if len(frame_paths) < MEDIAN_WINDOW + 30:
                print(f"  Only {len(frame_paths)} frames, skipping\n")
                continue

            print(f"  Processing {len(frame_paths)} frames ({len(frame_paths) / FPS:.1f}s)...")

            # Pre-compute profiles
            profiles = []
            for fp in frame_paths:
                img = Image.open(fp).convert("RGB")
                img = crop_center_square(img)
                buf = np.array(img.convert("L").resize(
                    (BUFFER_SIZE, BUFFER_SIZE), Image.LANCZOS), dtype=np.float64)
                profiles.append(compute_radial_profile(buf))

            profiles_arr = np.array(profiles)

            Track._next_id = 0
            tracks = []
            locked_id = None
            locked_angles = []

            for fi in range(MEDIAN_WINDOW, len(profiles)):
                time_s = fi / FPS

                # Median background
                window_profiles = profiles_arr[fi - MEDIAN_WINDOW:fi]
                median_profile = np.median(window_profiles, axis=0)
                residual = profiles_arr[fi] - median_profile
                residual -= np.mean(residual)

                peaks = find_residual_peaks(residual, max_peaks=5)
                match_tracks(tracks, peaks, time_s, locked_id)

                # Lock validation
                if locked_id is not None:
                    lt = next((t for t in tracks if t.id == locked_id), None)
                    if lt:
                        vel_err = abs(abs(lt.velocity) - EXPECTED_VEL)
                        if (vel_err > 4.0 and lt.frames > MIN_FRAMES_LOCK * 2) or lt.miss_count > 15:
                            print(f"  ** UNLOCK track {lt.id} at t={time_s:.1f}s, "
                                  f"vel={lt.velocity:.1f}°/s, miss={lt.miss_count}")
                            locked_id = None
                        else:
                            if lt.miss_count == 0:
                                locked_angles.append((time_s, lt.angle_deg))
                    else:
                        locked_id = None

                # Try lock
                if locked_id is None:
                    best = None
                    best_err = VEL_TOLERANCE
                    for t in tracks:
                        if t.frames < MIN_FRAMES_LOCK:
                            continue
                        if t.residual > MAX_RESIDUAL:
                            continue
                        err = abs(abs(t.velocity) - EXPECTED_VEL)
                        if err < best_err:
                            best_err = err
                            best = t
                    if best:
                        locked_id = best.id
                        locked_angles.append((time_s, best.angle_deg))
                        print(f"  ** LOCKED track {best.id} at t={time_s:.1f}s, "
                              f"vel={best.velocity:.1f}°/s, res={best.residual:.1f}°, "
                              f"frames={best.frames}")

                if fi % 30 == 0:
                    near6 = [(t.id, t.velocity, t.residual, t.frames)
                             for t in tracks
                             if t.frames >= 10 and abs(abs(t.velocity) - 6) < 5]
                    print(f"  t={time_s:.1f}s: {len(peaks)} pks, {len(tracks)} trks"
                          f"{' LOCKED=' + str(locked_id) if locked_id is not None else ''}"
                          f"  near6: {[(tid, f'{v:.1f}', f'{r:.1f}', fc) for tid, v, r, fc in near6]}")

            # Summary
            print(f"\n  Top tracks:")
            for t in sorted(tracks, key=lambda x: -x.frames)[:8]:
                marker = " <-- LOCKED" if t.id == locked_id else ""
                near6 = " [~6!]" if abs(abs(t.velocity) - 6) < 3 else ""
                print(f"    T{t.id}: vel={t.velocity:.1f}°/s res={t.residual:.1f}° "
                      f"f={t.frames}{near6}{marker}")

            if locked_angles:
                times = np.array([a[0] for a in locked_angles])
                angles = np.array([a[1] for a in locked_angles])

                for i in range(1, len(angles)):
                    diff = angles[i] - angles[i - 1]
                    if diff > 180: diff -= 360
                    if diff < -180: diff += 360
                    angles[i] = angles[i - 1] + diff

                mt = np.mean(times)
                ma = np.mean(angles)
                slope = np.sum((times - mt) * (angles - ma)) / max(
                    np.sum((times - mt) ** 2), 1e-10)
                pred = ma + slope * (times - mt)
                rms = np.sqrt(np.mean((angles - pred) ** 2))

                print(f"\n  Locked: vel={slope:.2f}°/s, {len(locked_angles)} frames "
                      f"({times[-1] - times[0]:.1f}s), RMS={rms:.2f}°")

                vel_err = abs(abs(slope) - 6)
                if vel_err < 2:
                    print(f"  RESULT: GOOD — {slope:.1f}°/s")
                    results.append(("GOOD", video_name, slope))
                elif vel_err < 4:
                    print(f"  RESULT: PARTIAL — {slope:.1f}°/s")
                    results.append(("PARTIAL", video_name, slope))
                else:
                    print(f"  RESULT: POOR — {slope:.1f}°/s")
                    results.append(("POOR", video_name, slope))
            else:
                # Check unlocked tracks
                best_track = None
                best_err = VEL_TOLERANCE + 1
                for t in tracks:
                    if t.frames >= 30:
                        err = abs(abs(t.velocity) - 6)
                        if err < best_err:
                            best_err = err
                            best_track = t
                if best_track and best_err < VEL_TOLERANCE:
                    print(f"\n  No lock, best T{best_track.id}: vel={best_track.velocity:.1f}°/s, "
                          f"res={best_track.residual:.1f}°, f={best_track.frames}")
                    results.append(("PARTIAL", video_name, best_track.velocity))
                else:
                    print(f"\n  RESULT: NO LOCK")
                    results.append(("NO LOCK", video_name, 0))

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
