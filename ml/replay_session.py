#!/usr/bin/env python3
"""Replay a SecondHandAnalyzer JSONL session log.

Pull session-YYYYMMDD-HHMMSS.jsonl off the device via the Files app
(WatchSync → sessions → ...) and pass it to this tool.

  ./replay_session.py session-20260510-145300.jsonl

Outputs:
  - A summary: did classical lock? when? what was the score curve?
  - PNG: angle vs time (ML, classical, wall-clock-expected) — eyeball test
         is whether any line tracks the +6°/s wall-clock line.
  - PNG: stack score vs time — shows whether classical is anywhere near the
         activation threshold of 2.0.
"""

import json
import os
import sys

import cv2
import numpy as np


def load(path):
    frames = []
    events = []
    header = None
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            kind = rec.get("type")
            if kind == "header":
                header = rec
            elif kind == "frame":
                frames.append(rec)
            elif kind == "event":
                events.append(rec)
    return header, frames, events


def summarize(header, frames, events):
    print(f"\n=== {header.get('session', '?')} ===")
    print(f"frames logged: {len(frames)}")
    print(f"events logged: {len(events)}")
    if not frames:
        return
    elapsed = frames[-1]["t"]
    print(f"duration: {elapsed:.1f} s, avg rate: {len(frames)/elapsed:.1f} fps")

    # Source mix (which path produced the angle)
    src_count = {}
    for f in frames:
        s = f.get("angleSource", "")
        src_count[s] = src_count.get(s, 0) + 1
    print(f"angle source counts: {src_count}")

    # Model state evolution
    activated = [e for e in events if e.get("event") == "model_activated"]
    reset = [e for e in events if e.get("event") == "model_reset"]
    print(f"model_activated events: {len(activated)}")
    print(f"model_reset events: {len(reset)}")
    for e in activated[:3]:
        print(f"  ACTIVATED frame={e.get('frame')} v={e.get('velocity'):.2f}°/s "
              f"score={e.get('score'):.2f}")
    for e in reset[:3]:
        print(f"  RESET frame={e.get('frame')} prev_v={e.get('previousVelocity'):.2f}°/s")

    # Stack score history
    stacks = [e for e in events if e.get("event") == "stack"]
    if stacks:
        scores = [e.get("bestScore", 0) for e in stacks]
        vels = [e.get("bestVelocity", 0) for e in stacks]
        print(f"stack attempts: {len(stacks)}")
        print(f"  bestScore: min={min(scores):.2f} mean={np.mean(scores):.2f} "
              f"max={max(scores):.2f}")
        print(f"  bestVelocity: median={np.median(vels):+.2f}°/s")
        above = sum(1 for s in scores if s > 2.0)
        print(f"  attempts above activation threshold (2.0): {above}/{len(scores)}")

    # Per-frame angle behavior: is anything tracking +6°/s?
    ts = np.array([f["t"] for f in frames])
    ml = np.array([f.get("mlAngle", np.nan) for f in frames], dtype=float)
    pa = np.array([f.get("primaryAngle", np.nan) for f in frames], dtype=float)
    wall = np.array([f.get("wall", 0) for f in frames], dtype=float)
    expected = (wall * 6.0) % 360.0  # second hand position from wall clock

    # ML drift
    if np.isfinite(ml).sum() > 30:
        ml_err = ((ml - expected + 540) % 360) - 180
        print(f"ML angle vs wall: mean abs err = {np.nanmean(np.abs(ml_err)):.1f}°, "
              f"median |err| = {np.nanmedian(np.abs(ml_err)):.1f}°")
    if np.isfinite(pa).sum() > 30:
        pa_err = ((pa - expected + 540) % 360) - 180
        print(f"primary angle vs wall: mean abs err = {np.nanmean(np.abs(pa_err)):.1f}°, "
              f"median |err| = {np.nanmedian(np.abs(pa_err)):.1f}°")

    # Final drift
    last = frames[-1]
    if last.get("drift") is not None:
        print(f"final drift: {last['drift']:.1f} ± {last.get('uncertainty', 0):.1f} s/day")
    else:
        print("final drift: not computed (likely no classical lock)")
    print(f"offsets accumulated: {last.get('offsetsCount', 0)}")


def plot_angles(header, frames, out_path):
    if not frames:
        return
    ts = np.array([f["t"] for f in frames])
    wall = np.array([f.get("wall", 0) for f in frames], dtype=float)
    ml = np.array([f.get("mlAngle", np.nan) for f in frames], dtype=float)
    pa = np.array([f.get("primaryAngle", np.nan) for f in frames], dtype=float)
    expected = (wall * 6.0) % 360.0

    H, W = 720, 1400
    img = np.zeros((H, W, 3), dtype=np.uint8)
    pad = 30
    if ts.max() <= ts.min():
        return
    def x_of(t):
        return int(pad + (t - ts.min()) / (ts.max() - ts.min()) * (W - 2 * pad))
    def y_of(a):
        return int(pad + (a / 360.0) * (H - 2 * pad))

    # Expected (wall-clock predicted second hand position)
    for t, e in zip(ts, expected):
        cv2.circle(img, (x_of(t), y_of(e)), 1, (200, 200, 200), -1)
    # ML angle
    for t, a in zip(ts, ml):
        if np.isfinite(a):
            cv2.circle(img, (x_of(t), y_of(a)), 1, (255, 200, 100), -1)
    # Primary (= classical when locked)
    for t, a in zip(ts, pa):
        if np.isfinite(a):
            cv2.circle(img, (x_of(t), y_of(a)), 2, (0, 255, 0), -1)

    cv2.putText(img, f"angle vs time   gray=wall-clock expected, "
                f"orange=ML, green=primary (classical when locked)",
                (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)
    cv2.putText(img, f"y=angle 0-360   x=time {ts.max()-ts.min():.0f}s",
                (10, H - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
    cv2.imwrite(out_path, img)
    print(f"wrote {out_path}")


def plot_scores(header, events, out_path):
    stacks = [e for e in events if e.get("event") == "stack"]
    if not stacks:
        return
    H, W = 360, 1400
    img = np.zeros((H, W, 3), dtype=np.uint8)
    pad = 30
    n = len(stacks)
    scores = np.array([e.get("bestScore", 0) for e in stacks], dtype=float)
    if scores.max() <= 0:
        return
    s_max = max(scores.max(), 3.0)
    for i, s in enumerate(scores):
        x = int(pad + (i / max(n - 1, 1)) * (W - 2 * pad))
        y = int(H - pad - (s / s_max) * (H - 2 * pad))
        cv2.circle(img, (x, y), 2, (0, 255, 0), -1)
    # Activation line at score = 2.0
    y_thr = int(H - pad - (2.0 / s_max) * (H - 2 * pad))
    cv2.line(img, (pad, y_thr), (W - pad, y_thr), (0, 0, 255), 1)
    cv2.putText(img, f"stack bestScore over time   red line = activation threshold (2.0)",
                (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)
    cv2.imwrite(out_path, img)
    print(f"wrote {out_path}")


def main(argv):
    if len(argv) < 2:
        print("usage: replay_session.py <session.jsonl>")
        return
    path = argv[1]
    if not os.path.exists(path):
        print(f"file not found: {path}")
        return
    header, frames, events = load(path)
    if header is None:
        print("no header — file may be corrupt")
    summarize(header or {}, frames, events)
    base = os.path.splitext(os.path.basename(path))[0]
    out_dir = os.path.join(os.path.dirname(__file__), "diagnostics_replay")
    os.makedirs(out_dir, exist_ok=True)
    plot_angles(header or {}, frames,
                os.path.join(out_dir, f"{base}_angles.png"))
    plot_scores(header or {}, events,
                os.path.join(out_dir, f"{base}_scores.png"))


if __name__ == "__main__":
    main(sys.argv)
