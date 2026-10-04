#!/usr/bin/env python3
"""Where a video jumps, halts or freezes, measured — for LTX clips joined into one
(storyboards, chains), where those are the faults that show (#48).

Each pair of frames (grey, 320 px wide) gets an optical flow (Farneback, OpenCV):
how far the picture moves, and what the movement does not explain — the next
frame against this one moved by the flow. From these:

  jump    what the flow does not explain, far above the second around it, in one
          to three frames: a background that snaps over, a clip that reaches its
          end picture only in its last frame, a join that does not fit. Planned
          cuts (--cuts) are left out.
  halt    the movement falls by more than 70 % within half a second and stays down
          for one: someone stopping dead instead of running out.
  freeze  next to no movement for a second and more: a clip done early, waiting
          at its end picture. Nobody stands that still.

A dissolve — someone fading out at one place and in at another, a ghost — changes
each frame only a little; it does not show here. That takes a look at the
pictures (storyboard's --vision).

    videocheck.py VIDEO [--cuts F,...] [--joins F,...] [--json]

--joins (the first frame of each later clip) names the join an event lies at.
"""
import argparse
import json
import subprocess
import sys

import numpy as np

W, H = 320, 176
WIN = 6                       # frames per window for the movement curve (1/4 s at 24 fps)


def frames(path):
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-vf", f"scale={W}:{H},format=gray",
                          "-f", "rawvideo", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, H, W)


def flow_measures(x):
    """Per pair of frames: the mean movement (pixels at 320 wide) and the mean
    change the movement does not explain."""
    import cv2
    gx, gy = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    move, rest = [], []
    for a, b in zip(x[:-1], x[1:]):
        back = cv2.calcOpticalFlowFarneback(b, a, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        warped = cv2.remap(a, gx + back[..., 0], gy + back[..., 1], cv2.INTER_LINEAR)
        move.append(float(np.linalg.norm(back, axis=2).mean()))
        rest.append(float(np.abs(b.astype(np.float32) - warped.astype(np.float32)).mean()))
    return np.array(move), np.array(rest)


def events(move, rest, fps=24, cuts=()):
    found = []
    # jumps: unexplained change far above the second around it
    for i in range(len(rest)):
        if i + 1 in cuts:
            continue
        around = np.r_[rest[max(0, i - fps // 2):i], rest[i + 1:i + 1 + fps // 2]]
        base = float(np.median(around)) if len(around) else 0.0
        ratio = rest[i] / (base + 1e-3)
        if rest[i] >= 2.0 and ratio >= 2.5:
            found.append({"kind": "jump", "frame": i + 1, "strength": round(float(ratio), 1)})
    # merge a jump spread over neighbouring frames into one, the strongest
    merged = []
    for e in found:
        if merged and e["frame"] - merged[-1]["last"] <= 2:
            merged[-1]["last"] = e["frame"]
            if e["strength"] > merged[-1]["strength"]:
                merged[-1].update(frame=e["frame"], strength=e["strength"])
        else:
            merged.append({**e, "last": e["frame"]})
    # the movement curve in windows of 1/4 s; cuts start a new stretch
    curve = [float(move[k:k + WIN].mean()) for k in range(0, len(move) - WIN + 1, WIN)]
    cut_windows = {c // WIN for c in cuts}
    for k in range(len(curve) - 2 - 4):
        if any(w in cut_windows for w in range(k, k + 7)):
            continue
        high, low = curve[k], min(curve[k + 1:k + 3])
        after = curve[k + 1:k + 7]
        if high >= 0.18 and low <= 0.3 * high and max(after[1:]) <= 0.4 * high:
            merged.append({"kind": "halt", "frame": (k + 1) * WIN, "strength": round(high / max(low, 1e-3), 1)})
    still = 0
    for k, v in enumerate(curve + [1.0]):
        if v <= 0.06 and (k * WIN) // WIN not in cut_windows:
            still += 1
            continue
        if still * WIN >= fps:
            merged.append({"kind": "freeze", "frame": (k - still) * WIN, "seconds": round(still * WIN / fps, 2)})
        still = 0
    # a halt right before a freeze is one fault: keep both, they read together
    return sorted(merged, key=lambda e: e["frame"]), curve


def where(frame, joins):
    """'clip N' or 'join N/N+1' for a frame, from the first frames of the later clips."""
    if not joins:
        return ""
    clip = 1 + sum(frame >= j for j in joins)
    near = [j for j in joins if abs(frame - j) <= 3]
    return f"join {joins.index(near[0]) + 1}/{joins.index(near[0]) + 2}" if near else f"clip {clip}"


def describe(e, fps, joins):
    t = e["frame"] / fps
    at = where(e["frame"], joins)
    at = f", {at}" if at else ""
    if e["kind"] == "jump":
        return f"{t:6.2f} s (frame {e['frame']}{at}): JUMP, {e['strength']}x the change around it"
    if e["kind"] == "halt":
        return f"{t:6.2f} s (frame {e['frame']}{at}): HALT, the movement drops {e['strength']}x within 1/2 s"
    return f"{t:6.2f} s (frame {e['frame']}{at}): FREEZE, {e['seconds']} s nearly without movement"


def check(path, fps=24, cuts=(), joins=()):
    x = frames(path)
    move, rest = flow_measures(x)
    found, curve = events(move, rest, fps, set(cuts))
    return found, curve, len(x)


def main():
    ap = argparse.ArgumentParser(description="where a video jumps, halts or freezes")
    ap.add_argument("video")
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--cuts", default="", help="frames where a planned cut begins (left out)")
    ap.add_argument("--joins", default="", help="the first frame of each later clip")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    nums = lambda s: [int(v) for v in s.split(",") if v.strip()]
    found, curve, n = check(a.video, a.fps, nums(a.cuts), nums(a.joins))
    if a.json:
        print(json.dumps({"frames": n, "events": found, "movement": [round(v, 3) for v in curve]}))
        return
    print(f"{a.video}: {n} frames")
    for e in found:
        print("  " + describe(e, a.fps, nums(a.joins)))
    if not found:
        print("  no jump, halt or freeze")
    sys.exit(1 if found else 0)


if __name__ == "__main__":
    main()
