"""Step 3: door crossings, relative height and crops for every tracked person, in every recording.

A track whose foot point moves from the doorway side of the door line into the office is an entry;
the opposite is an exit.

Height without calibration: on a flat floor, a person's box height in pixels grows linearly with how
low their feet are in the image, h = a * (foot_y - horizon_y). Fitting a and horizon_y on everyone
walking near the door gives each person's height relative to the average (1.00 = average,
1.05 = 5% taller). Only boxes not cut by the image border count.
"""
import json
import pickle
import sys
from collections import defaultdict

import cv2
import numpy as np

from config import EDGE, FPS, H, OUT_DIR, W, SideTracker, sequences, stitch

WINDOW_S = 2.0      # crops and height are taken within this many seconds of the crossing
CROPS_PER_EVENT = 8
TRACK_DIR = sys.argv[1] if len(sys.argv) > 1 else "tracks"  # output of 02_track.py


def unclipped(box):
    x1, y1, x2, y2 = box
    return y1 > EDGE and y2 < H - EDGE and x1 > EDGE and x2 < W - EDGE


events, near_by_event = [], []
for s in sequences():
    rows = pickle.load(open(OUT_DIR / TRACK_DIR / f"{s['seq']}.pkl", "rb"))
    tracks = defaultdict(list)
    for f, tid, box, conf, kpts in rows:
        tracks[tid].append((f, box, conf, kpts))
    for obs in tracks.values():
        obs.sort(key=lambda o: o[0])
    tracks = stitch(tracks)  # the tracker breaks people in the doorway
    for tid, obs in tracks.items():
        side = SideTracker(s["door"])
        for f, box, conf, kpts in obs:
            x1, y1, x2, y2 = box
            if y2 > H - EDGE:  # feet out of frame
                continue
            kind = side.update((x1 + x2) / 2, y2)
            if kind:
                events.append({"seq": s["seq"], "day": s["day"], "track": tid, "frame": f, "time": f / FPS,
                               "kind": kind, "track_end": obs[-1][0] / FPS})
                near_by_event.append([o for o in obs if abs(o[0] - f) <= WINDOW_S * FPS])

# Fit h = a * (foot_y - horizon_y) on every unclipped box near a crossing, once per camera mount
fits = {}
for mount in {s["door"]["y"] for s in sequences()}:
    seqs = {s["seq"] for s in sequences() if s["door"]["y"] == mount}
    pts = np.array([(o[1][3], o[1][3] - o[1][1]) for e, near in zip(events, near_by_event) if e["seq"] in seqs
                    for o in near if unclipped(o[1])])
    slope, icpt = np.polyfit(pts[:, 0], pts[:, 1], 1)
    fits[mount] = (slope, -icpt / slope)
    print(f"mount with door at y {mount}: box height = {slope:.3f} * (foot_y - {-icpt / slope:.0f}) px, "
          f"from {len(pts)} boxes")
door_of = {s["seq"]: s["door"]["y"] for s in sequences()}

for e, near in zip(events, near_by_event):
    slope, horizon = fits[door_of[e["seq"]]]
    rel = [(o[1][3] - o[1][1]) / (slope * (o[1][3] - horizon)) for o in near if unclipped(o[1])]
    e["height_rel"] = float(np.median(rel)) if rel else None
    e["height_samples"] = len(rel)

# Crops: largest boxes first (closest to the camera, most pixels for face and clothes)
crop_dir = OUT_DIR / "crops"
crop_dir.mkdir(parents=True, exist_ok=True)
for old in crop_dir.glob("*.jpg"):
    old.unlink()
for s in sequences():
    need = defaultdict(list)
    for k, (e, near) in enumerate(zip(events, near_by_event)):
        if e["seq"] != s["seq"]:
            continue
        near = sorted(near, key=lambda o: -(o[1][3] - o[1][1]))
        picks = sorted(near[: CROPS_PER_EVENT * 3], key=lambda o: o[0])[::3][:CROPS_PER_EVENT]
        for o in picks:
            need[o[0]].append((k, o[1]))
        e["crops"] = []
    cap = cv2.VideoCapture(str(s["video"]))
    last = max(need) if need else -1
    f = 0
    while f <= last:
        ok, frame = cap.read()
        if not ok:
            break
        for k, box in need.get(f, []):
            x1, y1, x2, y2 = [int(round(v)) for v in box]
            name = f"e{k:04d}_{s['seq']}_f{f:05d}.jpg"
            cv2.imwrite(str(crop_dir / name), frame[max(y1, 0):min(y2, H), max(x1, 0):min(x2, W)])
            events[k]["crops"].append(name)
        f += 1

order = sorted(range(len(events)), key=lambda k: (events[k]["seq"], events[k]["frame"]))
json.dump([events[k] for k in order], open(OUT_DIR / "events.json", "w"), indent=1)
n_in = sum(e["kind"] == "enter" for e in events)
print(f"{len(events)} crossings ({n_in} entries, {len(events) - n_in} exits)")
