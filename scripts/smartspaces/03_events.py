"""Step 3: door crossings, height and crops for every tracked person.

The foot point (bottom of the box) goes to the floor through the camera calibration.
A track that moves from the corridor side of the door line to the room side is an entry;
the opposite is an exit. Height comes from the top of the box over that floor point.
"""
import json
import pickle
from collections import defaultdict

import cv2
import numpy as np

from config import CAMERA, FPS, OUT_DIR, SCENE_DIR, SideTracker, height_from_head, image_to_ground, load_camera

WINDOW_S = 2.0      # crops and height are taken within this many seconds of the crossing
CROPS_PER_EVENT = 8
EDGE = 4            # boxes touching the image border by this many pixels are cut off: no height from them

P = load_camera()
rows = pickle.load(open(OUT_DIR / "tracks.pkl", "rb"))
W, H = 1920, 1080

tracks = defaultdict(list)
for f, tid, box, conf, kpts in rows:
    tracks[tid].append((f, box, conf, kpts))

events = []
for tid, obs in tracks.items():
    obs.sort(key=lambda o: o[0])
    side = SideTracker()
    for f, box, conf, kpts in obs:
        x1, y1, x2, y2 = box
        if y2 > H - EDGE:  # feet out of frame: the floor point would be wrong
            continue
        gx, gy = image_to_ground(P, (x1 + x2) / 2, y2)
        kind = side.update(gx, gy)
        if kind:
            events.append({"track": tid, "frame": f, "time": f / FPS, "kind": kind,
                           "track_end": obs[-1][0] / FPS})

# Height and crop frames for each event
need = defaultdict(list)  # frame -> [(event index, box)]
for k, e in enumerate(events):
    near = [o for o in tracks[e["track"]] if abs(o[0] - e["frame"]) <= WINDOW_S * FPS]
    heights = []
    for f, box, conf, kpts in near:
        x1, y1, x2, y2 = box
        if y1 < EDGE or y2 > H - EDGE or x1 < EDGE or x2 > W - EDGE:
            continue
        gx, gy = image_to_ground(P, (x1 + x2) / 2, y2)
        heights.append(height_from_head(P, gx, gy, y1))
    e["height_m"] = float(np.median(heights)) if heights else None
    e["height_samples"] = len(heights)
    # Largest boxes first: closest to the camera, most pixels for face and clothing
    near.sort(key=lambda o: -(o[1][3] - o[1][1]))
    picks = sorted(near[: CROPS_PER_EVENT * 3], key=lambda o: o[0])[:: 3][:CROPS_PER_EVENT]
    for f, box, conf, kpts in picks:
        need[f].append((k, box))
    e["crops"] = []

crop_dir = OUT_DIR / "crops"
crop_dir.mkdir(parents=True, exist_ok=True)
for old in crop_dir.glob("*.jpg"):
    old.unlink()
cap = cv2.VideoCapture(str(SCENE_DIR / "videos" / f"{CAMERA}.mp4"))
f = 0
last = max(need) if need else -1
while f <= last:
    ok, frame = cap.read()
    if not ok:
        break
    for k, box in need.get(f, []):
        x1, y1, x2, y2 = [int(round(v)) for v in box]
        crop = frame[max(y1, 0):min(y2, H), max(x1, 0):min(x2, W)]
        name = f"e{k:03d}_f{f:05d}.jpg"
        cv2.imwrite(str(crop_dir / name), crop)
        events[k]["crops"].append(name)
    f += 1

events.sort(key=lambda e: e["frame"])
json.dump(events, open(OUT_DIR / "events.json", "w"), indent=1)
n_in = sum(e["kind"] == "enter" for e in events)
print(f"{len(events)} crossings ({n_in} entries, {len(events) - n_in} exits) from {len(tracks)} tracks")
