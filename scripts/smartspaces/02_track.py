"""Step 2: detect and track people with YOLO11-pose on the door camera.

Saves, for every frame, each track's box, confidence and 17 COCO keypoints.
"""
import pickle
import sys
import time
from pathlib import Path

from ultralytics import YOLO

from config import CAMERA, OUT_DIR, SCENE_DIR, WEIGHTS

MODEL = WEIGHTS / "yolo11s-pose.pt"
IMGSZ = 960
CONF = 0.3
TRACKER = Path(__file__).resolve().parents[1] / "botsort_reid.yaml"  # ByteTrack swapped people crossing in the doorway
MAX_FRAMES = int(sys.argv[1]) if len(sys.argv) > 1 else None  # quick test: python 02_track.py 300

model = YOLO(str(MODEL))
video = SCENE_DIR / "videos" / f"{CAMERA}.mp4"
rows = []
t0 = time.time()
for f, r in enumerate(model.track(str(video), stream=True, imgsz=IMGSZ, conf=CONF, classes=[0],
                                  tracker=str(TRACKER), persist=True, verbose=False)):
    if r.boxes.id is not None:
        boxes = r.boxes.xyxy.numpy()
        ids = r.boxes.id.int().numpy()
        confs = r.boxes.conf.numpy()
        kpts = r.keypoints.data.numpy()  # (n, 17, 3): x, y, visibility
        for b, i, c, k in zip(boxes, ids, confs, kpts):
            rows.append((f, int(i), b, float(c), k))
    if f % 300 == 0:
        print(f"frame {f}  {f / max(time.time() - t0, 1e-6):.1f} fps", flush=True)
    if MAX_FRAMES and f + 1 >= MAX_FRAMES:
        break

OUT_DIR.mkdir(parents=True, exist_ok=True)
pickle.dump(rows, open(OUT_DIR / "tracks.pkl", "wb"))
print(f"{len(rows)} detections, {len({r[1] for r in rows})} tracks, {time.time() - t0:.0f} s")
