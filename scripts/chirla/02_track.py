"""Step 2: detect and track people with YOLO11-pose on every camera 1 recording.

Runs the recordings in parallel (one process each) and saves, per recording, each track's box,
confidence and 17 COCO keypoints. Uses the GPU when there is one: on an NVIDIA L4 (3 workers) all ten
recordings take about 9 min; on a 16-thread CPU the same run was on course for about 8 hours.

Usage: python 02_track.py [workers] [imgsz] [conf] [output folder]

"""
import pickle
import sys
import time
from multiprocessing import Pool
from pathlib import Path

from config import OUT_DIR, WEIGHTS, sequences

MODEL = WEIGHTS / "yolo11s-pose.pt"
IMGSZ = int(sys.argv[2]) if len(sys.argv) > 2 else 640  # people at the door are 150 px or taller
CONF = float(sys.argv[3]) if len(sys.argv) > 3 else 0.3
TRACKER = Path(__file__).resolve().parents[1] / "botsort_reid.yaml"
WORKERS = int(sys.argv[1]) if len(sys.argv) > 1 else 4
THREADS = 4
TRACK_DIR = sys.argv[4] if len(sys.argv) > 4 else "tracks"


def track(s):
    import torch
    torch.set_num_threads(THREADS)
    from ultralytics import YOLO

    out = OUT_DIR / TRACK_DIR / f"{s['seq']}.pkl"
    if out.exists():
        return s["seq"], "cached"
    model = YOLO(str(MODEL))
    rows, t0 = [], time.time()
    for f, r in enumerate(model.track(str(s["video"]), stream=True, imgsz=IMGSZ, conf=CONF, classes=[0],
                                      tracker=str(TRACKER), persist=True, verbose=False)):
        if r.boxes.id is not None:
            bx, kp = r.boxes.cpu(), r.keypoints.cpu()
            for b, i, c, k in zip(bx.xyxy.numpy(), bx.id.int().numpy(), bx.conf.numpy(), kp.data.numpy()):
                rows.append((f, int(i), b, float(c), k))
    out.parent.mkdir(parents=True, exist_ok=True)
    pickle.dump(rows, open(out, "wb"))
    return s["seq"], f"{f + 1} frames, {len({r[1] for r in rows})} tracks, {time.time() - t0:.0f} s"


if __name__ == "__main__":
    seqs = sorted(sequences(), key=lambda s: -s["video"].stat().st_size)  # longest first
    with Pool(WORKERS) as pool:
        for seq, msg in pool.imap_unordered(track, seqs):
            print(seq, msg, flush=True)
