"""Step 1: the answer key, from CHIRLA's per-frame identity boxes.

Every annotated person's foot point goes through the same door rule the system uses (config.SideTracker),
so the true crossings and the detected ones are measured the same way. The identities are the
dataset's own: the same number is the same person on every day.
"""
import json
from collections import defaultdict

from config import EDGE, FPS, H, OUT_DIR, SideTracker, drop_door_flips, sequences

out = {"fps": FPS, "sequences": []}
for s in sequences():
    ann = json.load(open(s["ann"]))
    tracks = defaultdict(list)
    for key, people in ann.items():
        for p in people:
            tracks[p["id"]].append((int(key) - 1, p["BboxP"]))  # annotation frames start at 1
    events = []
    for pid, obs in tracks.items():
        obs.sort(key=lambda o: o[0])
        side = SideTracker(s["door"])
        for f, (x1, y1, x2, y2) in obs:
            if y2 > H - EDGE:
                continue
            kind = side.update((x1 + x2) / 2, y2)
            if kind:
                events.append({"frame": f, "time": f / FPS, "kind": kind, "person": pid})
    events.sort(key=lambda e: e["frame"])
    events, n_drop = drop_door_flips(events, "person")
    n_frames = max(int(k) for k in ann)
    out["sequences"].append({"seq": s["seq"], "day": s["day"], "start": s["start"].isoformat(),
                             "n_frames": n_frames, "people_seen": sorted(tracks), "events": events})
    n_in = sum(e["kind"] == "enter" for e in events)
    print(f"{s['seq']} {s['start']}  {n_frames / FPS / 60:4.1f} min  {len(tracks):2d} people seen  "
          f"{n_in} entries, {len(events) - n_in} exits, by {len({e['person'] for e in events})} people "
          f"({n_drop} doorway flips dropped)")

OUT_DIR.mkdir(parents=True, exist_ok=True)
json.dump(out, open(OUT_DIR / "gt.json", "w"), indent=1)
