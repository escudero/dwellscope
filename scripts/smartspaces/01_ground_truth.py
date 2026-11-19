"""Step 1: ground truth from the synthetic scene's 3D boxes.

For every person: door crossings (enter/exit with the frame), time inside the room,
true height, and whether Camera_21 could see them at the crossing.
"""
import json

from config import CAMERA, FPS, OUT_DIR, SCENE_DIR, SideTracker, in_room_x, DOOR_Y

gt = json.load(open(SCENE_DIR / "ground_truth.json"))
n_frames = len(gt)

people = {}
for f in range(n_frames):
    for o in gt[str(f)]:
        p = people.setdefault(o["object id"], {"frames": {}, "heights": []})
        x, y, _ = o["3d location"]
        p["frames"][f] = (x, y, CAMERA in o["2d bounding box visible"])
        p["heights"].append(o["3d bounding box scale"][2])

events, persons = [], {}
for pid, p in sorted(people.items()):
    side = SideTracker()
    inside_frames = 0
    for f in sorted(p["frames"]):
        x, y, visible = p["frames"][f]
        inside_frames += in_room_x(x) and y > DOOR_Y
        kind = side.update(x, y)
        if kind:
            events.append({"frame": f, "time": f / FPS, "kind": kind, "person": pid, "visible": visible})
    hs = sorted(p["heights"])
    persons[pid] = {"height_m": hs[len(hs) // 2], "seconds_inside": inside_frames / FPS,
                    "starts_inside": side.state is not None and (p["frames"][min(p["frames"])][1] > DOOR_Y
                                                                  and in_room_x(p["frames"][min(p["frames"])][0]))}

visitors = {pid for pid, v in persons.items() if v["seconds_inside"] > 0}
out = {"n_frames": n_frames, "fps": FPS, "events": events,
       "persons": {pid: persons[pid] for pid in sorted(visitors)},
       "all_heights": {pid: v["height_m"] for pid, v in persons.items()}}
OUT_DIR.mkdir(parents=True, exist_ok=True)
json.dump(out, open(OUT_DIR / "gt.json", "w"), indent=1)

print(f"{len(events)} door crossings by {len(visitors)} people "
      f"({sum(e['kind'] == 'enter' for e in events)} entries, {sum(e['kind'] == 'exit' for e in events)} exits)")
for pid in sorted(visitors):
    v = persons[pid]
    print(f"  person {pid:2d}: {v['height_m']:.2f} m, {v['seconds_inside']:.1f} s inside, "
          f"starts inside: {v['starts_inside']}")
