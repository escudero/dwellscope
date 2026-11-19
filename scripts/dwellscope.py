"""dwellscope: one camera at a door -> JSON history of who went in, who came out, and for how long.

Usage:
    python dwellscope.py VIDEO --door DOOR.json [--out history.json] [--video-out annotated.mp4]
    python dwellscope.py VIDEO --door DOOR.json --check door_check.png    # only draws the door line

DOOR.json is the only per-camera setting: the door line, as two points in image pixels, and any one
point inside the room (it tells which side is "inside"). Optional keys tune the door rule:
    {"line": [[200, 810], [760, 810]], "inside": [480, 1000]}
    {"line": ..., "inside": ..., "margin_px": 45, "start": "2026-11-19T19:01:00"}

The pipeline is the one in scripts/chirla/ (see the README), in one pass over one video:
  1. YOLO11-pose + BoT-SORT track every person.
  2. Tracks the tracker broke in the doorway are joined; a person's foot point crossing the door line
     by more than a margin is an entry or an exit. Relative height comes from box height against foot
     position, fitted on this video.
  3. Each crossing gets crops; SCRFD + ArcFace embed the face and CLIP-ReID the clothes and hair.
     Entry: face first, then the look of people seen earlier, else a new person. Exit: the best of the
     people inside on clothes, height and face.
  4. Entries are paired with the next exit of the same person into visits.
"""
import argparse
import json
import subprocess
import sys
from collections import defaultdict, deque
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TRACKER = Path(__file__).resolve().parent / "botsort_reid.yaml"

# Door rule (see scripts/chirla/config.py). Pixel values are for a 720 px tall image and scale with height.
MARGIN_PX_720 = 30    # pixels past the line before a person changes side
SMOOTH_S = 0.5        # the foot point is the median of the last 0.5 s: a box cut by someone in front jumps
DOOR_DWELL_S = 2.0    # a crossing undone by the same track within 2 s was someone standing in the doorway
EDGE = 4              # boxes this close to the bottom of the image have their feet cut off
LINK_S = 1.5          # a track starting this soon after another ended...
LINK_PX_720 = 120     # ...this close to where it ended continues it

# Identification (see scripts/chirla/04_identify.py)
WINDOW_S = 2.0        # crops and height come from within 2 s of the crossing
CROPS_PER_EVENT = 8
FACE_MIN_SCORE = 0.60
FACE_MATCH = 0.25
FACE_W = 0.5
HEIGHT_W = 3.5
NEW_PERSON = {"body": 0.85, "height": -0.07}
MODES = ["face+body+height", "face+body", "body+height", "body", "height", "face"]


def log(msg):
    print(msg, file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------------------------- door
class Door:
    """A door line between two points; `inside` is any point in the room."""

    def __init__(self, cfg, height):
        (x1, y1), (x2, y2) = cfg["line"]
        self.p, self.d = np.array([x1, y1], float), np.array([x2 - x1, y2 - y1], float)
        self.length = float(np.linalg.norm(self.d))
        if self.length == 0:
            sys.exit("door: the two points of 'line' are the same")
        side = self._cross(np.array(cfg["inside"], float) - self.p)
        if side == 0:
            sys.exit("door: the 'inside' point lies on the door line")
        self.sign = 1.0 if side > 0 else -1.0
        self.margin = float(cfg.get("margin_px", MARGIN_PX_720 * height / 720))
        self.line, self.inside = cfg["line"], cfg["inside"]

    def _cross(self, q):
        return self.d[0] * q[1] - self.d[1] * q[0]

    def where(self, u, v):
        """Signed distance to the line in pixels (> 0 inside) and position along it (0..1 within the door)."""
        q = np.array([u, v], float) - self.p
        return self.sign * self._cross(q) / self.length, float(q @ self.d) / self.length ** 2


class SideTracker:
    """Which side of the door a track is on; returns 'enter' or 'exit' when it crosses."""

    def __init__(self, door, smooth):
        self.door, self.state, self.recent = door, None, deque(maxlen=smooth)

    def update(self, u, v):
        dist, t = self.door.where(u, v)
        self.recent.append(dist)
        dist = sorted(self.recent)[len(self.recent) // 2]
        if not 0 < t < 1 and dist < 0:  # outside the room but not in front of the door: someone passing by
            return None
        m = self.door.margin
        if self.state is None:  # a new track decides its side only once it is clearly off the line
            if abs(dist) > m:
                self.state = dist > 0
            return None
        if not self.state and dist > m:
            self.state = True
            return "enter"
        if self.state and dist < -m:
            self.state = False
            return "exit"
        return None


def stitch(tracks, fps, link_px):
    """Join tracks the tracker broke in the doorway: {tid: [(frame, box)]} -> {chain id: observations}."""
    foot = lambda o: ((o[1][0] + o[1][2]) / 2, o[1][3])
    chain_ends, chains, chain_of = {}, {}, {}
    for t in sorted(tracks, key=lambda t: tracks[t][0][0]):
        obs = tracks[t]
        f0, (u0, v0) = obs[0][0], foot(obs[0])
        best, best_d = None, link_px
        for c, (fe, (ue, ve)) in chain_ends.items():
            d = ((u0 - ue) ** 2 + (v0 - ve) ** 2) ** 0.5
            if 0 < f0 - fe <= LINK_S * fps and d < best_d:
                best, best_d = c, d
        c = best if best is not None else t
        chains.setdefault(c, []).extend(obs)
        chain_of[t] = c
        chain_ends[c] = (obs[-1][0], foot(obs[-1]))
    return chains, chain_of


def drop_door_flips(events):
    """Drop pairs of opposite crossings by the same track less than DOOR_DWELL_S apart."""
    drop, by = set(), defaultdict(list)
    for k, e in enumerate(events):
        by[e["track"]].append(k)
    for ks in by.values():
        for a, b in zip(ks, ks[1:]):
            ea, eb = events[a], events[b]
            if a not in drop and ea["kind"] != eb["kind"] and eb["time"] - ea["time"] < DOOR_DWELL_S:
                drop |= {a, b}
    return [e for k, e in enumerate(events) if k not in drop], len(drop)


# ------------------------------------------------------------------------------------------ pipeline
def track(video, weights, imgsz, conf):
    from ultralytics import YOLO

    model = YOLO(str(weights / "yolo11s-pose.pt"))
    tracks = defaultdict(list)
    for f, r in enumerate(model.track(str(video), stream=True, imgsz=imgsz, conf=conf, classes=[0],
                                      tracker=str(TRACKER), persist=True, verbose=False)):
        if r.boxes.id is not None:
            for b, i in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.id.int().cpu().numpy()):
                tracks[int(i)].append((f, b))
        if f % 500 == 0:
            log(f"  tracking: frame {f}")
    return tracks


def find_crossings(tracks, door, fps, W, H):
    chains, chain_of = stitch(tracks, fps, LINK_PX_720 * H / 720)
    smooth = max(1, round(SMOOTH_S * fps))
    events, near = [], []
    for cid, obs in chains.items():
        side = SideTracker(door, smooth)
        for f, (x1, y1, x2, y2) in obs:
            if y2 > H - EDGE:
                continue
            kind = side.update((x1 + x2) / 2, y2)
            if kind:
                events.append({"track": cid, "frame": f, "time": f / fps, "kind": kind})
                near.append([o for o in obs if abs(o[0] - f) <= WINDOW_S * fps])
    order = sorted(range(len(events)), key=lambda k: events[k]["frame"])
    events, near = [events[k] for k in order], [near[k] for k in order]

    # Relative height: on a flat floor box height grows linearly with how low the feet are in the image
    unclipped = lambda b: b[1] > EDGE and b[3] < H - EDGE and b[0] > EDGE and b[2] < W - EDGE
    pts = np.array([(b[3], b[3] - b[1]) for ns in near for _, b in ns if unclipped(b)])
    fit = None
    if len(pts) >= 10 and np.ptp(pts[:, 0]) > 0:
        slope, icpt = np.polyfit(pts[:, 0], pts[:, 1], 1)
        if slope > 0:
            fit = (slope, -icpt / slope)
    for e, ns in zip(events, near):
        rel = [(b[3] - b[1]) / (fit[0] * (b[3] - fit[1])) for _, b in ns
               if fit and unclipped(b) and b[3] > fit[1]] if fit else []
        e["height_rel"] = float(np.median(rel)) if rel else None
    return events, near, chain_of


def grab_crops(video, near, H, W):
    """Up to CROPS_PER_EVENT crops per crossing, largest boxes first (most pixels for face and clothes)."""
    need, crops = defaultdict(list), [[] for _ in near]
    for k, ns in enumerate(near):
        ns = sorted(ns, key=lambda o: -(o[1][3] - o[1][1]))
        for f, box in sorted(ns[: CROPS_PER_EVENT * 3], key=lambda o: o[0])[::3][:CROPS_PER_EVENT]:
            need[f].append((k, box))
    cap, last, f = cv2.VideoCapture(str(video)), max(need, default=-1), 0
    while f <= last:
        ok, frame = cap.read()
        if not ok:
            break
        for k, box in need.get(f, []):
            x1, y1, x2, y2 = [int(round(v)) for v in box]
            crops[k].append(frame[max(y1, 0):min(y2, H), max(x1, 0):min(x2, W)].copy())
        f += 1
    return crops


def embed(crops, weights):
    import onnxruntime as ort
    from insightface.app import FaceAnalysis

    face_app = FaceAnalysis(name="buffalo_l", root=str(weights / "insightface"),
                            allowed_modules=["detection", "recognition"], providers=["CPUExecutionProvider"])
    face_app.prepare(ctx_id=-1, det_size=(640, 640))
    reid = ort.InferenceSession(str(weights / "person_vit_clip_reid.onnx"), providers=["CPUExecutionProvider"])
    mean, std = np.array([0.485, 0.456, 0.406], np.float32), np.array([0.229, 0.224, 0.225], np.float32)
    faces, bodies, scores = [], [], []
    for imgs in crops:
        imgs = [im for im in imgs if im.shape[0] > 20 and im.shape[1] > 10]
        if imgs:
            batch = np.stack([((cv2.resize(im, (128, 256))[:, :, ::-1] / 255.0 - mean) / std).transpose(2, 0, 1)
                              for im in imgs]).astype(np.float32)
            v = reid.run(None, {"input": batch})[0].mean(0)
            bodies.append(v / np.linalg.norm(v))
        else:
            bodies.append(np.zeros(512, np.float32))
        best, best_score = None, 0.0
        for im in imgs:
            top = im[: max(im.shape[0] // 2, 1)]
            scale = 640 / max(top.shape[:2])
            if scale > 1:
                top = cv2.resize(top, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            for face in face_app.get(top):
                if face.det_score > best_score:
                    best, best_score = face.normed_embedding, float(face.det_score)
        faces.append(best if best is not None and best_score >= FACE_MIN_SCORE else np.zeros(512, np.float32))
        scores.append(best_score)
    return np.array(faces), np.array(bodies), np.array(scores)


def identify(events, faces, bodies, mode):
    use_face, use_body, use_height = "face" in mode, "body" in mode, "height" in mode
    new_t = NEW_PERSON["body"] if use_body else NEW_PERSON["height"]
    has_face = np.linalg.norm(faces, axis=1) > 0

    class Person:
        def __init__(self, pid):
            self.pid, self.faces, self.bodies, self.heights = pid, [], [], []
            self.inside, self.since = False, None

    def face_sim(p, k):
        if use_face and p.faces and has_face[k]:
            return max(float(faces[k] @ f) for f in p.faces)
        return None

    def look_sim(p, e):
        k, s, used = e["k"], 0.0, False
        if use_body and p.bodies:
            s, used = s + max(float(bodies[k] @ b) for b in p.bodies), True
        if use_height and p.heights and e["height_rel"] is not None:
            s, used = s - HEIGHT_W * abs(e["height_rel"] - float(np.median(p.heights))), True
        return s if used else None

    def who_enters(cands, e):
        by_face = [(s, p) for p in cands if (s := face_sim(p, e["k"])) is not None]
        if by_face:
            s, p = max(by_face, key=lambda t: t[0])
            if s >= FACE_MATCH:
                return p, "face", s
        by_look = [(s, p) for p in cands if (s := look_sim(p, e)) is not None]
        if by_look:
            s, p = max(by_look, key=lambda t: t[0])
            if s >= new_t:
                return p, "look", s
        return None, "new", None

    def who_exits(cands, e):
        scored = []
        for p in cands:
            s, used = 0.0, False
            if (ls := look_sim(p, e)) is not None:
                s, used = s + ls, True
            if (fs := face_sim(p, e["k"])) is not None:
                s, used = s + (FACE_W if (use_body or use_height) else 1.0) * fs, True
            scored.append((s if used else -np.inf, p))
        s, p = max(scored, key=lambda t: t[0])
        if not np.isfinite(s):
            return min(cands, key=lambda q: q.since), "longest inside", None
        return p, "look" if len(cands) > 1 else "only one inside", s

    people = []
    for e in events:
        inside = [q for q in people if q.inside]
        if e["kind"] == "enter" or not inside:
            cands = [q for q in people if not q.inside] if e["kind"] == "enter" else people
            p, how, s = who_enters(cands, e)
            if p is None:
                p = Person(len(people) + 1)
                people.append(p)
        else:
            p, how, s = who_exits(inside, e)
        p.inside, p.since = e["kind"] == "enter", e["time"]
        p.bodies.append(bodies[e["k"]])
        if has_face[e["k"]]:
            p.faces.append(faces[e["k"]])
        if e["height_rel"] is not None:
            p.heights.append(e["height_rel"])
        e.update(person=p.pid, identified_by=how, score=None if s is None else round(float(s), 3),
                 face_found=bool(has_face[e["k"]]))
    return len(people)


def history(events, duration, start):
    """Pair each entry with the next exit of the same person; flag what a door log cannot close."""
    stamp = lambda t: (start + timedelta(seconds=t)).isoformat(timespec="seconds") if start else None
    visits, open_, warnings = [], {}, []
    for e in events:
        p = e["person"]
        if e["kind"] == "enter":
            if p in open_:
                warnings.append(f"person {p}: two entries in a row (at {open_[p]['enter_s']} s and {e['time']:.1f} s); "
                                "an exit was probably missed")
                visits.append(open_.pop(p))
            open_[p] = {"person": p, "enter_s": round(e["time"], 1), "exit_s": None, "duration_s": None}
        elif p in open_:
            v = open_.pop(p)
            v.update(exit_s=round(e["time"], 1), duration_s=round(e["time"] - v["enter_s"], 1))
            visits.append(v)
        else:
            warnings.append(f"person {p}: exit at {e['time']:.1f} s with no entry (already inside when the video started)")
            visits.append({"person": p, "enter_s": None, "exit_s": round(e["time"], 1), "duration_s": None})
    for p, v in open_.items():
        warnings.append(f"person {p}: entered at {v['enter_s']} s and was still inside when the video ended")
        visits.append(v)
    visits.sort(key=lambda v: (v["enter_s"] if v["enter_s"] is not None else -1, v["person"]))
    for v in visits:
        v["enter_at"], v["exit_at"] = (stamp(v["enter_s"]) if v["enter_s"] is not None else None,
                                       stamp(v["exit_s"]) if v["exit_s"] is not None else None)
    people = defaultdict(lambda: {"visits": 0, "time_inside_s": 0.0})
    for v in visits:
        people[v["person"]]["visits"] += 1
        people[v["person"]]["time_inside_s"] += v["duration_s"] or 0.0
    people = [{"person": p, "visits": d["visits"], "time_inside_s": round(d["time_inside_s"], 1)}
              for p, d in sorted(people.items())]
    return visits, people, warnings


# ------------------------------------------------------------------------------------------- drawing
def fonts(H):
    from PIL import ImageFont

    path = "/usr/share/fonts/truetype/dejavu/DejaVuSans{}.ttf"
    sc = H / 1080
    f = lambda size, b=False: ImageFont.truetype(path.format("-Bold" if b else ""), max(10, round(size * sc)))
    return {"label": f(30, True), "hud": f(36, True), "big": f(52, True), "small": f(27), "row": f(30)}


def draw_door(d, door, H):
    w = max(2, round(4 * H / 1080))
    d.line([tuple(door.line[0]), tuple(door.line[1])], fill=(255, 60, 60, 230), width=w)
    x, y = door.inside
    r = max(4, round(10 * H / 1080))
    d.ellipse([x - r, y - r, x + r, y + r], fill=(60, 200, 90, 230))


def check_door(video, door, out):
    from PIL import Image, ImageDraw

    cap = cv2.VideoCapture(str(video))
    ok, frame = cap.read()
    if not ok:
        sys.exit(f"cannot read {video}")
    H = frame.shape[0]
    img = Image.fromarray(frame[:, :, ::-1])
    d = ImageDraw.Draw(img, "RGBA")
    draw_door(d, door, H)
    F = fonts(H)
    d.text((door.inside[0] + 16, door.inside[1] - 16), "dentro", font=F["label"], fill=(60, 200, 90))
    img.save(out)
    log(f"wrote {out}: red = door line, green = the 'inside' point")


COLORS = [(230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200), (245, 130, 48), (145, 30, 180),
          (70, 240, 240), (240, 50, 230), (210, 245, 60), (250, 190, 212), (0, 128, 128), (170, 110, 40)]
HOW = {"face": "rosto", "look": "roupa e altura", "new": "pessoa nova", "only one inside": "único lá dentro",
       "longest inside": "há mais tempo dentro"}


def fmt_time(s):
    s = int(round(s))
    return f"{s // 60}:{s % 60:02d}"


def render(video, out, tracks, chain_of, events, visits, door, fps, W, H, start, speed):
    from PIL import Image, ImageDraw

    F = fonts(H)
    sc = H / 1080
    by_frame = defaultdict(list)
    for tid, obs in tracks.items():
        for f, box in obs:
            by_frame[f].append((chain_of[tid], box))
    marks, events_at = defaultdict(list), defaultdict(list)
    for e in events:
        marks[e["track"]].append((e["frame"], e["person"]))
        events_at[e["frame"]].append(e)
    label = lambda c, f: min(marks[c], key=lambda m: abs(m[0] - f))[1] if marks.get(c) else None
    color = lambda p: COLORS[(p - 1) % len(COLORS)]

    cap = cv2.VideoCapture(str(video))
    ff = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
                           "-s", f"{W}x{H}", "-r", str(fps), "-i", "-", "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
                           "-c:v", "libx264", "-crf", "21", "-pix_fmt", "yuv420p", str(out)], stdin=subprocess.PIPE)
    inside, callouts, f = set(), [], 0
    hud_h, lab_h = round(64 * sc), round(38 * sc)
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        for e in events_at.get(f, []):
            p = e["person"]
            if e["kind"] == "enter":
                inside.add(p)
                txt = f"ENTROU  Pessoa {p:02d}  · {HOW[e['identified_by']]}"
            else:
                inside.discard(p)
                txt = f"SAIU  Pessoa {p:02d}  · {HOW[e['identified_by']]}"
            callouts.append((f + 3 * fps, txt, color(p)))
        if f % speed == 0:
            img = Image.fromarray(frame[:, :, ::-1])
            d = ImageDraw.Draw(img, "RGBA")
            draw_door(d, door, H)
            for c, (x1, y1, x2, y2) in by_frame.get(f, []):
                p = label(c, f)
                col = color(p) if p else (200, 200, 200)
                d.rectangle([x1, y1, x2, y2], outline=col + (255,), width=max(2, round(3 * sc)))
                if p:
                    t = f"Pessoa {p:02d}"
                    ty = max(y1 - lab_h, hud_h + 6)
                    d.rectangle([x1, ty, x1 + d.textlength(t, font=F["label"]) + 10, ty + lab_h], fill=col + (220,))
                    d.text((x1 + 5, ty + 2), t, font=F["label"], fill=(0, 0, 0))
            d.rectangle([0, 0, W, hud_h], fill=(0, 0, 0, 170))
            d.text((14, 9 * sc), f"Sala restrita · dentro agora: {len(inside)}", font=F["hud"], fill=(255, 255, 255))
            t = (start + timedelta(seconds=f / fps)).strftime("%d/%m/%Y %H:%M:%S") if start else fmt_time(f / fps)
            t += f" · {speed}x" if speed > 1 else ""
            d.text((W - 14 - d.textlength(t, font=F["hud"]), 9 * sc), t, font=F["hud"], fill=(255, 255, 255))
            callouts = [c for c in callouts if c[0] > f]
            for i, (_, txt, col) in enumerate(callouts[-3:]):
                y, bh = hud_h + 16 * sc + i * 56 * sc, 46 * sc
                d.rectangle([10, y, 22 + d.textlength(txt, font=F["label"]), y + bh], fill=(0, 0, 0, 190))
                d.rectangle([10, y, 16, y + bh], fill=col + (255,))
                d.text((22, y + 5 * sc), txt, font=F["label"], fill=(255, 255, 255))
            ff.stdin.write(np.asarray(img).tobytes())
        f += 1

    # Closing card: the visit log
    card = Image.new("RGB", (W, H), (18, 18, 22))
    d = ImageDraw.Draw(card)
    x0 = 60 * sc
    d.text((x0, 50 * sc), "Registro da sala: entradas e saídas", font=F["big"], fill=(255, 255, 255))
    y = 150 * sc
    cols = [0, 330, 530, 730]
    for dx, t in zip(cols, ("pessoa", "entrou", "saiu", "tempo dentro")):
        d.text((x0 + dx * sc, y), t, font=F["small"], fill=(150, 150, 150))
    y += 50 * sc
    for v in visits:
        if y > H - 60 * sc:
            d.text((x0, y), "…", font=F["row"], fill=(230, 230, 230))
            break
        cells = (f"Pessoa {v['person']:02d}", fmt_time(v["enter_s"]) if v["enter_s"] is not None else "?",
                 fmt_time(v["exit_s"]) if v["exit_s"] is not None else "?",
                 f"{v['duration_s']:.1f} s" if v["duration_s"] is not None else "—")
        for i, (dx, t) in enumerate(zip(cols, cells)):
            d.text((x0 + dx * sc, y), t, font=F["row"], fill=color(v["person"]) if i == 0 else (230, 230, 230))
        y += 46 * sc
    arr = np.asarray(card).tobytes()
    for _ in range(round(fps * 7)):
        ff.stdin.write(arr)
    ff.stdin.close()
    ff.wait()


# ---------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="Door camera -> JSON history of entries, exits and time inside.")
    ap.add_argument("video", type=Path)
    ap.add_argument("--door", type=Path, required=True, help="JSON with 'line' [[x1,y1],[x2,y2]] and 'inside' [x,y]")
    ap.add_argument("--out", type=Path, help="history JSON (default: print to stdout)")
    ap.add_argument("--video-out", type=Path, help="also write an annotated video here")
    ap.add_argument("--check", type=Path, help="only draw the door on the first frame into this image, and stop")
    ap.add_argument("--start", help="wall-clock time of the first frame, ISO 8601 (overrides 'start' in --door)")
    ap.add_argument("--cues", choices=MODES, default="face+body+height", help="which cues name people")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--speed", type=int, default=1, help="annotated video: keep one frame in N")
    ap.add_argument("--weights", type=Path, default=ROOT / "weights")
    a = ap.parse_args()

    cfg = json.loads(a.door.read_text())
    cap = cv2.VideoCapture(str(a.video))
    if not cap.isOpened():
        sys.exit(f"cannot open {a.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    door = Door(cfg, H)
    if a.check:
        check_door(a.video, door, a.check)
        return
    start = a.start or cfg.get("start")
    start = datetime.fromisoformat(start) if start else None

    log(f"{a.video.name}: {W}x{H}, {fps:g} fps, {n} frames; door margin {door.margin:.0f} px")
    tracks = track(a.video, a.weights, a.imgsz, a.conf)
    for obs in tracks.values():
        obs.sort(key=lambda o: o[0])
    events, near, chain_of = find_crossings(tracks, door, fps, W, H)
    for k, e in enumerate(events):
        e["k"] = k
    log(f"  {len(tracks)} tracks, {len(events)} door crossings")
    if events:
        faces, bodies, _ = embed(grab_crops(a.video, near, H, W), a.weights)
    else:
        faces = bodies = np.zeros((0, 512), np.float32)
    events, n_flips = drop_door_flips(events)
    n_people = identify(events, faces, bodies, a.cues)
    duration = n / fps
    visits, people, warnings = history(events, duration, start)

    stamp = lambda t: (start + timedelta(seconds=t)).isoformat(timespec="seconds") if start else None
    result = {
        "video": {"file": str(a.video), "width": W, "height": H, "fps": fps, "frames": n,
                  "duration_s": round(duration, 2), "start": start.isoformat() if start else None},
        "door": {"line": door.line, "inside": door.inside, "margin_px": round(door.margin, 1)},
        "cues": a.cues,
        "events": [{"time_s": round(e["time"], 2), "at": stamp(e["time"]), "frame": e["frame"], "kind": e["kind"],
                    "person": e["person"], "identified_by": e["identified_by"], "score": e["score"],
                    "face_found": e["face_found"]} for e in events],
        "visits": visits,
        "people": people,
        "warnings": warnings,
    }
    text = json.dumps(result, indent=1, ensure_ascii=False)
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(text)
        log(f"wrote {a.out}")
    else:
        print(text)
    log(f"  {len(events)} crossings after dropping {n_flips} doorway flips; {n_people} people; "
        f"{len(warnings)} warnings")

    if a.video_out:
        a.video_out.parent.mkdir(parents=True, exist_ok=True)
        render(a.video, a.video_out, tracks, chain_of, events, visits, door, fps, W, H, start, max(1, a.speed))
        log(f"wrote {a.video_out}")


if __name__ == "__main__":
    main()
