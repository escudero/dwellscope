"""Step 4: who is each crossing, across all recordings and days?

Cues per crossing:
- face: SCRFD finds it, ArcFace embeds it (InsightFace buffalo_l). Seen at entries, rarely at exits.
- body: CLIP-ReID vector of clothes and hair. CHIRLA participants changed clothes between sessions,
  so a person's body vectors are only compared within the same day.
- height: the relative height from step 3.

Rules, in the order the idea states them:
- Entry (candidates: people outside the office). If a face is found and matches someone with
  cosine >= FACE_MATCH, it is that person. Otherwise, the clothes (and height) of people seen
  earlier the same day decide, if the score reaches NEW_PERSON. Otherwise, it is someone new.
- Exit (candidates: only the people inside now). The best of
      score = body cosine (same day) - HEIGHT_W * |relative height difference| + FACE_W * face cosine
  wins. If nobody is inside, the person was already in the office when the recording started, and
  the exit is treated like an entry against everyone.
- Each recording starts with the office empty in the system's eyes.

Modes, for the table of what each cue is worth:
    python 04_identify.py [face+body+height | face+body | body+height | body | height | face]
With no usable cue for an exit ("face" mode with the face not seen), the person inside longest leaves.
"""
import json
import sys

import cv2
import numpy as np
import onnxruntime as ort
from insightface.app import FaceAnalysis

from config import OUT_DIR, WEIGHTS, drop_door_flips

MODE = sys.argv[1] if len(sys.argv) > 1 else "face+body+height"
FACE_MIN_SCORE = 0.60  # SCRFD detection score to accept a face
FACE_MATCH = 0.25      # ArcFace cosine to call two faces the same person
FACE_W = 0.5
HEIGHT_W = 3.5         # 3% of height difference (about 5 cm) costs 0.10 of body cosine
NEW_PERSON = {"body": 0.85, "height": -0.07}  # body modes use the body value

USE_FACE, USE_BODY, USE_HEIGHT = "face" in MODE, "body" in MODE, "height" in MODE
NEW_T = NEW_PERSON["body"] if USE_BODY else NEW_PERSON["height"]

events = json.load(open(OUT_DIR / "events.json"))
crop_dir = OUT_DIR / "crops"
cache_file = OUT_DIR / "embeddings.npz"


def embed_all():
    face_app = FaceAnalysis(name="buffalo_l", root=str(WEIGHTS / "insightface"),
                            allowed_modules=["detection", "recognition"], providers=["CPUExecutionProvider"])
    face_app.prepare(ctx_id=-1, det_size=(640, 640))
    reid = ort.InferenceSession(str(WEIGHTS / "person_vit_clip_reid.onnx"), providers=["CPUExecutionProvider"])
    mean = np.array([0.485, 0.456, 0.406], np.float32)
    std = np.array([0.229, 0.224, 0.225], np.float32)
    faces, bodies, face_scores = [], [], []
    for i, e in enumerate(events):
        imgs = [cv2.imread(str(crop_dir / n)) for n in e["crops"]]
        imgs = [im for im in imgs if im is not None and im.shape[0] > 20 and im.shape[1] > 10]
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
        face_scores.append(best_score)
        if i % 50 == 0:
            print(f"embedded {i}/{len(events)}", flush=True)
    np.savez(cache_file, faces=np.array(faces), bodies=np.array(bodies), face_scores=np.array(face_scores))


if not cache_file.exists():
    embed_all()
emb = np.load(cache_file)
faces, bodies, face_scores = emb["faces"], emb["bodies"], emb["face_scores"]
has_face = np.linalg.norm(faces, axis=1) > 0
for k, e in enumerate(events):
    e["k"] = k

# Someone standing in the doorway makes the track flip sides: drop crossings the same track undoes within 2 s
for e in events:
    e["seq_track"] = f"{e['seq']}/{e['track']}"
events, n_drop = drop_door_flips(events, "seq_track")


class Person:
    def __init__(self, pid):
        self.pid, self.faces, self.bodies, self.heights = pid, [], {}, []
        self.inside, self.since = False, None

    def add(self, e):
        k = e["k"]
        self.bodies.setdefault(e["day"], []).append(bodies[k])
        if has_face[k]:
            self.faces.append(faces[k])
        if e["height_rel"] is not None:
            self.heights.append(e["height_rel"])


people = []


def face_sim(p, k):
    if USE_FACE and p.faces and has_face[k]:
        return max(float(faces[k] @ f) for f in p.faces)
    return None


def look_sim(p, e):
    """Clothes and height against what this person looked like earlier the same day."""
    k = e["k"]
    s, used = 0.0, False
    if USE_BODY:
        if e["day"] not in p.bodies:
            return None
        s += max(float(bodies[k] @ b) for b in p.bodies[e["day"]])
        used = True
    if USE_HEIGHT and p.heights and e["height_rel"] is not None:
        if not USE_BODY and e["day"] not in p.bodies:
            return None
        s -= HEIGHT_W * abs(e["height_rel"] - float(np.median(p.heights)))
        used = True
    return s if used else None


def who_enters(cands, e):
    """Face first; then the look from earlier the same day; else nobody (a new person)."""
    k = e["k"]
    by_face = [(face_sim(p, k), p) for p in cands]
    by_face = [t for t in by_face if t[0] is not None]
    if by_face:
        s, p = max(by_face, key=lambda t: t[0])
        if s >= FACE_MATCH:
            return p, "face", s
    by_look = [(look_sim(p, e), p) for p in cands]
    by_look = [t for t in by_look if t[0] is not None]
    if by_look:
        s, p = max(by_look, key=lambda t: t[0])
        if s >= NEW_T:
            return p, "look", s
    return None, "new", None


def who_exits(cands, e):
    k = e["k"]
    scored = []
    for p in cands:
        s, used = 0.0, False
        ls = look_sim(p, e)
        if ls is not None:
            s, used = s + ls, True
        fs = face_sim(p, k)
        if fs is not None:
            s, used = s + (FACE_W if (USE_BODY or USE_HEIGHT) else 1.0) * fs, True
        scored.append((s if used else -np.inf, p))
    s, p = max(scored, key=lambda t: t[0])
    if not np.isfinite(s):  # no cue at all: the person inside longest leaves
        p = min(cands, key=lambda q: q.since)
        return p, "longest inside", None
    return p, "look" if len(cands) > 1 else "only one inside", s


seq = None
for e in events:
    if e["seq"] != seq:  # a new recording: nobody is inside yet
        seq = e["seq"]
        for p in people:
            p.inside = False
    inside = [q for q in people if q.inside]
    if e["kind"] == "enter" or not inside:
        cands = [q for q in people if not q.inside] if e["kind"] == "enter" else people
        p, how, s = who_enters(cands, e)
        if p is None:
            p = Person(len(people) + 1)
            people.append(p)
    else:
        p, how, s = who_exits(inside, e)
    p.inside = e["kind"] == "enter"
    p.since = e["time"]
    p.add(e)
    e.update(person=p.pid, how=how, score=None if s is None else round(float(s), 3),
             face_found=bool(has_face[e["k"]]), face_score=float(face_scores[e["k"]]))

json.dump(events, open(OUT_DIR / f"identified_{MODE}.json", "w"), indent=1)
ent = [e for e in events if e["kind"] == "enter"]
ext = [e for e in events if e["kind"] == "exit"]
print(f"[{MODE}] {len(events)} crossings after dropping {n_drop} doorway flips; {len(people)} people; "
      f"faces at {sum(e['face_found'] for e in ent)}/{len(ent)} entries, {sum(e['face_found'] for e in ext)}/{len(ext)} exits")
