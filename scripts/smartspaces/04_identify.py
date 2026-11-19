"""Step 4: who is each crossing?

Entry (person facing the camera): SCRFD finds the face and ArcFace (InsightFace buffalo_l) embeds it.
Every crossing also gets a CLIP-ReID body vector (clothes and hair) and the height from step 3.

Each person the system knows keeps a bank of body vectors (front views from entries, back views from
exits), the face vectors seen so far and a height. A crossing is scored against a person as

    score = best body cosine  -  HEIGHT_W * |height difference in m|  +  FACE_W * face cosine (if both have a face)

- Entry: the candidates are the people outside the room. Below NEW_PERSON, it is someone new. Someone the
  system thinks is still inside can also win, by MISSED_EXIT_MARGIN: their exit was missed (occlusion).
- Exit: the candidates are only the people inside the room now. If nobody is inside, the person was
  already in the room when the recording started.

Modes, for the table of what each cue is worth: python 04_identify.py [face+body+height | body+height | body | height | face]
"""
import json
import sys

import cv2
import numpy as np
import onnxruntime as ort
from insightface.app import FaceAnalysis

from config import OUT_DIR, WEIGHTS

MODE = sys.argv[1] if len(sys.argv) > 1 else "face+body+height"
FACE_MIN_SCORE = 0.60  # SCRFD detection score to accept a face
HEIGHT_W = 2.0         # 5 cm of height difference costs 0.10 of body cosine
FACE_W = 0.3
NEW_PERSON = {"face+body+height": 0.85, "body+height": 0.85, "body": 0.90, "height": -0.06, "face": 0.40}[MODE]
MISSED_EXIT_MARGIN = float("inf")  # off: with 0.05 to 0.2 it merged the four ~1.9 m people (see README)
DOOR_DWELL_S = 2.0     # the same track crossing back within this time was standing on the door line

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
    for e in events:
        imgs = [cv2.imread(str(crop_dir / n)) for n in e["crops"]]
        imgs = [im for im in imgs if im is not None and im.shape[0] > 20 and im.shape[1] > 10]
        # Body: CLIP-ReID on every crop, averaged
        if imgs:
            batch = np.stack([((cv2.resize(im, (128, 256))[:, :, ::-1] / 255.0 - mean) / std).transpose(2, 0, 1)
                              for im in imgs]).astype(np.float32)
            v = reid.run(None, {"input": batch})[0].mean(0)
            bodies.append(v / np.linalg.norm(v))
        else:
            bodies.append(np.zeros(512, np.float32))
        # Face: best SCRFD detection in the upper half of each crop, upscaled
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
    np.savez(cache_file, faces=np.array(faces), bodies=np.array(bodies), face_scores=np.array(face_scores))


if not cache_file.exists():
    embed_all()
emb = np.load(cache_file)
faces, bodies, face_scores = emb["faces"], emb["bodies"], emb["face_scores"]
has_face = np.linalg.norm(faces, axis=1) > 0
for k, e in enumerate(events):
    e["k"] = k

# Someone standing on the door line makes the track flip sides: drop crossings undone by the same track
drop = set()
by_track = {}
for e in events:
    by_track.setdefault(e["track"], []).append(e)
for es in by_track.values():
    for a, b in zip(es, es[1:]):
        if a["k"] not in drop and a["kind"] != b["kind"] and b["time"] - a["time"] < DOOR_DWELL_S:
            drop |= {a["k"], b["k"]}
events = [e for e in events if e["k"] not in drop]

USE_FACE, USE_BODY, USE_HEIGHT = "face" in MODE, "body" in MODE, "height" in MODE


class Person:
    def __init__(self, pid):
        self.pid, self.faces, self.bodies, self.heights, self.inside = pid, [], [], [], False

    def add(self, k):
        self.bodies.append(bodies[k])
        if has_face[k] and events_by_k[k]["kind"] == "enter":
            self.faces.append(faces[k])
        if events_by_k[k]["height_m"] is not None:
            self.heights.append(events_by_k[k]["height_m"])


events_by_k = {e["k"]: e for e in events}
people = []


def score(p, k):
    s = 0.0
    if USE_BODY:
        s += max(float(bodies[k] @ b) for b in p.bodies)
    if USE_HEIGHT and p.heights and events_by_k[k]["height_m"] is not None:
        s -= HEIGHT_W * abs(events_by_k[k]["height_m"] - float(np.median(p.heights)))
    if USE_FACE and p.faces and has_face[k] and events_by_k[k]["kind"] == "enter":
        s += (FACE_W if USE_BODY else 1.0) * max(float(faces[k] @ f) for f in p.faces)
    return s


def best_of(cands, k):
    scored = [(score(p, k), p) for p in cands]
    return max(scored, key=lambda s: s[0]) if scored else (-np.inf, None)


for e in events:
    k = e["k"]
    if e["kind"] == "enter":
        s, p = best_of([q for q in people if not q.inside], k)
        # Someone still "inside" entering again means their exit was missed. Only accept that
        # when they beat everyone outside by a clear margin.
        s_in, p_in = best_of([q for q in people if q.inside], k)
        if p_in is not None and s_in > max(s, NEW_PERSON) + MISSED_EXIT_MARGIN:
            s, p = s_in, p_in
        if p is None or s < NEW_PERSON:
            p = Person(len(people) + 1)
            people.append(p)
        e["missed_exit_before"] = p.inside
        p.inside = True
    else:
        s, p = best_of([q for q in people if q.inside], k)
        if p is None:  # nobody inside: this person was in the room when the recording started
            s, p = best_of(people, k)
            if p is None or s < NEW_PERSON:
                p = Person(len(people) + 1)
                people.append(p)
        p.inside = False
    p.add(k)
    e["person"] = p.pid
    e["score"] = None if not np.isfinite(s) else round(float(s), 3)
    e["face_found"] = bool(has_face[k])
    e["face_score"] = float(face_scores[k])

json.dump(events, open(OUT_DIR / f"identified_{MODE}.json", "w"), indent=1)
ent = [e for e in events if e["kind"] == "enter"]
ext = [e for e in events if e["kind"] == "exit"]
print(f"[{MODE}] {len(events)} crossings after dropping {len(drop)} door-line flips; {len(people)} people; "
      f"faces at {sum(e['face_found'] for e in ent)}/{len(ent)} entries, {sum(e['face_found'] for e in ext)}/{len(ext)} exits")
