"""Step 6: the video for the post.

The door camera with each tracked person's name, the door line, a "dentro agora" counter and,
at every crossing, what identified the person (face at the entry; clothes and height at the exit).
Ends on the table of time in the room against the ground truth. On-screen text is in Portuguese.

Usage: python 06_video.py [mode] [speed]
"""
import json
import pickle
import subprocess
import sys
from collections import defaultdict

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from config import CAMERA, DOOR_Y, FPS, OUT_DIR, ROOM_X, SCENE_DIR, fmt_time, load_camera, project

MODE = sys.argv[1] if len(sys.argv) > 1 else "face+body+height"
SPEED = int(sys.argv[2]) if len(sys.argv) > 2 else 4
OUT_W, OUT_H = 1280, 720
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_B = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
COLORS = [(230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200), (245, 130, 48), (145, 30, 180),
          (70, 240, 240), (240, 50, 230), (210, 245, 60), (250, 190, 212), (0, 128, 128), (170, 110, 40)]

pred = json.load(open(OUT_DIR / f"identified_{MODE}.json"))
res = next(r for r in json.load(open(OUT_DIR / "results.json"))["modes"] if r["mode"] == MODE)
rows = pickle.load(open(OUT_DIR / "tracks.pkl", "rb"))
P = load_camera()

by_frame = defaultdict(list)
for f, tid, box, conf, kpts in rows:
    by_frame[f].append((tid, box))

# A track carries the name decided at its crossings; before its first crossing it is unnamed
track_label = defaultdict(list)  # tid -> [(frame, person)]
for e in pred:
    track_label[e["track"]].append((e["frame"], e["person"]))
events_at = defaultdict(list)
for e in pred:
    events_at[e["frame"]].append(e)

font = lambda s, b=False: ImageFont.truetype(FONT_B if b else FONT, s)
F_LABEL, F_HUD, F_BIG, F_SMALL = font(22, True), font(26, True), font(40, True), font(20)


def label_for(tid, f):
    marks = track_label.get(tid)
    if not marks:
        return None
    return min(marks, key=lambda m: abs(m[0] - f))[1]


def door_polyline():
    pts = [project(P, x, DOOR_Y) for x in np.linspace(ROOM_X[0], ROOM_X[1], 40)]
    return [(float(u), float(v)) for u, v in pts if -500 < u < 2500]


DOOR = door_polyline()
sx, sy = OUT_W / 1920, OUT_H / 1080
cap = cv2.VideoCapture(str(SCENE_DIR / "videos" / f"{CAMERA}.mp4"))
n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
out_path = OUT_DIR / f"dwellscope_{MODE.replace('+', '_')}.mp4"
ff = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
                       "-s", f"{OUT_W}x{OUT_H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-crf", "20",
                       "-pix_fmt", "yuv420p", str(out_path)], stdin=subprocess.PIPE)

inside = set()
callouts = []  # (until_frame, text, color)
for f in range(n):
    ok, frame = cap.read()
    if not ok:
        break
    for e in events_at.get(f, []):
        if e["kind"] == "enter":
            inside.add(e["person"])
            how = "rosto" if e["face_found"] else "roupa e altura (sem rosto)"
            txt = f"ENTROU  Funcionário {e['person']:02d}  · {how}"
        else:
            inside.discard(e["person"])
            h = f" · {e['height_m']:.2f} m" if e["height_m"] else ""
            txt = f"SAIU  de costas: roupa{h} → Funcionário {e['person']:02d}"
        callouts.append((f + 3 * FPS, txt, COLORS[(e["person"] - 1) % len(COLORS)]))
    if f % SPEED:
        continue
    img = Image.fromarray(cv2.resize(frame, (OUT_W, OUT_H))[:, :, ::-1])
    d = ImageDraw.Draw(img, "RGBA")
    d.line([(u * sx, v * sy) for u, v in DOOR], fill=(255, 60, 60, 220), width=4)
    for tid, box in by_frame.get(f, []):
        pid = label_for(tid, f)
        x1, y1, x2, y2 = box[0] * sx, box[1] * sy, box[2] * sx, box[3] * sy
        col = COLORS[(pid - 1) % len(COLORS)] if pid else (200, 200, 200)
        d.rectangle([x1, y1, x2, y2], outline=col + (255,), width=3)
        if pid:
            t = f"Func. {pid:02d}"
            tw = d.textlength(t, font=F_LABEL)
            d.rectangle([x1, y1 - 28, x1 + tw + 10, y1], fill=col + (220,))
            d.text((x1 + 5, y1 - 26), t, font=F_LABEL, fill=(0, 0, 0))
    d.rectangle([0, 0, OUT_W, 50], fill=(0, 0, 0, 170))
    d.text((16, 10), f"Sala restrita · dentro agora: {len(inside)}", font=F_HUD, fill=(255, 255, 255))
    t = f"{fmt_time(f / FPS)}  ·  {SPEED}x"
    d.text((OUT_W - 16 - d.textlength(t, font=F_HUD), 10), t, font=F_HUD, fill=(255, 255, 255))
    d.text((16, OUT_H - 30), "porta (linha vermelha) · vídeo sintético NVIDIA PhysicalAI-SmartSpaces",
           font=F_SMALL, fill=(255, 255, 255, 200))
    callouts = [c for c in callouts if c[0] > f]
    for i, (_, txt, col) in enumerate(callouts[-3:]):
        y = 64 + i * 42
        d.rectangle([12, y, 24 + d.textlength(txt, font=F_LABEL), y + 34], fill=(0, 0, 0, 190))
        d.rectangle([12, y, 18, y + 34], fill=col + (255,))
        d.text((24, y + 5), txt, font=F_LABEL, fill=(255, 255, 255))
    ff.stdin.write(np.asarray(img).tobytes())

# Closing card: time in the room, system against ground truth
card = Image.new("RGB", (OUT_W, OUT_H), (18, 18, 22))
d = ImageDraw.Draw(card)
d.text((60, 40), "Tempo na sala restrita em 5 minutos", font=F_BIG, fill=(255, 255, 255))
d.text((60, 100), "sistema (câmera na porta) × gabarito (caixas 3D do simulador)", font=F_SMALL, fill=(180, 180, 180))
y = 150
for r in sorted(res["rows"], key=lambda r: -r["true_s"]):
    err = r["est_s"] - r["true_s"]
    col = (120, 220, 120) if abs(err) <= 10 else (240, 190, 80) if abs(err) <= 30 else (240, 90, 90)
    d.text((60, y), f"Pessoa {r['gt_person']:2d}", font=F_HUD, fill=(230, 230, 230))
    d.text((300, y), f"gabarito {fmt_time(r['true_s'])}", font=F_HUD, fill=(230, 230, 230))
    d.text((620, y), f"sistema {fmt_time(r['est_s'])}", font=F_HUD, fill=(230, 230, 230))
    d.text((940, y), f"{err:+.0f} s", font=F_HUD, fill=col)
    y += 46
d.text((60, OUT_H - 50), f"erro médio: {res['dwell_mae_s']:.0f} s por pessoa · passagens achadas: "
       f"{res['recall']:.0%} · saídas de costas com o nome certo: {res['exit_id_acc']:.0%}",
       font=F_SMALL, fill=(200, 200, 200))
arr = np.asarray(card).tobytes()
for _ in range(FPS * 6):
    ff.stdin.write(arr)
ff.stdin.close()
ff.wait()
print("wrote", out_path)
