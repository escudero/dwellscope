"""Step 6: the video for the post, on one real recording.

The office camera with each tracked person's name, the door line, a "dentro agora" counter and, at
every crossing, what identified the person. Ends on the day-by-day log (system against ground truth).
On-screen text is in Portuguese.

Usage: python 06_video.py [seq] [mode] [speed]
"""
import json
import pickle
import subprocess
import sys
from collections import defaultdict

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from config import FPS, OUT_DIR, fmt_time, sequences, stitch

SEQ = sys.argv[1] if len(sys.argv) > 1 else "seq_025"
MODE = sys.argv[2] if len(sys.argv) > 2 else "face+body+height"
SPEED = int(sys.argv[3]) if len(sys.argv) > 3 else 3
OUT_W, OUT_H = 1080, 720
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_B = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
COLORS = [(230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200), (245, 130, 48), (145, 30, 180),
          (70, 240, 240), (240, 50, 230), (210, 245, 60), (250, 190, 212), (0, 128, 128), (170, 110, 40)]
HOW = {"face": "rosto", "look": "roupa e altura", "new": "pessoa nova", "only one inside": "único lá dentro",
       "longest inside": "há mais tempo dentro"}

s = next(s for s in sequences() if s["seq"] == SEQ)
pred = [e for e in json.load(open(OUT_DIR / f"identified_{MODE}.json")) if e["seq"] == SEQ]
res = next(r for r in json.load(open(OUT_DIR / "results.json"))["modes"] if r["mode"] == MODE)
rows = pickle.load(open(OUT_DIR / "tracks" / f"{SEQ}.pkl", "rb"))

# Events name stitched tracks (see 03_events.py): label every box with its stitched track
raw = defaultdict(list)
for f, tid, box, conf, kpts in rows:
    raw[tid].append((f, box, tid))
for obs in raw.values():
    obs.sort(key=lambda o: o[0])
by_frame = defaultdict(list)
for chain, obs in stitch(raw).items():
    for f, box, tid in obs:
        by_frame[f].append((chain, box))
track_label = defaultdict(list)
events_at = defaultdict(list)
for e in pred:
    track_label[e["track"]].append((e["frame"], e["person"]))
    events_at[e["frame"]].append(e)

font = lambda size, b=False: ImageFont.truetype(FONT_B if b else FONT, size)
F_LABEL, F_HUD, F_BIG, F_SMALL, F_ROW = font(20, True), font(24, True), font(34, True), font(18), font(19)


def label_for(tid, f):
    marks = track_label.get(tid)
    return min(marks, key=lambda m: abs(m[0] - f))[1] if marks else None


cap = cv2.VideoCapture(str(s["video"]))
n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
out_path = OUT_DIR / f"dwellscope_chirla_{SEQ}.mp4"
ff = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
                       "-s", f"{OUT_W}x{OUT_H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-crf", "21",
                       "-pix_fmt", "yuv420p", str(out_path)], stdin=subprocess.PIPE)
dy, (dx0, dx1) = s["door"]["y"], s["door"]["x"]
inside, callouts = set(), []
for f in range(n):
    ok, frame = cap.read()
    if not ok:
        break
    for e in events_at.get(f, []):
        col = COLORS[(e["person"] - 1) % len(COLORS)]
        if e["kind"] == "enter":
            inside.add(e["person"])
            txt = f"ENTROU  Funcionário {e['person']:02d}  · {HOW[e['how']]}"
        else:
            inside.discard(e["person"])
            txt = f"SAIU de costas  → Funcionário {e['person']:02d}  · {HOW[e['how']]}"
        callouts.append((f + 3 * FPS, txt, col))
    if f % SPEED:
        continue
    img = Image.fromarray(frame[:, :, ::-1])
    d = ImageDraw.Draw(img, "RGBA")
    d.line([(dx0, dy), (dx1, dy)], fill=(255, 60, 60, 220), width=4)
    for tid, box in by_frame.get(f, []):
        pid = label_for(tid, f)
        x1, y1, x2, y2 = box
        col = COLORS[(pid - 1) % len(COLORS)] if pid else (200, 200, 200)
        d.rectangle([x1, y1, x2, y2], outline=col + (255,), width=3)
        if pid:
            t = f"Func. {pid:02d}"
            tw = d.textlength(t, font=F_LABEL)
            ty = max(y1 - 26, 50)
            d.rectangle([x1, ty, x1 + tw + 10, ty + 26], fill=col + (220,))
            d.text((x1 + 5, ty + 2), t, font=F_LABEL, fill=(0, 0, 0))
    d.rectangle([0, 0, OUT_W, 46], fill=(0, 0, 0, 170))
    d.text((14, 9), f"Sala restrita · dentro agora: {len(inside)}", font=F_HUD, fill=(255, 255, 255))
    clock = (s["start"].hour * 3600 + s["start"].minute * 60 + s["start"].second + f / FPS)
    t = f"{s['start']:%d/%m/%Y} {int(clock // 3600):02d}:{int(clock % 3600 // 60):02d}:{int(clock % 60):02d} · {SPEED}x"
    d.text((OUT_W - 14 - d.textlength(t, font=F_HUD), 9), t, font=F_HUD, fill=(255, 255, 255))
    d.rectangle([0, OUT_H - 30, OUT_W, OUT_H], fill=(0, 0, 0, 150))
    d.text((14, OUT_H - 26), "porta (linha vermelha) · vídeo real: dataset CHIRLA, Univ. de Alicante, CC BY 4.0",
           font=F_SMALL, fill=(255, 255, 255, 220))
    callouts = [c for c in callouts if c[0] > f]
    for i, (_, txt, col) in enumerate(callouts[-3:]):
        y = 58 + i * 40
        d.rectangle([10, y, 22 + d.textlength(txt, font=F_LABEL), y + 32], fill=(0, 0, 0, 190))
        d.rectangle([10, y, 16, y + 32], fill=col + (255,))
        d.text((22, y + 5), txt, font=F_LABEL, fill=(255, 255, 255))
    ff.stdin.write(np.asarray(img.resize((OUT_W, OUT_H))).tobytes())

# Closing card: the day-by-day log, system against ground truth
card = Image.new("RGB", (OUT_W, OUT_H), (18, 18, 22))
d = ImageDraw.Draw(card)
d.text((40, 30), "Livro de ponto da sala: tempo dentro por dia", font=F_BIG, fill=(255, 255, 255))
d.text((40, 78), "sistema (uma câmera na porta) × gabarito (anotação do CHIRLA) · 7 dias, 10 gravações",
       font=F_SMALL, fill=(180, 180, 180))
rows_ = sorted(res["rows"], key=lambda r: (r["day"], r["person"]))
half = (len(rows_) + 1) // 2
for col, part in enumerate((rows_[:half], rows_[half:])):
    x0, y = 40 + col * 520, 116
    for dx, t in ((0, "dia"), (90, "pessoa"), (190, "gabarito"), (300, "sistema"), (410, "erro")):
        d.text((x0 + dx, y), t, font=F_SMALL, fill=(150, 150, 150))
    y += 28
    for r in part:
        err = r["est_s"] - r["true_s"]
        c = (120, 220, 120) if abs(err) <= 10 else (240, 190, 80) if abs(err) <= 30 else (240, 90, 90)
        for dx, t in ((0, f"{r['day'][8:10]}/{r['day'][5:7]}"), (90, f"{r['person']:2d}"),
                      (190, fmt_time(r["true_s"])), (300, fmt_time(r["est_s"]))):
            d.text((x0 + dx, y), t, font=F_SMALL, fill=(230, 230, 230))
        d.text((x0 + 410, y), f"{round(err):+d} s", font=F_SMALL, fill=c)
        y += 25
d.text((40, OUT_H - 70), f"{len(res['rows'])} pares pessoa-dia · erro mediano {res['dwell_median_s']:.0f} s, médio {res['dwell_mae_s']:.0f} s · "
       f"{res['dwell_within_10s']:.0%} com até 10 s de erro", font=F_SMALL, fill=(200, 200, 200))
d.text((40, OUT_H - 42), f"passagens achadas: {res['recall']:.0%} · entradas com o nome certo: "
       f"{res['entry_id_acc']:.0%} · saídas de costas com o nome certo: {res['exit_id_acc']:.0%}",
       font=F_SMALL, fill=(200, 200, 200))
arr = np.asarray(card).tobytes()
for _ in range(FPS * 7):
    ff.stdin.write(arr)
ff.stdin.close()
ff.wait()
print("wrote", out_path)
