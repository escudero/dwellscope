"""Shared settings for the real-video phase: CHIRLA camera 1, a small office with one door.

There is no camera calibration in CHIRLA, so everything here is in image pixels (1080 x 720).
The camera sits in the corner opposite the door: entries walk toward it, exits walk away from it.
"""
import re
from collections import deque
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data/raw/chirla"
OUT_DIR = ROOT / "data/output/chirla"
WEIGHTS = ROOT / "weights"

FPS = 30
W, H = 1080, 720

# The door line, drawn by hand on the image (pixels). A foot point above the line (smaller y)
# is in the doorway or the corridor behind it; below it is the office. The camera was remounted
# between July and December 2023, so there are two lines.
DOORS = {
    "jun-jul": {"y": 235, "x": (300, 620)},  # between the door hinge (390, 265) and the right jamb (515, 150)
    "dec": {"y": 300, "x": (260, 520)},      # threshold between the jambs at x 300 and 460
}
HYSTERESIS = 30       # pixels past the line before a person changes side: e.g. inside below y 280, outside above 190
SMOOTH = 15           # median of the last 15 foot points (0.5 s): a box cut by someone in front jumps up
DOOR_DWELL_S = 2.0    # a crossing undone by the same person within 2 s was someone standing in the doorway
EDGE = 4              # boxes this close to the image border are cut off


def sequences():
    """Every camera 1 recording, oldest first: seq, video, annotation file, start time, day."""
    out = []
    for v in sorted((RAW / "videos").glob("seq_*__camera_1_*.avi")):
        seq, stamp = re.match(r"(seq_\d+)__camera_1_(.+)\.avi", v.name).groups()
        start = datetime.strptime(stamp, "%Y-%m-%d-%H:%M:%S")
        ann = RAW / "annotations" / f"{seq}__camera_1_{stamp}.json"
        door = DOORS["dec" if start.month == 12 else "jun-jul"]
        out.append({"seq": seq, "video": v, "ann": ann, "start": start, "day": start.date().isoformat(), "door": door})
    return sorted(out, key=lambda s: s["start"])


class SideTracker:
    """Which side of the door a track is on; returns 'enter' or 'exit' when it crosses."""

    def __init__(self, door):
        self.door_y, self.door_x = door["y"], door["x"]
        self.state = None  # True = inside the office
        self.recent = deque(maxlen=SMOOTH)

    def update(self, u, v):
        self.recent.append(v)
        v = sorted(self.recent)[len(self.recent) // 2]
        if not self.door_x[0] < u < self.door_x[1] and v < self.door_y:
            return None
        if self.state is None:
            # a new track decides its side only once it is clearly off the door line
            if abs(v - self.door_y) > HYSTERESIS:
                self.state = v > self.door_y
            return None
        if not self.state and v > self.door_y + HYSTERESIS:
            self.state = True
            return "enter"
        if self.state and v < self.door_y - HYSTERESIS:
            self.state = False
            return "exit"
        return None


def drop_door_flips(events, key):
    """Drop pairs of opposite crossings by the same person (or track) less than DOOR_DWELL_S apart."""
    drop, by = set(), {}
    for k, e in enumerate(events):
        by.setdefault(e[key], []).append(k)
    for ks in by.values():
        for a, b in zip(ks, ks[1:]):
            ea, eb = events[a], events[b]
            if a not in drop and ea["kind"] != eb["kind"] and eb["time"] - ea["time"] < DOOR_DWELL_S:
                drop |= {a, b}
    return [e for k, e in enumerate(events) if k not in drop], len(drop)


LINK_S = 1.5    # a track that starts this soon after another one ended...
LINK_PX = 120   # ...this close to where it ended (foot point, pixels) continues it


def stitch(tracks):
    """Join tracks the tracker broke in the doorway: {tid: [(frame, box, ...)]} -> {chain id: observations}.
    Each new track continues the closest track that ended within LINK_S and LINK_PX of its start."""
    foot = lambda o: ((o[1][0] + o[1][2]) / 2, o[1][3])
    order = sorted(tracks, key=lambda t: tracks[t][0][0])
    chain_of, ends, chains = {}, {}, {}
    for t in order:
        obs = tracks[t]
        f0, (u0, v0) = obs[0][0], foot(obs[0])
        best, best_d = None, LINK_PX
        for c, (fe, (ue, ve)) in ends.items():
            d = ((u0 - ue) ** 2 + (v0 - ve) ** 2) ** 0.5
            if 0 < f0 - fe <= LINK_S * FPS and d < best_d:
                best, best_d = c, d
        c = best if best is not None else t
        chains.setdefault(c, []).extend(obs)
        chain_of[t] = c
        ends[c] = (obs[-1][0], foot(obs[-1]))
    return chains


def fmt_time(s):
    s = int(round(s))
    return f"{s // 60}:{s % 60:02d}"
