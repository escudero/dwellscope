"""Shared settings for every step of the synthetic (SmartSpaces) pipeline."""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SCENE_DIR = ROOT / "data/raw/smartspaces/Hospital_000"
OUT_DIR = ROOT / "data/output/smartspaces"
WEIGHTS = ROOT / "weights"

CAMERA = "Camera_21"  # inside the room, looking at the doorway: entries face it, exits show their back
FPS = 30

# The "restricted room": the hall north of the corridor in the synthetic hospital (world metres).
# Walls on x = -32.4 and x = -23.5; the open south side at y = 16 acts as the door.
ROOM_X = (-32.4, -23.5)
DOOR_Y = 16.0
HYSTERESIS = 0.4  # a person changes side only after going this far past the door line (metres)


def in_room_x(x):
    return ROOM_X[0] < x < ROOM_X[1]


def load_camera(cam=CAMERA):
    sensors = json.load(open(SCENE_DIR / "calibration.json"))["sensors"]
    s = next(s for s in sensors if s["id"] == cam)
    return np.array(s["cameraMatrix"], dtype=float)  # 3x4, world (x, y, z) -> image


def project(P, x, y, z=0.0):
    p = P @ np.array([x, y, z, 1.0])
    return p[:2] / p[2]


def image_to_ground(P, u, v):
    """Back-project an image point onto the floor (z = 0)."""
    H = P[:, [0, 1, 3]]
    w = np.linalg.solve(H, np.array([u, v, 1.0]))
    return w[:2] / w[2]


def height_from_head(P, x, y, v_head):
    """Height of a point above floor point (x, y) that projects to image row v_head."""
    # v = (P1 . X) / (P2 . X) with X = (x, y, z, 1): linear in z
    a1 = P[1, 0] * x + P[1, 1] * y + P[1, 3]
    a2 = P[2, 0] * x + P[2, 1] * y + P[2, 3]
    return (v_head * a2 - a1) / (P[1, 2] - v_head * P[2, 2])


class SideTracker:
    """Inside/outside state with hysteresis, so jitter on the door line is not a crossing."""

    def __init__(self):
        self.state = None

    def update(self, x, y):
        if not in_room_x(x) and y > DOOR_Y:
            return None
        if self.state is None:
            # a new track decides its side only once it is clearly off the door line
            if abs(y - DOOR_Y) > HYSTERESIS:
                self.state = y > DOOR_Y
            return None
        if not self.state and y > DOOR_Y + HYSTERESIS:
            self.state = True
            return "enter"
        if self.state and y < DOOR_Y - HYSTERESIS:
            self.state = False
            return "exit"
        return None


def fmt_time(seconds):
    m, s = divmod(int(round(seconds)), 60)
    return f"{m}:{s:02d}"
