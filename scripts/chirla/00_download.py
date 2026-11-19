"""Step 0: download CHIRLA camera 1 (10 videos, 1.36 GB, and their annotations, 21 MB) from ScienceDB.

ScienceDB needs no account but serves each connection slowly (about 100 KB/s). It accepts byte ranges,
so each file is fetched in 4 MB pieces over 32 connections: about 8 minutes in all.
The files land in data/raw/chirla/{videos,annotations}/ as {seq}__{original name}.

Usage: python 00_download.py [connections]
"""
import concurrent.futures as cf
import sys
import time
import urllib.request
from pathlib import Path

from config import RAW

CONNECTIONS = int(sys.argv[1]) if len(sys.argv) > 1 else 32
PIECE = 4 << 20
LIST = Path(__file__).with_name("camera1_files.txt")


def fetch(url, a, b, path):
    for _ in range(30):
        try:
            req = urllib.request.Request(url, headers={"Range": f"bytes={a}-{b}"})
            data = urllib.request.urlopen(req, timeout=90).read()
            if len(data) == b - a + 1:
                with open(path, "r+b") as f:
                    f.seek(a)
                    f.write(data)
                return len(data)
        except OSError:
            pass
        time.sleep(2)
    raise RuntimeError(f"could not download {path.name} at byte {a}")


jobs = []
for line in LIST.read_text().splitlines():
    if line.startswith("#") or not line.strip():
        continue
    rel, size, url = line.split()
    kind, seq, name = rel.split("/")
    path = RAW / kind / f"{seq}__{name}"
    if path.exists() and path.stat().st_size == int(size) and not path.with_suffix(".part").exists():
        continue
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.truncate(int(size))
    path.with_suffix(".part").touch()  # removed once every piece has arrived
    jobs += [(url, a, min(a + PIECE, int(size)) - 1, path) for a in range(0, int(size), PIECE)]

t0, done = time.time(), 0
with cf.ThreadPoolExecutor(CONNECTIONS) as pool:
    for i, fut in enumerate(cf.as_completed([pool.submit(fetch, *j) for j in jobs])):
        done += fut.result()
        if i % 20 == 0:
            print(f"{i}/{len(jobs)} pieces, {done / 1e6:.0f} MB, {done / 1e6 / max(time.time() - t0, 1):.1f} MB/s", flush=True)
for p in {j[3] for j in jobs}:
    p.with_suffix(".part").unlink()
print(f"done: {done / 1e9:.2f} GB in {(time.time() - t0) / 60:.0f} min")
