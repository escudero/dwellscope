"""Step 5: compare with the ground truth, per recording and per day.

1. Door crossings: each detected crossing is matched to a true one of the same kind, in the same
   recording, within 3 s.
2. Identity: each system person is mapped one-to-one to a true person (Hungarian on shared crossings,
   over all days at once: a person split across days counts as an error).
3. Time in the office: entries are paired with the next exit of the same person in the same recording,
   then summed per person and day.

Usage: python 05_evaluate.py [mode ...]   (default: every identified_*.json)
"""
import json
import sys
from collections import Counter, defaultdict

import numpy as np
from scipy.optimize import linear_sum_assignment

from config import OUT_DIR, fmt_time

TOL_S = 3.0
gt = json.load(open(OUT_DIR / "gt.json"))
FPS = gt["fps"]
SEQS = {s["seq"]: s for s in gt["sequences"]}
gt_events = [dict(e, seq=s["seq"], day=s["day"]) for s in gt["sequences"] for e in s["events"]]


def match_events(pred):
    pairs, used = {}, set()
    cands = sorted((abs(p["time"] - g["time"]), i, j) for i, p in enumerate(pred) for j, g in enumerate(gt_events)
                   if p["seq"] == g["seq"] and p["kind"] == g["kind"] and abs(p["time"] - g["time"]) <= TOL_S)
    for _, i, j in cands:
        if i not in pairs and j not in used:
            pairs[i] = j
            used.add(j)
    return pairs


def dwell(evts, key):
    """Seconds inside per (person, day): enter -> next exit in the same recording. An exit with no entry
    counts from the start of the recording; an entry with no exit, until its end; two entries in a row
    close the first stay when its track was last seen."""
    by = defaultdict(list)
    for e in sorted(evts, key=lambda e: (e["seq"], e["time"])):
        by[(e[key], e["seq"])].append(e)
    total = defaultdict(float)
    for (pid, seq), es in by.items():
        t_end = SEQS[seq]["n_frames"] / FPS
        s, t_in = 0.0, None
        for i, e in enumerate(es):
            if e["kind"] == "enter":
                if t_in is not None:
                    s += max(es[i - 1].get("track_end", t_in) - t_in, 0.0)
                t_in = e["time"]
            else:
                s += e["time"] - (t_in if t_in is not None else (0.0 if i == 0 else e["time"]))
                t_in = None
        if t_in is not None:
            s += t_end - t_in
        total[(pid, SEQS[seq]["day"])] += s
    return total


def evaluate(mode, verbose=True):
    pred = json.load(open(OUT_DIR / f"identified_{mode}.json"))
    pairs = match_events(pred)
    tp = len(pairs)
    prec, rec = tp / max(len(pred), 1), tp / len(gt_events)

    sys_ids = sorted({p["person"] for p in pred})
    gt_ids = sorted({g["person"] for g in gt_events})
    C = np.zeros((len(sys_ids), len(gt_ids)))
    for i, j in pairs.items():
        C[sys_ids.index(pred[i]["person"]), gt_ids.index(gt_events[j]["person"])] += 1
    r, c = linear_sum_assignment(-C)
    mapping = {sys_ids[a]: gt_ids[b] for a, b in zip(r, c) if C[a, b] > 0}
    correct = {k: sum(mapping.get(pred[i]["person"]) == gt_events[j]["person"]
                      for i, j in pairs.items() if pred[i]["kind"] == k) for k in ("enter", "exit")}
    n_kind = Counter(pred[i]["kind"] for i in pairs)
    face_entries = [i for i in pairs if pred[i]["kind"] == "enter" and pred[i]["face_found"]]

    true_dwell = dwell(gt_events, "person")
    sys_dwell = dwell([dict(p, mapped=mapping.get(p["person"], -p["person"])) for p in pred], "mapped")
    rows = [{"person": g, "day": d, "true_s": t, "est_s": sys_dwell.get((g, d), 0.0)}
            for (g, d), t in sorted(true_dwell.items(), key=lambda kv: (kv[0][1], kv[0][0]))]
    errs = np.array([r["est_s"] - r["true_s"] for r in rows])
    res = {"mode": mode, "detected": len(pred), "true": len(gt_events), "precision": prec, "recall": rec,
           "entry_id_acc": correct["enter"] / max(n_kind["enter"], 1),
           "exit_id_acc": correct["exit"] / max(n_kind["exit"], 1),
           "entries": n_kind["enter"], "exits": n_kind["exit"],
           "face_at_entries": len(face_entries),
           "people_found": len(sys_ids), "people_true": len(gt_ids),
           "dwell_mae_s": float(np.abs(errs).mean()), "dwell_median_s": float(np.median(np.abs(errs))),
           "dwell_within_10s": float((np.abs(errs) <= 10).mean()),
           "rows": rows, "mapping": {str(k): v for k, v in mapping.items()}}
    if verbose:
        print(f"\n== {mode}")
        print(f"crossings: {tp}/{len(gt_events)} found, {len(pred) - tp} false  (precision {prec:.0%}, recall {rec:.0%})")
        print(f"identity: entries {correct['enter']}/{n_kind['enter']}, exits {correct['exit']}/{n_kind['exit']}; "
              f"{len(sys_ids)} system people for {len(gt_ids)} true; face found at {len(face_entries)} entries")
        print(f"time in the office per person per day ({len(rows)} rows): mean error {res['dwell_mae_s']:.1f} s, "
              f"median {res['dwell_median_s']:.1f} s, "
              f"{res['dwell_within_10s']:.0%} within 10 s")
        if mode == "face+body+height":
            for r in rows:
                print(f"  {r['day']} person {r['person']:2d}: true {fmt_time(r['true_s'])}  "
                      f"system {fmt_time(r['est_s'])}  error {r['est_s'] - r['true_s']:+.0f} s")
    return res


def crossings_only():
    """The same time-in-office table with the detected crossings but the true names: the error that
    comes from missed and false crossings alone, before identity."""
    pred = json.load(open(OUT_DIR / "identified_face+body+height.json"))
    pairs = match_events(pred)
    named = [dict(pred[i], true_person=gt_events[j]["person"]) for i, j in pairs.items()]
    true_dwell, est = dwell(gt_events, "person"), dwell(named, "true_person")
    errs = np.array([est.get(k, 0.0) - t for k, t in true_dwell.items()])
    out = {"dwell_mae_s": float(np.abs(errs).mean()), "dwell_median_s": float(np.median(np.abs(errs))),
           "dwell_within_10s": float((np.abs(errs) <= 10).mean())}
    print(f"\n== detected crossings with the true names: mean error {out['dwell_mae_s']:.1f} s, "
          f"median {out['dwell_median_s']:.1f} s, "
          f"{out['dwell_within_10s']:.0%} within 10 s")
    return out


def exit_test():
    """The idea in isolation: with the true list of who is inside, can someone leaving with their back
    to the camera be named? Each true exit is compared with the entry of everyone inside at that moment
    (entries from the same recording)."""
    pred = json.load(open(OUT_DIR / "identified_face+body+height.json"))
    emb = np.load(OUT_DIR / "embeddings.npz")
    bodies, faces = emb["bodies"], emb["faces"]
    has_face = np.linalg.norm(faces, axis=1) > 0
    pairs = match_events(pred)
    truth = {j: i for i, j in pairs.items()}
    trials = []
    for seq in SEQS:
        last_entry, inside = {}, set()
        for j, g in sorted(((j, g) for j, g in enumerate(gt_events) if g["seq"] == seq), key=lambda t: t[1]["time"]):
            if g["kind"] == "enter":
                inside.add(g["person"])
                if j in truth:
                    last_entry[g["person"]] = pred[truth[j]]
            else:
                cands = [p for p in inside if p in last_entry]
                if j in truth and g["person"] in cands and len(cands) > 1:
                    trials.append((pred[truth[j]], g["person"], {p: last_entry[p] for p in cands}))
                inside.discard(g["person"])

    def hdiff(x, c):
        return abs(x["height_rel"] - c["height_rel"]) if x["height_rel"] and c["height_rel"] else 0.0

    def face(x, c):
        return float(faces[x["k"]] @ faces[c["k"]]) if has_face[x["k"]] and has_face[c["k"]] else 0.0

    cues = {
        "chance": None,
        "face": face,
        "height": lambda x, c: -hdiff(x, c),
        "clothes and hair": lambda x, c: float(bodies[x["k"]] @ bodies[c["k"]]),
        "clothes, hair and height": lambda x, c: float(bodies[x["k"]] @ bodies[c["k"]]) - 3.5 * hdiff(x, c),
    }
    out = {"trials": len(trials), "mean_inside": float(np.mean([len(t[2]) for t in trials])) if trials else 0,
           "face_at_exit": int(sum(has_face[t[0]["k"]] for t in trials))}
    print(f"\n== exit test: {len(trials)} exits with 2+ people inside (mean {out['mean_inside']:.1f} inside), "
          f"face found at {out['face_at_exit']}")
    for name, f in cues.items():
        if not trials:
            break
        if f is None:
            acc = float(np.mean([1 / len(t[2]) for t in trials]))
        else:
            acc = float(np.mean([max(t[2], key=lambda p: f(t[0], t[2][p])) == t[1] for t in trials]))
        out[name] = acc
        print(f"  {name:26s} {acc:.0%}")
    return out


if __name__ == "__main__":
    modes = sys.argv[1:] or sorted(p.stem.removeprefix("identified_") for p in OUT_DIR.glob("identified_*.json"))
    results = [evaluate(m) for m in modes]
    json.dump({"modes": results, "crossings_only": crossings_only(), "exit_test": exit_test()}, open(OUT_DIR / "results.json", "w"), indent=1)
