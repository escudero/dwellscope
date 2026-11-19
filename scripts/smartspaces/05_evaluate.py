"""Step 5: compare with the ground truth.

1. Door crossings: each detected crossing is matched to a true one of the same kind within 3 s.
2. Identity: each system person is mapped one-to-one to a true person (Hungarian on shared crossings).
3. Time in the room: entries are paired with the next exit of the same person and summed per person.

Usage: python 05_evaluate.py [mode ...]   (default: every identified_*.json in data/output/smartspaces)
"""
import json
import sys
from collections import Counter, defaultdict

import numpy as np
from scipy.optimize import linear_sum_assignment

from config import OUT_DIR, fmt_time

TOL_S = 3.0  # an occluded foot can delay the detected crossing by a few seconds
gt = json.load(open(OUT_DIR / "gt.json"))
T_END = gt["n_frames"] / gt["fps"]
gt_events = gt["events"]


def match_events(pred):
    pairs, used = {}, set()
    cands = sorted(((abs(p["time"] - g["time"]), i, j) for i, p in enumerate(pred) for j, g in enumerate(gt_events)
                    if p["kind"] == g["kind"] and abs(p["time"] - g["time"]) <= TOL_S))
    for _, i, j in cands:
        if i not in pairs and j not in used:
            pairs[i] = j
            used.add(j)
    return pairs


def dwell(evts, key):
    """Seconds inside per person: enter -> next exit. Exit with no entry counts from the start of the video;
    two entries in a row (missed exit) close the first stay when its track was last seen."""
    by = defaultdict(list)
    for e in sorted(evts, key=lambda e: e["time"]):
        by[e[key]].append(e)
    total = {}
    for pid, es in by.items():
        s, t_in = 0.0, None
        for i, e in enumerate(es):
            if e["kind"] == "enter":
                if t_in is not None:
                    # missed exit: close the stay when the entering track was last seen
                    s += max(es[i - 1].get("track_end", t_in) - t_in, 0.0)
                t_in = e["time"]
            else:
                s += e["time"] - (t_in if t_in is not None else (0.0 if i == 0 else e["time"]))
                t_in = None
        if t_in is not None:
            s += T_END - t_in
        total[pid] = s
    return total


def evaluate(mode, verbose=True):
    pred = json.load(open(OUT_DIR / f"identified_{mode}.json"))
    pairs = match_events(pred)
    tp = len(pairs)
    prec, rec = tp / max(len(pred), 1), tp / len(gt_events)

    # one-to-one map system person -> true person
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

    true_dwell = dwell(gt_events, "person")
    sys_dwell = dwell(pred, "person")
    rows = []
    for g in gt_ids:
        mine = [s for s, t in mapping.items() if t == g]
        est = sum(sys_dwell[s] for s in mine) if mine else 0.0
        rows.append((g, true_dwell[g], est))
    mae = np.mean([abs(a - b) for _, a, b in rows])
    res = {"mode": mode, "detected": len(pred), "true": len(gt_events), "precision": prec, "recall": rec,
           "entry_id_acc": correct["enter"] / max(n_kind["enter"], 1),
           "exit_id_acc": correct["exit"] / max(n_kind["exit"], 1),
           "people_found": len(sys_ids), "people_true": len(gt_ids), "dwell_mae_s": mae,
           "rows": [{"gt_person": g, "true_s": a, "est_s": b} for g, a, b in rows],
           "mapping": {str(k): v for k, v in mapping.items()}}
    if verbose:
        print(f"\n== {mode}")
        print(f"crossings: {tp}/{len(gt_events)} found, {len(pred) - tp} false  "
              f"(precision {prec:.0%}, recall {rec:.0%})")
        print(f"identity: entries {correct['enter']}/{n_kind['enter']}, exits {correct['exit']}/{n_kind['exit']}; "
              f"{len(sys_ids)} system people for {len(gt_ids)} true")
        for g, a, b in rows:
            print(f"  person {g:2d}: true {fmt_time(a)}  system {fmt_time(b)}  error {b - a:+.0f} s")
        print(f"mean absolute error: {mae:.1f} s per person")
    return res


modes = sys.argv[1:] or sorted(p.stem.removeprefix("identified_") for p in OUT_DIR.glob("identified_*.json"))
results = [evaluate(m) for m in modes]


def exit_test():
    """The idea in isolation: with the true list of who is inside, can clothes and height name a person
    leaving with their back to the camera? Each true exit is compared with the entry signatures of
    everyone inside at that moment."""
    pred = json.load(open(OUT_DIR / "identified_face+body+height.json"))
    emb = np.load(OUT_DIR / "embeddings.npz")
    bodies = emb["bodies"]
    pairs = match_events(pred)
    truth = {j: i for i, j in pairs.items()}  # gt event -> detected event
    last_entry, inside = {}, set()
    trials = []
    for j, g in sorted(enumerate(gt_events), key=lambda t: t[1]["time"]):
        if g["kind"] == "enter":
            inside.add(g["person"])
            if j in truth:
                last_entry[g["person"]] = pred[truth[j]]
        else:
            cands = [p for p in inside if p in last_entry]
            if j in truth and g["person"] in cands and len(cands) > 1:
                trials.append((pred[truth[j]], g["person"], {p: last_entry[p] for p in cands}))
            inside.discard(g["person"])
    cues = {
        "acaso": None,
        "altura": lambda x, c: -abs(x["height_m"] - c["height_m"]),
        "roupa e cabelo (CLIP-ReID)": lambda x, c: float(bodies[x["k"]] @ bodies[c["k"]]),
        "roupa, cabelo e altura": lambda x, c: float(bodies[x["k"]] @ bodies[c["k"]]) - 2.0 * abs(x["height_m"] - c["height_m"]),
    }
    out = {"trials": len(trials), "mean_inside": float(np.mean([len(t[2]) for t in trials])) if trials else 0}
    print(f"\n== exit test: {len(trials)} exits with 2+ people inside (mean {out['mean_inside']:.1f} inside)")
    for name, f in cues.items():
        if f is None:
            acc = float(np.mean([1 / len(t[2]) for t in trials]))
        else:
            acc = float(np.mean([max(t[2], key=lambda p: f(t[0], t[2][p])) == t[1] for t in trials]))
        out[name] = acc
        print(f"  {name:28s} {acc:.0%}")
    return out


json.dump({"modes": results, "exit_test": exit_test()}, open(OUT_DIR / "results.json", "w"), indent=1)
