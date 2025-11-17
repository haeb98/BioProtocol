#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

TOK_RX = re.compile(r"[A-Za-z0-9]+")
RESULTS_TITLE_RX = re.compile(r"\b(results?|findings?|outcomes?|observations?|analysis)\b", re.I)


def tset(s: str) -> set: return set(TOK_RX.findall((s or "").lower()))


def jacc(a: set, b: set) -> float:
    if not a and not b: return 1.0
    if not a or not b: return 0.0
    return len(a & b) / max(1, len(a | b))


def load_tasks(p: Path) -> dict:
    by_pid = defaultdict(list)
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except:
                continue
            pid = r.get("protocol_id")
            if not pid: continue
            by_pid[pid].append(r)
    return by_pid


def load_pairs(p: Path) -> dict:
    m = {}
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except:
                continue
            pid = r.get("protocol_id")
            if not pid: continue
            m[pid] = r
    return m


def results_tokens(pair_rec: dict) -> set:
    secs = (pair_rec.get("article") or {}).get("sections") or {}
    txts = []
    for title, txt in secs.items():
        if title and RESULTS_TITLE_RX.search(title):
            if txt and txt.strip(): txts.append(txt.strip())
    return tset("\n\n".join(txts))


def task_stats(tasks: list, methods_toks: set, results_toks: set):
    n = len(tasks)
    goal_rate = sum(1 for t in tasks if (t.get("goal") and t.get("goal").strip())) / max(1, n)
    overlaps = [];
    ralign = []
    for t in tasks:
        text = (t.get("title") or "") + " " + (t.get("description") or "")
        T = tset(text)
        overlaps.append(jacc(T, methods_toks))
        if results_toks:
            ralign.append(jacc(T, results_toks))
        else:
            ralign.append(0.0)
    return {
        "n_tasks": n,
        "goal_rate": goal_rate,
        "overlap_methods": (sum(overlaps) / len(overlaps) if overlaps else 0.0),
        "results_align": (sum(ralign) / len(ralign) if ralign else 0.0),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks-m", required=True)
    ap.add_argument("--tasks-mr", required=True)
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--out", default="reports/tasks_11_context_ablation.csv")
    args = ap.parse_args()

    tm = load_tasks(Path(args.tasks_m))
    tmr = load_tasks(Path(args.tasks_mr))
    pairs = load_pairs(Path(args.pairs))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as w:
        w.write(",".join([
            "protocol_id",
            "n_tasks_M", "goal_rate_M", "overlap_methods_M", "results_align_M",
            "n_tasks_MR", "goal_rate_MR", "overlap_methods_MR", "results_align_MR"
        ]) + "\n")

        deltas = []
        for pid, rec in pairs.items():
            methods_toks = tset(rec.get("sec_text") or "")
            r_toks = results_tokens(rec)

            sm = task_stats(tm.get(pid, []), methods_toks, r_toks)
            smr = task_stats(tmr.get(pid, []), methods_toks, r_toks)

            w.write(
                f"{pid},{sm['n_tasks']},{sm['goal_rate']:.4f},{sm['overlap_methods']:.4f},{sm['results_align']:.4f},"
                f"{smr['n_tasks']},{smr['goal_rate']:.4f},{smr['overlap_methods']:.4f},{smr['results_align']:.4f}\n"
            )
            deltas.append((
                smr['n_tasks'] - sm['n_tasks'],
                smr['goal_rate'] - sm['goal_rate'],
                smr['overlap_methods'] - sm['overlap_methods'],
                smr['results_align'] - sm['results_align']
            ))

    if deltas:
        import statistics as S
        dN = [d[0] for d in deltas];
        dG = [d[1] for d in deltas]
        dO = [d[2] for d in deltas];
        dR = [d[3] for d in deltas]
        print("[ABLATION SUMMARY: MR - M]")
        print(f"d(n_tasks): mean={S.mean(dN):.3f}, median={S.median(dN):.3f}")
        print(f"d(goal_rate): mean={S.mean(dG):.3f}, median={S.median(dG):.3f}")
        print(f"d(overlap_methods): mean={S.mean(dO):.3f}, median={S.median(dO):.3f}")
        print(f"d(results_align): mean={S.mean(dR):.3f}, median={S.median(dR):.3f}")
    print(f"[OK] ablation -> {args.out}")


if __name__ == "__main__":
    main()
