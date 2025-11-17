#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

TOK_RX = re.compile(r"[A-Za-z0-9]+")


def tset(s: str) -> set: return set(TOK_RX.findall((s or "").lower()))


def jacc(a: set, b: set) -> float:
    if not a and not b: return 1.0
    if not a or not b: return 0.0
    return len(a & b) / max(1, len(a | b))


def load_pairs(p: Path) -> dict:
    m = {}
    if not p or not p.exists(): return m
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--pairs", default="")
    ap.add_argument("--out", default="reports/tasks_10_quality.csv")
    args = ap.parse_args()

    pairs_map = load_pairs(Path(args.pairs)) if args.pairs else {}

    by_pid = defaultdict(list)
    with open(args.tasks, "r", encoding="utf-8") as f:
        for line in f:
            try:
                t = json.loads(line)
            except:
                continue
            pid = t.get("protocol_id");
            if not pid: continue
            by_pid[pid].append(t)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as w:
        w.write(",".join([
            "protocol_id", "n_tasks", "title_dup_rate", "materials_rate",
            "goal_rate", "mean_task_overlap"
        ]) + "\n")

        n_list = [];
        dup_list = [];
        mat_rate_list = [];
        goal_rate_list = [];
        ov_list = []
        for pid, tasks in by_pid.items():
            n = len(tasks);
            n_list.append(n)
            titles = [(t.get("title") or "").strip().lower() for t in tasks if t.get("title")]
            dup_rate = 0.0
            if titles:
                dup_rate = 1.0 - (len(set(titles)) / len(titles))
            dup_list.append(dup_rate)

            mat_rate = sum(1 for t in tasks if t.get("key_materials")) / max(1, n)
            goal_rate = sum(1 for t in tasks if (t.get("goal") and t.get("goal").strip())) / max(1, n)
            mat_rate_list.append(mat_rate);
            goal_rate_list.append(goal_rate)

            overlap = 0.0
            if pid in pairs_map:
                sec_text = pairs_map[pid].get("sec_text") or ""
                S = tset(sec_text)
                scores = []
                for t in tasks:
                    text = (t.get("title") or "") + " " + (t.get("description") or "")
                    scores.append(jacc(tset(text), S))
                if scores: overlap = sum(scores) / len(scores)
            ov_list.append(overlap)

            w.write(f"{pid},{n},{dup_rate:.4f},{mat_rate:.4f},{goal_rate:.4f},{overlap:.4f}\n")

    def s(x):
        return f"mean={statistics.mean(x):.3f}, median={statistics.median(x):.3f}"

    print("[SUMMARY]")
    print(f"n_tasks: {s(n_list)}")
    print(f"title_dup_rate: {s(dup_list)}")
    print(f"materials_rate: {s(mat_rate_list)}")
    print(f"goal_rate: {s(goal_rate_list)}")
    print(f"task_overlap (to Methods): {s(ov_list)}")
    print(f"[OK] report -> {args.out}")


if __name__ == "__main__":
    main()
