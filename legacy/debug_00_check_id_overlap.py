#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval/debug_00_check_id_overlap.py  (STRICT matching)

- Compare protocol_id sets between:
  (A) filter ids (CSV) and/or gold_pairs (JSONL)
  (B) arts JSONL (gold_articles_sections_pmc.jsonl)

- No normalization. Exact string match only.
"""

import argparse
import csv
import json
from pathlib import Path


def read_ids_csv(p: Path) -> set:
    if not p: return set()
    ids = set()
    with p.open("r", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    if not rows: return ids
    header = [h.strip().lower() for h in rows[0]]
    if "protocol_id" in header:
        idx = header.index("protocol_id")
        data = rows[1:]
        for r in data:
            if len(r) > idx and r[idx].strip():
                ids.add(r[idx].strip())
    else:
        # 첫 열 사용, 헤더가 'id'/'protocol_id'면 스킵
        cands = [r[0].strip() for r in rows if r]
        if cands and cands[0].lower() in ("id", "protocol_id"):
            cands = cands[1:]
        ids.update([x for x in cands if x])
    return ids


def read_ids_gold_pairs(p: Path) -> set:
    if not p: return set()
    ids = set()
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            r = json.loads(line)
            pid = r.get("protocol_id", "").strip()
            if pid: ids.add(pid)
    return ids


def read_ids_arts(p: Path) -> set:
    ids = set()
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            r = json.loads(line)
            # arts 파일은 protocol_id가 반드시 있어야 함 (정확 매칭)
            pid = (r.get("protocol_id") or "").strip()
            if pid: ids.add(pid)
    return ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--filter-ids", help="data/splits/test_biop_ids_top15.csv")
    ap.add_argument("--gold-pairs", help="data/gold/gold_pairs_testset_top15.jsonl")
    ap.add_argument("--arts", required=True, help="data/gold/gold_articles_sections_pmc.jsonl")
    args = ap.parse_args()

    filt = set()
    if args.filter_ids:
        filt |= read_ids_csv(Path(args.filter_ids))
    if args.gold_pairs:
        filt |= read_ids_gold_pairs(Path(args.gold_pairs))

    arts = read_ids_arts(Path(args.arts))

    inter = filt & arts
    only_filt = filt - arts
    only_arts = arts - filt

    print(f"[STATS] filter_ids={len(filt)} arts_ids={len(arts)} overlap={len(inter)}")
    if inter:
        print("[OVERLAP] sample (up to 15):")
        for i, x in enumerate(sorted(inter)):
            if i >= 15: break
            print(" ", x)
    if only_filt:
        print("\n[MISSING in arts] present in filter/gold_pairs but NOT in arts (up to 15):")
        for i, x in enumerate(sorted(only_filt)):
            if i >= 15: break
            print(" ", x)
    if only_arts:
        print("\n[NOT requested] present in arts but NOT requested (up to 15):")
        for i, x in enumerate(sorted(only_arts)):
            if i >= 15: break
            print(" ", x)


if __name__ == "__main__":
    main()
