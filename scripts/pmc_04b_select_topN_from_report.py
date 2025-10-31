#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pmc_04b_select_topN_from_report.py
- 전수 리포트(CSV) 기반으로 스코어링하여 정확히 N개를 선별
- 기본 스코어: score = 0.3*action + 0.2*material + 0.5*param
- suspect_mismatch=1 기본 제외(옵션으로 해제 가능)
- 선택된 N개에 대해 최종 페어(JSONL)와 protocol_id CSV 출력
- 섹션 whitelist/blacklist + prefer_regular + min_section 적용
"""

import argparse
import csv
import json
import pathlib
import re
from collections import defaultdict

try:
    import orjson as _orjson


    def jloads(b):
        return _orjson.loads(b)
except Exception:
    import json as _orjson


    def jloads(b):
        return _orjson.loads(b)

TOKSPLIT = re.compile(r"\W+")


def tokset(s): return set(t for t in TOKSPLIT.split((s or "").lower()) if t)


def load_bio(path):
    data = jloads(pathlib.Path(path).read_bytes())
    return {str(r.get("id") or "").strip(): r for r in data if r.get("id")}


def load_arts(path):
    arr = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                arr.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    best = {}
    grp = defaultdict(list)
    for r in arr: grp[r.get("protocol_id", "")].append(r)

    def totlen(x):
        return sum(len(v) for v in (x.get("sections") or {}).values())

    for pid, recs in grp.items():
        if pid: best[pid] = sorted(recs, key=totlen, reverse=True)[0]
    return best


def load_report(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bio", required=True)
    ap.add_argument("--arts", required=True)
    ap.add_argument("--report", required=True)
    ap.add_argument("--target", type=int, default=15)
    ap.add_argument("--weights", default="0.3,0.2,0.5", help="action,material,param weights")
    ap.add_argument("--exclude-suspect", action="store_true", default=True)
    ap.add_argument("--allow-suspect", action="store_true", help="override to include suspect if needed")
    ap.add_argument("--min-section", type=int, default=150)
    ap.add_argument("--whitelist",
                    default="methods|materials|experimental|star methods|experimental model and subject details")
    ap.add_argument("--blacklist", default="results|discussion|abstract")
    ap.add_argument("--prefer-regular", action="store_true", default=True)
    ap.add_argument("--out", default="data/gold/gold_pairs_testset_topN.jsonl")
    ap.add_argument("--emit-ids", default="data/splits/test_biop_ids_topN.csv")
    args = ap.parse_args()

    # suspect 처리 규칙
    exclude_suspect = args.exclude_suspect and not args.allow_suspect

    w_action, w_mat, w_param = [float(x.strip()) for x in args.weights.split(",")]
    wl = re.compile(args.whitelist, re.I) if args.whitelist.strip() else None
    bl = re.compile(args.blacklist, re.I) if args.blacklist.strip() else None

    bio = load_bio(args.bio)
    arts = load_arts(args.arts)
    rep = load_report(args.report)

    # 전수 리포트에서 점수 계산
    scored = []
    for row in rep:
        pid = row.get("protocol_id", "").strip()
        if not pid: continue
        # 수치 캐스팅
        try:
            aov = float(row.get("action_overlap", 0) or 0)
            mov = float(row.get("material_overlap", 0) or 0)
            pov = float(row.get("param_coverage", 0) or 0)
            sus = int(row.get("suspect_mismatch", 0) or 0)
        except Exception:
            aov = mov = pov = 0.0;
            sus = 0
        if exclude_suspect and sus == 1:
            continue
        score = w_action * aov + w_mat * mov + w_param * pov
        scored.append((score, pid))

    scored.sort(reverse=True)  # 점수 높은 순
    picked = [pid for _, pid in scored[:args.target]]

    outp = pathlib.Path(args.out);
    outp.parent.mkdir(parents=True, exist_ok=True)
    idcsv = pathlib.Path(args.emit_ids);
    idcsv.parent.mkdir(parents=True, exist_ok=True)

    # 선택된 ID에 대해 최종 페어(JSONL) 생성
    with open(args.out, "w", encoding="utf-8") as jf, open(args.emit_ids, "w", encoding="utf-8", newline="") as idf:
        iw = csv.writer(idf);
        iw.writerow(["protocol_id"])
        for pid in picked:
            b = bio.get(pid);
            a = arts.get(pid)
            if not b or not a:
                continue
            raw_secs = a.get("sections") or {}
            # 섹션 필터
            if wl or bl:
                filtered = {}
                for title, txt in raw_secs.items():
                    t = title or ""
                    if wl and not wl.search(t):  # whitelist 미포함
                        continue
                    if bl and bl.search(t):  # blacklist 포함
                        continue
                    filtered[t] = txt
            else:
                filtered = raw_secs

            stats = a.get("stats") or {}
            if args.prefer_regular:
                reg = {k: v for k, v in filtered.items() if (stats.get(k, {}).get("source") == "regular")}
                chosen = reg if reg else filtered
            else:
                chosen = filtered

            secs = {k: v for k, v in chosen.items() if len(v) >= args.min_section}

            pair = {
                "protocol_id": pid,
                "domain": (b.get("classification") or {}).get("primary_domain", "Unknown"),
                "article": {
                    "pmcid": a.get("pmcid", ""),
                    "xml_path": a.get("xml_path", ""),
                    "title": a.get("title", ""),
                    "meta": a.get("meta", {}),
                    "sections": secs
                },
                "protocol": {
                    "title": b.get("title") or "",
                    "keywords": b.get("keywords") or "",
                    "url": b.get("url") or "",
                    "hierarchical_protocol": b.get("hierarchical_protocol") or {},
                    "text_flat": "\n".join([
                        (b.get("title") or ""),
                        (b.get("input") or ""),
                        (b.get("protocol") or "")
                    ])
                }
            }
            json.dump(pair, jf, ensure_ascii=False);
            jf.write("\n")
            iw.writerow([pid])

    print(f"[OK] selected_topN={len(picked)} (target={args.target})")
    print(f"[OK] wrote pairs:  {outp}")
    print(f"[OK] wrote ids:    {idcsv}")


if __name__ == "__main__":
    main()
