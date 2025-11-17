#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as w:
        for r in rows: w.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_map(map_csv: Path):
    if not map_csv: return {}
    idx = {}
    with map_csv.open(newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            pmcid = (row.get("pmcid") or row.get("PMCID") or "").strip()
            if not pmcid: continue
            idx[pmcid] = {
                "doi": (row.get("doi") or "").strip(),
                "pubmed_url": (row.get("pubmed_url") or "").strip(),
                "original_article_url": (row.get("original_article_url") or "").strip(),
            }
    return idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="pmc_03 output jsonl (one section per line)")
    ap.add_argument("--out", required=True, help="grouped jsonl (one article per line)")
    ap.add_argument("--map", default="", help="pmc_map_from_urls.csv (to enrich meta: doi, pubmed_url...)")
    args = ap.parse_args()

    map_idx = load_map(Path(args.map)) if args.map else {}

    by_key = defaultdict(lambda: {
        "protocol_id": "", "pmcid": "", "domain": "", "title": "",
        "sections": {}, "stats": {}, "meta": {}
    })

    with open(args.inp, "r", encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            pid = r.get("protocol_id", "")
            pmcid = r.get("pmcid", "")
            if not pid or not pmcid:
                continue
            key = (pid, pmcid)

            g = by_key[key]
            g["protocol_id"] = pid
            g["pmcid"] = pmcid
            g["domain"] = r.get("domain") or g.get("domain") or "Unknown"

            # 섹션 합치기 (같은 title이면 더 긴 쪽)
            title = r.get("title") or "Methods"
            text = r.get("text") or ""
            if not text:
                continue
            prev = g["sections"].get(title, "")
            if len(text) > len(prev):
                g["sections"][title] = text
                g["stats"][title] = {
                    "chars": len(text),
                    "tokens": len(text.split()),
                    "source": r.get("source") or r.get("match_type") or ""
                }

    # meta 보강(doi/pubmed_url/원문) - pmcid 기준 join
    for (_, pmcid), rec in by_key.items():
        meta = rec.get("meta") or {}
        m = map_idx.get(pmcid, {})
        for k in ("doi", "pubmed_url", "original_article_url"):
            if m.get(k) and not meta.get(k):
                meta[k] = m[k]
        rec["meta"] = meta

    rows = [rec for rec in by_key.values() if rec["sections"]]
    write_jsonl(Path(args.out), rows)
    print(f"[OK] grouped {len(rows)} articles -> {args.out}")


if __name__ == "__main__":
    main()
