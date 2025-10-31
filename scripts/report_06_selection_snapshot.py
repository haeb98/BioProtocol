#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
report_06_selection_snapshot.py
- TopN으로 선택된 ID 목록과 전수 리포트(csv)를 합쳐,
  선택 사유(지표)와 분포 요약을 산출.
"""

import argparse
import csv
import json
import pathlib
import statistics as stats
from collections import Counter, defaultdict


def load_ids_csv(path):
    ids = []
    with open(path, "r", encoding="utf-8") as fin:
        rdr = csv.DictReader(fin)
        for row in rdr:
            pid = row.get("protocol_id", "").strip()
            if pid: ids.append(pid)
    return ids


def load_report_all(path):
    rows = []
    with open(path, "r", encoding="utf-8") as fin:
        rdr = csv.DictReader(fin)
        for row in rdr:
            rows.append(row)
    by_id = {row["protocol_id"]: row for row in rows}
    return rows, by_id


def to_float(x):
    try:
        return float(x)
    except:
        return 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", required=True, help="data/splits/test_biop_ids_top15.csv")
    ap.add_argument("--report", required=True, help="data/gold/testset_consistency_report_all.csv")
    ap.add_argument("--bio", required=False, default="data/raw/bio_protocol.json")
    ap.add_argument("--outdir", default="reports/top15_snapshot")
    args = ap.parse_args()

    outd = pathlib.Path(args.outdir);
    outd.mkdir(parents=True, exist_ok=True)

    ids = load_ids_csv(args.ids)
    rows_all, by_id = load_report_all(args.report)

    picked = []
    for pid in ids:
        r = by_id.get(pid)
        if r: picked.append(r)

    # 개별 표: 선택된 15개 지표
    out_csv = outd / "top15_metrics.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as fout:
        w = csv.writer(fout)
        w.writerow(["protocol_id", "domain", "pmcid",
                    "action_overlap", "material_overlap", "param_coverage",
                    "title_sim", "keyword_sim", "sections_total_chars",
                    "section_list", "suspect_mismatch", "notes"])
        for r in picked:
            w.writerow([
                r["protocol_id"], r.get("domain", ""), r.get("pmcid", ""),
                r.get("action_overlap", "0"), r.get("material_overlap", "0"), r.get("param_coverage", "0"),
                r.get("title_sim", "0"), r.get("keyword_sim", "0"), r.get("sections_total_chars", "0"),
                r.get("section_list", ""), r.get("suspect_mismatch", "0"), r.get("notes", "")
            ])

    # 분포 요약
    def col(vals):
        vals = [to_float(x) for x in vals if x not in (None, "")]
        if not vals:
            return {"n": 0, "mean": 0, "median": 0, "p25": 0, "p75": 0, "min": 0, "max": 0}
        vals_sorted = sorted(vals)
        n = len(vals_sorted)
        mean = sum(vals_sorted) / n
        med = stats.median(vals_sorted)
        p25 = vals_sorted[int(0.25 * (n - 1))]
        p75 = vals_sorted[int(0.75 * (n - 1))]
        return {"n": n, "mean": round(mean, 3), "median": round(med, 3),
                "p25": round(p25, 3), "p75": round(p75, 3),
                "min": round(vals_sorted[0], 3), "max": round(vals_sorted[-1], 3)}

    domains = [r.get("domain", "Unknown") for r in picked]
    dom_counts = Counter(domains)

    metrics = {
        "action_overlap": col([r.get("action_overlap", "0") for r in picked]),
        "material_overlap": col([r.get("material_overlap", "0") for r in picked]),
        "param_coverage": col([r.get("param_coverage", "0") for r in picked]),
        "title_sim": col([r.get("title_sim", "0") for r in picked]),
        "keyword_sim": col([r.get("keyword_sim", "0") for r in picked]),
        "sections_total_chars": col([r.get("sections_total_chars", "0") for r in picked]),
    }

    with open(outd / "top15_summary.json", "w", encoding="utf-8") as jout:
        json.dump({
            "n_selected": len(picked),
            "domain_distribution": dom_counts,
            "metrics_summary": metrics,
        }, jout, ensure_ascii=False, indent=2)

    # 도메인별 평균
    by_dom = defaultdict(list)
    for r in picked: by_dom[r.get("domain", "Unknown")].append(r)
    with open(outd / "top15_by_domain.csv", "w", newline="", encoding="utf-8") as fout2:
        w = csv.writer(fout2)
        w.writerow(["domain", "n", "action_overlap_mean", "param_coverage_mean", "material_overlap_mean"])
        for d, arr in sorted(by_dom.items(), key=lambda x: x[0]):
            w.writerow([d, len(arr),
                        round(sum(to_float(a.get("action_overlap", "0")) for a in arr) / len(arr), 3),
                        round(sum(to_float(a.get("param_coverage", "0")) for a in arr) / len(arr), 3),
                        round(sum(to_float(a.get("material_overlap", "0")) for a in arr) / len(arr), 3)
                        ])

    print(f"[OK] wrote: {out_csv}")
    print(f"[OK] wrote: {outd / 'top15_summary.json'}")
    print(f"[OK] wrote: {outd / 'top15_by_domain.csv'}")


if __name__ == "__main__":
    main()
