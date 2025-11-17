#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Pick test set ONLY from XML-ready articles.

Flow:
  pmc_map_from_urls.csv  ─┐
                          ├─> filter by status (ok/low_conf) → check JATS file exists
  data/gold/pmc_jats/  ───┘                                 → check <body> exists
  data/raw/bio_protocol.json → domain, title

Outputs:
  - data/splits/test_biop_ids.csv            (ONLY protocol_id; pmc_04와 100% 호환)
  - data/splits/domain_counts_xmlready.csv   (eligible 도메인 분포)
  - reports/xml_ready_candidates.csv         (진단용: 후보 전량 + has_xml/has_body/why_not)
"""

# !/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Pick test set ONLY from XML-ready articles.

Flow:
  - pmc_map_from_urls.csv: filter by status (ok/low_conf) and has PMCID
  - data/gold/pmc_jats/: require XML file exists; optionally require <body> and Methods-like section
  - data/raw/bio_protocol.json: lookup domain (classification.primary_domain preferred)

Outputs:
  - data/splits/test_biop_ids.csv              (IDs only; pmc_04와 100% 호환)
  - data/splits/domain_counts_pubmed.csv       (eligible distribution and picked count by domain)
  - reports/xml_ready_candidates.csv           (diagnostic: why_not, has_xml/body/methods)
"""

import argparse
import csv
import json
import random
import re
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict, Counter
from pathlib import Path

# Methods-like titles seen in JATS
METHOD_TITLES = [
    "materials and methods", "material and methods", "methods", "methods and materials",
    "experimental procedures", "experimental procedure", "experimental methods",
    "patients and methods", "materials & methods", "methodology", "methods/design"
]


def localname(tag: str) -> str:
    return tag.split('}', 1)[1] if '}' in tag else tag


def read_map_csv(p: Path):
    rows = []
    with p.open(newline='', encoding='utf-8') as f:
        r = csv.DictReader(f)
        need = {"protocol_id", "pmcid"}
        if not need.issubset(set(r.fieldnames)):
            raise SystemExit(f"[ERR] {p} must contain {need}, got {r.fieldnames}")
        for rec in r:
            rows.append(rec)
    return rows


def extract_domain_from_classification(rec: dict) -> str | None:
    cls = rec.get("classification") or {}
    # 가장 흔한 필드
    cand = [
        cls.get("primary_domain"),
        cls.get("domain"),
        cls.get("primaryTopic"),
        cls.get("topic"),
    ]
    for c in cand:
        if c and str(c).strip():
            return str(c).strip()
    return None


def read_bio_protocol_json(p: Path):
    """
    Expect list of records like:
      { "id": "BPXXXX", "title": "...", "classification": { "primary_domain": "Cell Biology", ... }, ... }
    """
    if not p.exists():
        raise SystemExit(f"[ERR] not found: {p}")
    data = json.loads(p.read_text(encoding='utf-8'))
    domap = {}
    found, unknown = 0, 0
    for rec in data:
        pid = str(rec.get("protocol_id") or rec.get("id") or "").strip()
        if not pid:
            continue
        dom = extract_domain_from_classification(rec)
        if not dom:
            # 폴백 후보(top-level에서 혹시 존재하면)
            dom = (rec.get("domain") or rec.get("category") or rec.get("collection") or "").strip()
        if dom:
            domap[pid] = dom
            found += 1
        else:
            domap[pid] = "unknown"
            unknown += 1
    # 간단 통계
    print(f"[domain] mapped={found}, unknown={unknown} (total={len(domap)})", file=sys.stderr)
    return domap


def parse_xml_strip_ns(xml_path: Path):
    it = ET.iterparse(str(xml_path))
    for _, el in it:
        if '}' in el.tag:
            el.tag = el.tag.split('}', 1)[1]
    return it.root


def has_body(xml_path: Path) -> bool:
    try:
        root = parse_xml_strip_ns(xml_path)
        return any(el.tag == "body" for el in root.iter())
    except Exception:
        return False


def is_methods_sec(sec) -> bool:
    sec_type = (sec.attrib.get('sec-type') or "").lower()
    if any(k in sec_type for k in ["methods", "materials"]):
        return True
    for ch in sec:
        if localname(ch.tag) == "title":
            t = "".join(ch.itertext()).strip().lower()
            norm = re.sub(r"[^a-z0-9\s]+", "", t)
            for pat in METHOD_TITLES:
                if pat in norm:
                    return True
    return False


def has_methods(xml_path: Path) -> bool:
    try:
        root = parse_xml_strip_ns(xml_path)
        body = next((b for b in root.iter() if b.tag == "body"), None)
        if body is None:
            return False
        for s in body.iter("sec"):
            if is_methods_sec(s):
                return True
            for sub in s.iter("sec"):
                if is_methods_sec(sub):
                    return True
        return False
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", required=True, help="data/gold/pmc_map_from_urls.csv")
    ap.add_argument("--jats-dir", dest="jats_dir", required=True, help="data/gold/pmc_jats")
    ap.add_argument("--bio", required=True, help="data/raw/bio_protocol.json")
    ap.add_argument("--accept-status", default="ok,low_conf", help="accept states (comma)")
    ap.add_argument("--require-body", action="store_true")
    ap.add_argument("--require-methods", action="store_true")
    ap.add_argument("--min-per-domain", type=int, default=30)
    ap.add_argument("--take-per-domain", type=int, default=5)
    ap.add_argument("--max-total", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-ids", default="data/splits/test_biop_ids.csv")
    ap.add_argument("--out-domain-counts", default="data/splits/domain_counts_pubmed.csv")
    ap.add_argument("--out-report", default="reports/xml_ready_candidates.csv")
    args = ap.parse_args()

    random.seed(args.seed)
    jdir = Path(args.jats_dir)

    # 1) 입력 로드
    rows = read_map_csv(Path(args.map))
    domap = read_bio_protocol_json(Path(args.bio))
    okset = {s.strip().lower() for s in args.accept_status.split(",") if s.strip()}

    # 2) XML/본문/Methods 필터링
    candidates = []
    eligibles = []

    for r in rows:
        st = (r.get("status") or r.get("state") or "").strip().lower()
        if st and st not in okset:
            continue
        pmcid = (r.get("pmcid") or r.get("PMCID") or "").strip()
        pid = (r.get("protocol_id") or r.get("biop_id") or "").strip()
        if not pmcid or not pid:
            continue

        xmlp = jdir / f"{pmcid}.xml"
        has_xml = xmlp.exists()
        has_bod = has_body(xmlp) if has_xml else False
        has_meth = has_methods(xmlp) if (has_xml and has_bod and args.require_methods) else (
                    args.require_methods is False)

        why_not = ""
        if not has_xml:
            why_not = "no_xml_file"
        elif args.require_body and not has_bod:
            why_not = "no_body"
        elif args.require_methods and not has_meth:
            why_not = "no_methods"

        domain = domap.get(pid, "unknown")

        candidates.append({
            "protocol_id": pid, "pmcid": pmcid, "status": st,
            "domain": domain, "xml_path": str(xmlp),
            "has_xml": "y" if has_xml else "n",
            "has_body": "y" if has_bod else "n",
            "has_methods": "y" if (has_meth if args.require_methods else has_bod) else "n",
            "why_not": why_not
        })

        if has_xml and (not args.require_body or has_bod) and (not args.require_methods or has_meth):
            eligibles.append({"protocol_id": pid, "domain": domain})

    # 3) 리포트/도메인 카운트
    Path(args.out_report).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_report, "w", newline="", encoding="utf-8") as f:
        fns = ["protocol_id", "pmcid", "status", "domain", "xml_path", "has_xml", "has_body", "has_methods", "why_not"]
        w = csv.DictWriter(f, fieldnames=fns);
        w.writeheader()
        for row in candidates: w.writerow(row)

    dom_counts_all = Counter([e["domain"] for e in eligibles])

    # 4) 도메인 균등 샘플링
    pool_by_dom = defaultdict(list)
    for e in eligibles:
        pool_by_dom[e["domain"]].append(e)

    # min-per-domain 필터
    pool_by_dom = {d: arr for d, arr in pool_by_dom.items() if len(arr) >= args.min_per_domain}

    picked = []
    for d, arr in pool_by_dom.items():
        random.shuffle(arr)
        picked.extend(arr[:args.take_per_domain])

    if len(picked) > args.max_total:
        random.shuffle(picked)
        picked = picked[:args.max_total]

    # 5) 저장 (IDs only)
    Path(args.out_ids).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_ids, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f);
        w.writerow(["protocol_id"])
        for r in picked: w.writerow([r["protocol_id"]])

    # 6) 도메인 요약 저장
    with open(args.out_domain_counts, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["domain", "count_all_candidates", "count_after_min_filter", "picked"])
        for d in sorted(dom_counts_all.keys(), key=lambda x: (-dom_counts_all[x], x)):
            after = len(pool_by_dom.get(d, []))
            pickd = sum(1 for x in picked if x["domain"] == d)
            w.writerow([d, dom_counts_all[d], after, pickd])

    print(f"[INFO] eligible(after XML/body/methods filters) = {sum(dom_counts_all.values())}", file=sys.stderr)
    print(f"[INFO] unique_domains = {len(dom_counts_all)}", file=sys.stderr)
    print(f"[OK] wrote IDs to {args.out_ids}", file=sys.stderr)
    print(f"[OK] wrote domain counts to {args.out_domain_counts}", file=sys.stderr)
    print(f"[OK] wrote diagnostic to {args.out_report}", file=sys.stderr)


if __name__ == "__main__":
    main()
