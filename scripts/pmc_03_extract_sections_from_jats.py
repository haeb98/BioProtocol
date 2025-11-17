#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
pmc_03_extract_sections_from_jats.py  (domain from bio_protocol.json)

변경 요점
- --bio 인자 추가: data/raw/bio_protocol.json에서 protocol_id → domain 매핑 생성
- 출력 레코드의 "domain"은 항상 bio_protocol.json에서 조회한 값으로 채움(없으면 "unknown")

기타
- --ids 제공 시 해당 protocol_id만 처리(테스트셋 제한)
- Methods 계열 섹션 탐지: sec-type → title → (짧으면) subsec 병합 → (여전히 부족하면) 휴리스틱
"""

import argparse
import csv
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

# 기본 Methods 계열 타이틀 후보(소문자 비교)
DEFAULT_METHOD_TITLES = [
    "materials and methods", "material and methods", "methods", "methods and materials",
    "experimental procedures", "experimental procedure", "experimental methods",
    "patients and methods", "materials & methods", "methodology", "methods/design",
    "protocol", "procedure"
]

# 휴리스틱: 단위/행동동사 검출용 (보편적 토큰만)
UNIT_RX = re.compile(
    r"\b(\d+(?:\.\d+)?)\s?(mL|ml|µl|μl|l|g|mg|µg|μg|kg|M|mM|nM|µM|μM|%|°C|min|h|hr|hrs|hours|sec|s)\b",
    re.I
)
VERB_RX = re.compile(
    r"\b(incubate|mix|centrifug(?:e|ation)|add|wash|prepare|measure|dilute|heat|cool|spin|pipett?e|"
    r"resuspend|vortex|aliquot|culture|transfect|plate|stain|fix|analy[sz]e|extract|elute|load|run)\b",
    re.I
)


def norm(s: str) -> str:
    s = (s or "").strip().lower()
    return re.sub(r"[^a-z0-9\s]+", "", s)


def localname(tag: str) -> str:
    return tag.split('}', 1)[1] if '}' in tag else tag


def parse_xml_strip_ns(xml_path: Path):
    it = ET.iterparse(str(xml_path))
    for _, el in it:
        if '}' in el.tag:
            el.tag = el.tag.split('}', 1)[1]
    return it.root


def remove_unwanted_subtrees(root):
    kill = {"table-wrap", "table", "fig", "fig-group", "graphic", "media", "caption", "supplementary-material"}
    for el in list(root.iter()):
        if localname(el.tag) in kill:
            el.clear()
            el.text = ""


def text_of(el) -> str:
    parts = []
    for t in el.itertext():
        t = t.strip()
        if t:
            parts.append(t)
    return "\n".join(parts).strip()


def collect_section_text(sec) -> str:
    chunks = []
    for child in sec.iter():
        tag = localname(child.tag)
        if tag in {"title"}:
            continue
        if tag in {"p", "sec", "list", "list-item", "def-list", "formula"}:
            t = text_of(child)
            if t:
                chunks.append(t)
    txt = "\n\n".join(chunks).strip()
    txt = re.sub(r"\n{3,}", "\n\n", txt)
    return txt


def find_body(root):
    for el in root.iter():
        if localname(el.tag) == "body":
            return el
    return None


def title_of(sec) -> str:
    t = sec.find("title")
    return text_of(t) if t is not None else ""


def is_methods_like_by_type(sec) -> bool:
    sec_type = (sec.attrib.get("sec-type") or "").lower()
    return ("method" in sec_type) or ("material" in sec_type) or ("procedure" in sec_type) or ("protocol" in sec_type)


def is_methods_like_by_title(sec, matcher, extra_titles_norm):
    ttl_norm = norm(title_of(sec))
    if not ttl_norm:
        return False
    return matcher(ttl_norm) or any(x in ttl_norm for x in extra_titles_norm)


def build_title_matcher(mode: str, base_titles_norm: list[str]):
    if mode == "contains":
        return lambda s: any(t in s for t in base_titles_norm)
    if mode == "startswith":
        return lambda s: any(s.startswith(t) for t in base_titles_norm)
    if mode == "exact":
        return lambda s: any(s == t for t in base_titles_norm)
    if mode == "regex":
        rx = [re.compile(t, re.I) for t in base_titles_norm]
        return lambda s: any(r.search(s) for r in rx)
    return lambda s: any(t in s for t in base_titles_norm)


def merge_with_subsecs(sec, target_chars: int) -> str:
    chunks = [collect_section_text(sec)]
    total = len(chunks[0])
    if total >= target_chars:
        return chunks[0]
    for sub in sec.iter("sec"):
        if sub is sec:
            continue
        t = collect_section_text(sub)
        if t:
            chunks.append(t)
            total += len(t)
            if total >= target_chars:
                break
    return "\n\n".join([c for c in chunks if c]).strip()


def heuristic_from_body(body, target_chars: int) -> str:
    paras = []
    for p in body.iter("p"):
        txt = text_of(p)
        if not txt:
            continue
        if UNIT_RX.search(txt) or VERB_RX.search(txt):
            paras.append(txt)
    out, acc = [], 0
    for t in paras:
        out.append(t)
        acc += len(t)
        if acc >= target_chars:
            break
    return "\n\n".join(out).strip()


def read_map_csv(map_csv: Path) -> dict:
    """pmcid -> meta(protocol_id) : domain은 bio JSON에서 채움"""
    idx = {}
    with map_csv.open(newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        need = {"protocol_id", "pmcid"}
        if not need.issubset(set(r.fieldnames)):
            raise SystemExit(f"[ERR] {map_csv} must contain {need}, got {r.fieldnames}")
        for row in r:
            pmcid = (row.get("pmcid") or row.get("PMCID") or "").strip()
            pid = (row.get("protocol_id") or row.get("biop_id") or "").strip()
            if not pmcid or not pid:
                continue
            idx[pmcid] = {"protocol_id": pid}
    return idx


def read_ids_csv(ids_csv: Path) -> set:
    """화이트리스트 protocol_id 집합(테스트셋 제한)"""
    s = set()
    with ids_csv.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            pid = (row.get("protocol_id") or "").strip()
            if pid:
                s.add(pid)
    return s


def extract_domain_from_classification(rec: dict) -> str | None:
    cls = rec.get("classification") or {}
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


def build_domain_map_from_bio(bio_json: Path) -> dict:
    """protocol_id -> domain (없으면 'unknown')"""
    if not bio_json.exists():
        raise SystemExit(f"[ERR] not found: {bio_json}")
    data = json.loads(bio_json.read_text(encoding='utf-8'))
    domap = {}
    found, unknown = 0, 0
    for rec in data:
        pid = str(rec.get("protocol_id") or rec.get("id") or "").strip()
        if not pid:
            continue
        dom = extract_domain_from_classification(rec)
        if not dom:
            dom = (rec.get("domain") or rec.get("category") or rec.get("collection") or "").strip()
        if dom:
            domap[pid] = dom;
            found += 1
        else:
            domap[pid] = "unknown";
            unknown += 1
    print(f"[domain] mapped={found}, unknown={unknown}, total={len(domap)}")
    return domap


def write_jsonl(path: Path, records: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as w:
        for rec in records:
            w.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", required=True, help="CSV with protocol_id,pmcid (data/gold/pmc_map_from_urls.csv)")
    ap.add_argument("--jats", required=True, help="Directory of PMC*.xml (data/gold/pmc_jats)")
    ap.add_argument("--bio", required=True, help="data/raw/bio_protocol.json (for domain lookup)")
    ap.add_argument("--out", required=True, help="JSONL output (data/gold/gold_articles_sections_pmc.jsonl)")
    ap.add_argument("--log", default="runs/pmc_03_extract_log.jsonl", help="processing log jsonl")
    ap.add_argument("--ids", default="", help="(optional) CSV with 'protocol_id' to restrict extraction to test set")

    # 섹션 선택/문턱값
    ap.add_argument("--match", choices=["contains", "startswith", "exact", "regex"], default="contains")
    ap.add_argument("--min-chars", type=int, default=100)
    ap.add_argument("--merge-target-chars", type=int, default=220)
    ap.add_argument("--heuristic-target-chars", type=int, default=300)
    ap.add_argument("--extra-sections", default="", help="추가 허용 섹션 타이틀(콤마 구분)")

    args = ap.parse_args()

    jdir = Path(args.jats)
    outp = Path(args.out)
    logp = Path(args.log)

    map_idx = read_map_csv(Path(args.map))  # pmcid -> {protocol_id}
    domap = build_domain_map_from_bio(Path(args.bio))  # protocol_id -> domain
    ids_whitelist = read_ids_csv(Path(args.ids)) if args.ids else set()

    base_titles = [norm(t) for t in DEFAULT_METHOD_TITLES]
    extra_titles_norm = [norm(x) for x in args.extra_sections.split(",") if x.strip()]
    matcher = build_title_matcher(args.match, base_titles)

    logs = []
    out_rows = []

    xml_paths = sorted(jdir.glob("PMC*.xml"))
    for xmlp in xml_paths:
        pmcid = xmlp.stem
        meta = map_idx.get(pmcid)
        if not meta:
            logs.append({"pmcid": pmcid, "result": "skip_no_map", "xml_path": str(xmlp)})
            continue

        pid = meta["protocol_id"]
        if ids_whitelist and pid not in ids_whitelist:
            logs.append({"pmcid": pmcid, "protocol_id": pid, "result": "skip_not_in_ids"})
            continue

        # 파싱
        try:
            root = parse_xml_strip_ns(xmlp)
        except Exception as e:
            logs.append({"pmcid": pmcid, "protocol_id": pid, "result": "parse_error", "error": str(e)})
            continue

        body = find_body(root)
        if body is None:
            logs.append({"pmcid": pmcid, "protocol_id": pid, "result": "no_body"})
            continue

        remove_unwanted_subtrees(body)

        picked = None
        picked_src = None
        picked_match = None
        picked_sectype = None
        picked_title = None

        # 1) sec-type 우선
        for sec in body.iter("sec"):
            if is_methods_like_by_type(sec):
                txt = collect_section_text(sec)
                if len(txt) >= args.min_chars:
                    picked = txt
                    picked_src = "regular"
                    picked_match = "sec-type"
                    picked_sectype = (sec.attrib.get("sec-type") or "").lower()
                    picked_title = title_of(sec)
                    break

        # 2) title 매칭
        if not picked:
            for sec in body.iter("sec"):
                if is_methods_like_by_title(sec, matcher, extra_titles_norm):
                    txt = collect_section_text(sec)
                    if len(txt) >= args.min_chars:
                        picked = txt
                        picked_src = "regular"
                        picked_match = "title"
                        picked_sectype = (sec.attrib.get("sec-type") or "").lower()
                        picked_title = title_of(sec)
                        break

        # 3) merge (짧으면 하위 섹션 병합)
        if not picked:
            candidates = []
            for sec in body.iter("sec"):
                stype_ok = is_methods_like_by_type(sec)
                title_ok = is_methods_like_by_title(sec, matcher, extra_titles_norm)
                if stype_ok or title_ok:
                    candidates.append((sec, "sec-type" if stype_ok else "title"))
            for sec, mtype in candidates:
                merged = merge_with_subsecs(sec, args.merge_target_chars)
                if len(merged) >= args.merge_target_chars:
                    picked = merged
                    picked_src = "merged"
                    picked_match = mtype
                    picked_sectype = (sec.attrib.get("sec-type") or "").lower()
                    picked_title = title_of(sec)
                    break

        # 4) heuristic
        if not picked:
            htxt = heuristic_from_body(body, args.heuristic_target_chars)
            if len(htxt) >= args.min_chars:
                picked = htxt
                picked_src = "heuristic"
                picked_match = "heuristic"
                picked_sectype = ""
                picked_title = "Heuristic Methods"

        if not picked:
            logs.append({
                "pmcid": pmcid,
                "protocol_id": pid,
                "result": "no_sections_matched",
                "xml_path": str(xmlp)
            })
            continue

        domain_val = domap.get(pid, "unknown")
        out_rows.append({
            "protocol_id": pid,
            "pmcid": pmcid,
            "domain": domain_val,
            "title": picked_title,
            "text": picked,
            "source": picked_src,
            "sec_type": picked_sectype,
            "chars": len(picked),
            "xml_path": str(xmlp),
            "match_type": picked_match
        })

        logs.append({
            "pmcid": pmcid,
            "protocol_id": pid,
            "domain": domain_val,
            "result": "ok",
            "source": picked_src,
            "match_type": picked_match,
            "sec_type": picked_sectype,
            "title": picked_title,
            "chars": len(picked)
        })

    write_jsonl(outp, out_rows)
    write_jsonl(logp, logs)

    print(f"[OK] extracted sections: {len(out_rows)} -> {outp}")
    print(f"[OK] log: {logp}")


if __name__ == "__main__":
    main()
