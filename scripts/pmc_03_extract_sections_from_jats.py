#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pmc_03_extract_sections_from_jats.py (v2)

- Namespace-safe (local-name())
- sec/@sec-type aware
- Fuzzy title matching (exact/startswith/contains/regex) + normalization (&->and, punctuation strip)
- Merge short subsections under same parent to reach length threshold
- Heuristic fallback: harvest body paragraphs with action/param signals
- Detect publisher-blocked XML / no <body>
- Annotate section source: regular | merged | heuristic
- Rich logging: reasons per PMCID

Usage:
python scripts/pmc_03_extract_sections_from_jats.py \
  --map data/gold/pmc_map_from_urls.csv \
  --jats data/gold/pmc_jats \
  --out data/gold/gold_articles_sections_pmc.jsonl \
  --log runs/pmc_03_extract_log.jsonl \
  --extra-sections "Abstract,Supplementary Methods,Results" \
  --match contains --min-chars 160 --merge-target-chars 220
"""

import argparse
import csv
import json
import pathlib
import re
from typing import Dict, Tuple

from lxml import etree

DEFAULT_SECTIONS = [
    "methods", "materials and methods", "methods and materials",
    "material and methods", "experimental procedures",
    "methodology", "protocol", "procedure", "experimental"
]
DEFAULT_SEC_TYPES = {
    "methods", "materials|methods", "materials-and-methods", "methodology", "experimental-procedures"
}

# action/param signals for heuristic fallback
ACTION_LEX = {
    "incubate", "centrifuge", "mix", "add", "transfer", "pipette", "aliquot", "vortex",
    "resuspend", "pellet", "wash", "dry", "filter", "measure", "dilute", "prepare",
    "heat", "cool", "shake", "spin", "stain", "fix", "mount", "label", "load", "elute"
}
RE_PARAM = re.compile(r"\b(\d+(?:\.\d+)?)\s*(°c|degc|s|min|h|hr|hrs|ms|rpm|g|xg|µl|ul|ml|l|%)\b", re.I)


def norm_title(s: str) -> str:
    s = (s or "").strip().lower()
    s = s.replace("&", " and ")
    s = re.sub(r"[-_/]+", " ", s)
    s = re.sub(r"[\[\]\(\):,]", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s


def fuzzy_match(tnorm: str, targets: set, mode: str) -> bool:
    if mode == "exact": return tnorm in targets
    if mode == "startswith": return any(tnorm.startswith(x) for x in targets)
    if mode == "contains": return any(x in tnorm for x in targets)
    # regex
    return any(re.search(p, tnorm) for p in targets)


def strip_tables_figs(node):
    for bad in node.xpath('.//*[local-name()="table-wrap" or local-name()="fig" or local-name()="caption"]'):
        bad.getparent().remove(bad)


def get_text(node) -> str:
    return " ".join("".join(node.itertext()).split())


def section_title(node) -> str:
    t = node.xpath('./*[local-name()="title"][1]')
    if not t: return ""
    return get_text(t[0])


def find_body(root):
    b = root.xpath('.//*[local-name()="body"]')
    return b[0] if b else None


def detect_publisher_block(root) -> bool:
    # Scan comments for known blocking messages
    for c in root.xpath('//comment()'):
        if "publisher does not allow downloading of the full text in xml form" in (c.text or "").lower():
            return True
    return False


def from_nodes_collect_sections(nodes, targets: set, sec_types: set, match_mode: str, min_chars: int):
    got = {}  # title -> (text, source)
    for sec in nodes:
        st = (sec.get("sec-type") or "").strip().lower()
        title = section_title(sec)
        title_norm = norm_title(title)
        keep = False
        if st and st in sec_types:
            keep = True
        if not keep and title_norm:
            if fuzzy_match(title_norm, targets, match_mode):
                keep = True
        if keep:
            sec_copy = etree.fromstring(etree.tostring(sec))
            strip_tables_figs(sec_copy)
            txt = get_text(sec_copy)
            if len(txt) >= min_chars:
                got[title or st or "UNTITLED"] = (txt, "regular")
    return got


def merge_short_subsecs(root, parent_min_merge: int, targets: set, sec_types: set, match_mode: str, min_chars: int):
    """
    For each top-level sec that roughly matches targets/sec-type, merge its short subsec texts.
    """
    merged = {}
    for sec in root.xpath('.//*[local-name()="sec"]'):
        st = (sec.get("sec-type") or "").strip().lower()
        ttl = section_title(sec)
        tnorm = norm_title(ttl)
        hint = False
        if st and st in sec_types: hint = True
        if not hint and tnorm: hint = fuzzy_match(tnorm, targets, match_mode)
        if not hint: continue

        # merge all subsec texts (even if short)
        sec_copy = etree.fromstring(etree.tostring(sec))
        strip_tables_figs(sec_copy)
        parts = []
        # include own paragraph
        own_paras = sec_copy.xpath('./*[local-name()="p"]')
        parts += [get_text(p) for p in own_paras]
        # include subsec paragraphs
        for sub in sec_copy.xpath('.//*[local-name()="subsec"]'):
            parts += [get_text(p) for p in sub.xpath('./*[local-name()="p"]')]
        merged_txt = " ".join([p for p in parts if p])
        if len(merged_txt) >= parent_min_merge:
            merged[ttl or st or "MERGED_SECTION"] = (merged_txt, "merged")
    return merged


def heuristic_methods(root, target_chars: int) -> Tuple[str, bool]:
    """
    Collect body paragraphs that include action verbs or number+unit.
    Returns (text, ok)
    """
    body = find_body(root)
    if not body: return "", False
    paras = body.xpath('.//*[local-name()="p"]')
    buf = []
    for p in paras:
        txt = get_text(p).strip()
        if not txt: continue
        low = txt.lower()
        has_action = any(a in low for a in ACTION_LEX)
        has_param = bool(RE_PARAM.search(low))
        if has_action or has_param:
            buf.append(txt)
        if sum(len(x) for x in buf) >= target_chars:
            break
    if sum(len(x) for x in buf) >= target_chars:
        return "\n\n".join(buf), True
    return "", False


def read_map_csv(path: pathlib.Path) -> Dict[str, Dict]:
    rows = list(csv.DictReader(open(path, "r", encoding="utf-8")))
    idx = {}
    for r in rows:
        pmc = (r.get("pmcid") or "").strip()
        if not pmc: continue
        idx[pmc] = {
            "protocol_id": (r.get("protocol_id") or "").strip(),
            "domain": (r.get("biop_domain") or "").strip(),
            "title_biop": (r.get("biop_title") or "").strip()
        }
    return idx


def extract_article_meta(root) -> Dict:
    def one(xp):
        n = root.xpath(xp)
        if not n: return ""
        return get_text(n[0]) if isinstance(n[0], etree._Element) else str(n[0])

    meta = {}
    meta["article_title"] = one('.//*[local-name()="article-title"][1]')
    meta["journal_title"] = one('.//*[local-name()="journal-title"][1]')
    y = root.xpath('.//*[local-name()="pub-date"]/*[local-name()="year"][1]')
    meta["year"] = get_text(y[0]) if y else ""
    doi = root.xpath('.//*[@pub-id-type="doi"][1]')
    meta["doi"] = get_text(doi[0]) if doi else ""
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", required=True)
    ap.add_argument("--jats", required=True)
    ap.add_argument("--out", default="data/gold/gold_articles_sections_pmc.jsonl")
    ap.add_argument("--log", default="runs/pmc_03_extract_log.jsonl")
    ap.add_argument("--sections", default=",".join(DEFAULT_SECTIONS))
    ap.add_argument("--extra-sections", default="")
    ap.add_argument("--sec-types", default=",".join(DEFAULT_SEC_TYPES))
    ap.add_argument("--match", choices=["exact", "contains", "startswith", "regex"], default="contains")
    ap.add_argument("--min-chars", type=int, default=160)
    ap.add_argument("--merge-target-chars", type=int, default=220)
    ap.add_argument("--heuristic-target-chars", type=int, default=300)
    args = ap.parse_args()

    targets = {norm_title(x) for x in args.sections.split(",") if x.strip()}
    if args.extra_sections:
        targets |= {norm_title(x) for x in args.extra_sections.split(",") if x.strip()}
    sec_types = {x.strip().lower() for x in args.sec_types.split(",") if x.strip()}

    map_idx = read_map_csv(pathlib.Path(args.map))
    jdir = pathlib.Path(args.jats)
    outp = pathlib.Path(args.out);
    outp.parent.mkdir(parents=True, exist_ok=True)
    logp = pathlib.Path(args.log);
    logp.parent.mkdir(parents=True, exist_ok=True)

    logs = [];
    n_ok = 0
    with open(outp, "w", encoding="utf-8") as fout:
        for xmlp in sorted(jdir.glob("PMC*.xml")):
            pmcid = xmlp.stem
            meta = map_idx.get(pmcid)
            if not meta:
                logs.append({"pmcid": pmcid, "result": "skip_no_map"});
                continue
            try:
                root = etree.parse(str(xmlp)).getroot()
            except Exception as e:
                logs.append({"pmcid": pmcid, "result": "parse_error", "error": str(e)});
                continue

            blocked = detect_publisher_block(root)
            has_body = bool(find_body(root))

            # 1) regular sections by title/sec-type
            got = from_nodes_collect_sections(
                root.xpath('.//*[local-name()="sec"]'), targets, sec_types, args.match, args.min_chars
            )

            # 2) merged parent if short subsecs
            merged = merge_short_subsecs(root, args.merge_target_chars, targets, sec_types, args.match, args.min_chars)
            for k, (v, src) in merged.items():
                if k not in got: got[k] = (v, src)

            # 3) heuristic fallback when still empty
            if not got:
                htxt, ok = heuristic_methods(root, args.heuristic_target_chars)
                if ok:
                    got["Heuristic Methods"] = (htxt, "heuristic")

            if not got:
                reason = "no_sections_matched"
                if not has_body: reason = "no_body"
                if blocked: reason = "publisher_blocked_xml"
                logs.append({"pmcid": pmcid, "result": reason})
                continue

            art = extract_article_meta(root)
            # pack
            secs = {k: v for k, (v, src) in got.items()}
            srcs = {k: src for k, (v, src) in got.items()}
            stats = {k: {"chars": len(v), "tokens": len(v.split()), "source": srcs[k]} for k, v in secs.items()}

            rec = {
                "protocol_id": meta["protocol_id"],
                "pmcid": pmcid,
                "domain": meta.get("domain", ""),
                "title": art.get("article_title", ""),
                "xml_path": str(xmlp),
                "sections": secs,
                "stats": stats,
                "meta": {
                    "journal": art.get("journal_title", ""),
                    "year": art.get("year", ""),
                    "doi": art.get("doi", ""),
                    "publisher_blocked": blocked,
                    "no_body": not has_body
                }
            }
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            logs.append({"pmcid": pmcid, "result": "ok", "sections": list(secs.keys()), "sources": srcs})
            n_ok += 1

    with open(logp, "w", encoding="utf-8") as lf:
        for r in logs: lf.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[OK] {outp} (rows={n_ok})")
    print(f"[LOG] {logp}")


if __name__ == "__main__":
    main()
