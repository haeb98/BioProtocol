#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pmc_04_build_gold_pairs_for_testset.py (v3)
- 느슨/엄격 임계 모두 지원
- 섹션 whitelist/blacklist + prefer_regular
- 페어(JSONL)와 리포트(CSV) 항상 생성
"""

import argparse
import csv
import json
import pathlib
import re
from collections import defaultdict

# orjson이 있으면 사용 (속도↑), 없으면 표준 json로 대체
try:
    import orjson as _orjson


    def jloads(b):
        return _orjson.loads(b)
except Exception:
    import json as _orjson


    def jloads(b):
        return _orjson.loads(b)

ACTIONS = {"incubate", "centrifuge", "mix", "add", "transfer", "pipette", "aliquot", "vortex",
           "resuspend", "pellet", "wash", "dry", "filter", "measure", "dilute", "prepare",
           "heat", "cool", "shake", "spin", "stain", "fix", "mount", "label", "load", "elute"}

MATERIAL_HINTS = {"buffer", "pbs", "ethanol", "methanol", "bleach", "nacl", "glycerol", "yeast", "agar", "water", "h2o",
                  "dmso", "triton", "edta", "mgso4", "kcl", "k2hpo4", "kh2po4", "nh4", "medium", "media", "reagent",
                  "solution"}

RE_PARAM = re.compile(r"\b(\d+(?:\.\d+)?)\s*(°c|degc|s|min|h|hr|hrs|ms|rpm|g|xg|µl|ul|ml|l|%)\b", re.I)
TOKSPLIT = re.compile(r"\W+")


def tokset(s): return set(t for t in TOKSPLIT.split((s or "").lower()) if t)


def jacc(a, b): return (len(a & b) / max(1, len(a | b))) if (a or b) else 0.0


def extract_actions(t): return sorted(list(ACTIONS & tokset(t)))


def extract_materials(t):
    toks = tokset(t);
    mats = set()
    for h in MATERIAL_HINTS:
        if h in toks: mats.add(h)
    for tt in list(toks):
        if re.match(r"^(nacl|kcl|pbs|h2o|ethanol|methanol|glycerol|triton|edta|dmso)$", tt):
            mats.add(tt)
    return sorted(list(mats))


def extract_params(t): return [m.group(0) for m in RE_PARAM.finditer(t or "")]


def flatten_hier(h):
    if not isinstance(h, dict): return ""

    def keyer(k):
        return [int(p) if p.isdigit() else p for p in k.split(".")]

    lines = []
    for k in sorted(h.keys(), key=keyer):
        v = h[k]
        if isinstance(v, dict) and "title" in v:
            lines.append(v["title"])
        elif isinstance(v, str):
            lines.append(v)
    return "\n".join(lines)


def load_bio(path):
    data = jloads(pathlib.Path(path).read_bytes())
    return {str(r.get("id") or "").strip(): r for r in data if r.get("id")}


def load_ids(path):
    rows = set()
    with open(path, "r", encoding="utf-8") as f:
        hdr = f.readline().strip().split(",")
        idx = {h: i for i, h in enumerate(hdr)}
        for line in f:
            cells = line.rstrip("\n").split(",")
            pid = cells[idx.get("protocol_id", 0)].strip() if cells else ""
            if pid: rows.add(pid)
    return rows


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bio", required=True)
    ap.add_argument("--ids", required=True)
    ap.add_argument("--arts", required=True)
    ap.add_argument("--out", default="data/gold/gold_pairs_testset.jsonl")
    ap.add_argument("--report", default="data/gold/testset_consistency_report.csv")
    ap.add_argument("--min-section", type=int, default=150)
    ap.add_argument("--prefer-regular", action="store_true", default=False)

    # 임계: strong 필터 (없으면 전수 출력)
    ap.add_argument("--min-action-overlap", type=float, default=0.0)
    ap.add_argument("--min-material-overlap", type=float, default=0.0)
    ap.add_argument("--min-param-coverage", type=float, default=0.0)
    ap.add_argument("--filter-mode", choices=["any", "all"], default="any")

    # 섹션 필터링
    ap.add_argument("--whitelist-sections",
                    default="methods|materials|experimental|star methods|experimental model and subject details")
    ap.add_argument("--blacklist-sections", default="results|discussion|abstract")
    args = ap.parse_args()

    bio = load_bio(args.bio)
    ids = load_ids(args.ids)
    arts = load_arts(args.arts)

    outp = pathlib.Path(args.out);
    outp.parent.mkdir(parents=True, exist_ok=True)
    rep = pathlib.Path(args.report);
    rep.parent.mkdir(parents=True, exist_ok=True)

    wl = re.compile(args.whitelist_sections, re.I) if args.whitelist_sections.strip() else None
    bl = re.compile(args.blacklist_sections, re.I) if args.blacklist_sections.strip() else None

    # 리포트 헤더 미리 작성 (항상 파일 생성 보장)
    with open(rep, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "protocol_id", "pmcid", "domain",
            "doi_match", "title_sim", "keyword_sim",
            "action_overlap", "material_overlap", "param_coverage",
            "section_list", "sections_total_chars", "sections_total_tokens",
            "suspect_mismatch", "notes"
        ])

    def pass_filter(aov, mov, pov):
        flags = []
        if args.min_action_overlap > 0:  flags.append(aov >= args.min_action_overlap)
        if args.min_material_overlap > 0: flags.append(mov >= args.min_material_overlap)
        if args.min_param_coverage > 0:  flags.append(pov >= args.min_param_coverage)
        if not flags: return True
        return all(flags) if args.filter_mode == "all" else any(flags)

    n_all = 0;
    n_written = 0
    with open(outp, "w", encoding="utf-8") as jf, open(rep, "a", newline="", encoding="utf-8") as rf:
        w = csv.writer(rf)
        for pid in sorted(ids):
            n_all += 1
            b = bio.get(pid)
            a = arts.get(pid)
            if not b or not a:
                w.writerow([pid, "", (b or {}).get("classification", {}).get("primary_domain", "Unknown"),
                            0, 0, 0, 0, 0, 0, "", 0, 0, 1, "no_bio" if not b else "no_article_sections"])
                continue

            # 섹션 필터
            raw_secs = a.get("sections") or {}
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
            sec_text = "\n\n".join([secs[k] for k in sorted(secs.keys())])
            sec_chars = len(sec_text);
            sec_toks = len((sec_text or "").split())

            prot_text = "\n".join([
                (b.get("title") or ""),
                (b.get("input") or ""),
                (b.get("protocol") or ""),
                (flatten_hier(b.get("hierarchical_protocol") or {}) or "")
            ])

            title_sim = jacc(tokset((b.get("title") or "")), tokset((a.get("title") or "")))
            keyword_sim = jacc(tokset((b.get("keywords") or "")), tokset(sec_text))

            act_b, act_m = set(extract_actions(prot_text)), set(extract_actions(sec_text))
            mat_b, mat_m = set(extract_materials(prot_text)), set(extract_materials(sec_text))
            par_b = extract_params(prot_text)

            action_overlap = jacc(act_b, act_m)
            material_overlap = jacc(mat_b, mat_m)
            param_cov = (sum(1 for p in par_b if p.lower() in sec_text.lower()) / max(1, len(par_b))) if par_b else 0.0

            doi_b = (b.get("url") or "").lower()
            doi_b = doi_b.split("doi.org/")[-1] if "doi.org/" in doi_b else ""
            doi_a = ((a.get("meta") or {}).get("doi") or "").lower()
            doi_match = int(doi_b and doi_a and (doi_b == doi_a))

            suspect = 1 if (action_overlap < 0.15 and material_overlap < 0.15 and param_cov < 0.15) else 0
            notes = "" if secs else "all_sections_too_short"

            # strong 임계가 있으면 통과분만 JSONL 기록(전수 리포트를 원하면 임계 0으로 실행)
            if not pass_filter(action_overlap, material_overlap, param_cov):
                w.writerow([
                    pid, a.get("pmcid", ""),
                    (b.get("classification") or {}).get("primary_domain", "Unknown"),
                    round(doi_match, 3), round(title_sim, 3), round(keyword_sim, 3),
                    round(action_overlap, 3), round(material_overlap, 3), round(param_cov, 3),
                    "|".join(sorted(secs.keys())), sec_chars, sec_toks, suspect, "filtered_out"
                ])
                continue

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
                    "text_flat": prot_text
                },
                "consistency": {
                    "doi_match": bool(doi_match),
                    "title_sim": round(title_sim, 3),
                    "keyword_sim": round(keyword_sim, 3),
                    "action_overlap": round(action_overlap, 3),
                    "material_overlap": round(material_overlap, 3),
                    "param_coverage": round(param_cov, 3),
                    "section_list": sorted(list(secs.keys())),
                    "sections_total_chars": sec_chars,
                    "sections_total_tokens": sec_toks,
                    "suspect_mismatch": bool(suspect),
                    "notes": notes
                }
            }
            json.dump(pair, open(outp, "a", encoding="utf-8"));
            open(outp, "a").write("\n")
            w.writerow([
                pid, a.get("pmcid", ""),
                (b.get("classification") or {}).get("primary_domain", "Unknown"),
                round(doi_match, 3), round(title_sim, 3), round(keyword_sim, 3),
                round(action_overlap, 3), round(material_overlap, 3), round(param_cov, 3),
                "|".join(sorted(secs.keys())), sec_chars, sec_toks, suspect, notes
            ])
            n_written += 1

    print(f"[OK] processed ids={len(ids)}, written_pairs={n_written}")
    print(f"[OK] wrote pairs:  {outp}")
    print(f"[OK] wrote report: {rep}")


if __name__ == "__main__":
    main()
