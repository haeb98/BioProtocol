#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import csv
import json
import pathlib
import re
from collections import defaultdict

import orjson

ACTIONS = {"incubate", "centrifuge", "mix", "add", "transfer", "pipette", "aliquot", "vortex",
           "resuspend", "pellet", "wash", "dry", "filter", "measure", "dilute", "prepare",
           "heat", "cool", "shake", "spin", "stain", "fix", "mount", "label", "load", "elute"}
MATERIAL_HINTS = {"buffer", "pbs", "ethanol", "methanol", "bleach", "nacl", "glycerol", "yeast",
                  "agar", "water", "h2o", "dmso", "triton", "edta", "mgso4", "kcl", "k2hpo4",
                  "kh2po4", "nh4", "medium", "media", "reagent", "solution"}
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


def load_ids(p):
    rows = []
    with open(p, "r", encoding="utf-8") as f:
        hdr = f.readline().strip().split(",")
        idx = {h: i for i, h in enumerate(hdr)}
        for line in f:
            cells = line.rstrip("\n").split(",")
            pid = cells[idx.get("protocol_id", 0)].strip() if cells else ""
            if pid: rows.append(pid)
    return set(rows)


def load_bio(p):
    data = orjson.loads(pathlib.Path(p).read_bytes())
    return {str(r.get("id") or "").strip(): r for r in data if r.get("id")}


def load_arts(p):
    arr = [];
    with open(p, "r", encoding="utf-8") as f:
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
    ap.add_argument("--out", default="data/gold/testset_consistency_report.csv")
    ap.add_argument("--min-section", type=int, default=150)
    args = ap.parse_args()

    bio = load_bio(args.bio)
    ids = load_ids(args.ids)
    arts = load_arts(args.arts)
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "protocol_id", "pmcid", "domain",
            "doi_match", "title_sim", "keyword_sim",
            "action_overlap", "material_overlap", "param_coverage",
            "section_list", "sections_total_chars", "sections_total_tokens",
            "suspect_mismatch", "notes"
        ])
        for pid in sorted(ids):
            b = bio.get(pid)
            a = arts.get(pid)
            if not b or not a:
                w.writerow([pid, "", (b or {}).get("classification", {}).get("primary_domain", "Unknown"),
                            0, 0, 0, 0, 0, 0, "", 0, 0, 1, "no_bio" if not b else "no_article_sections"])
                continue
            secs = {k: v for k, v in (a.get("sections") or {}).items() if len(v) >= args.min_section}
            sec_text = "\n\n".join([secs[k] for k in sorted(secs.keys())])
            sec_chars = len(sec_text);
            sec_toks = len((sec_text or "").split())
            prot_text = "\n".join([
                (b.get("title") or ""),
                (b.get("input") or ""),
                (b.get("protocol") or ""),
                "\n".join([vv["title"] if isinstance(vv, dict) and "title" in vv else vv
                           for kk, vv in (b.get("hierarchical_protocol") or {}).items()])
            ])
            kw_b = (b.get("keywords") or "").lower()
            title_b = (b.get("title") or "").lower()
            title_sim = jacc(tokset(title_b), tokset((a.get("title") or "").lower()))
            keyword_sim = jacc(tokset(kw_b), tokset(sec_text.lower()))
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
            w.writerow([
                pid, a.get("pmcid", ""),
                (b.get("classification") or {}).get("primary_domain", "Unknown"),
                doi_match, round(title_sim, 3), round(keyword_sim, 3),
                round(action_overlap, 3), round(material_overlap, 3), round(param_cov, 3),
                "|".join(sorted(secs.keys())), sec_chars, sec_toks, suspect, notes
            ])


if __name__ == "__main__":
    main()
