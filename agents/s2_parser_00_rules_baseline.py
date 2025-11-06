#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
agents/s2_parser_00_rules_baseline.py  (STRICT matching version)

- Exact match on protocol_id. No normalization.
- ID sources: --ids-from-gold (JSONL) and/or --filter-ids (CSV). Union of both if both present.
- Arts JSONL must contain 'protocol_id' (exact string match).

Usage:
python agents/s2_parser_00_rules_baseline.py \
  --arts data/gold/gold_articles_sections_pmc.jsonl \
  --ids-from-gold data/gold/gold_pairs_testset_top15.jsonl \
  --filter-ids data/splits/test_biop_ids_top15.csv \
  --out runs/s2_rules_top15.ir.jsonl \
  --trace reports/s2_parser_trace_ids_top15.csv
"""
import argparse
import csv
import json
import re
from pathlib import Path
from typing import Dict, Any, List, Set

# --- very light IE rules (same as before) ---
ACTION_VERBS = ["add", "mix", "vortex", "incubate", "centrifuge", "spin", "resuspend",
                "transfer", "pipette", "wash", "dry", "measure", "heat", "cool", "stir",
                "filter", "dilute", "aliquot", "store", "prepare", "sonicate"]
ACTION_RX = re.compile(r"\b(" + "|".join(ACTION_VERBS) + r")\b", re.I)
NUM_UNIT_RXS = [re.compile(p, re.I) for p in [
    r"(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>°C|degC|C|° F|K)",
    r"(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>min|mins|minute|minutes|h|hr|hours|sec|s)",
    r"(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>µ?L|uL|mL|L|µl|ul)",
    r"(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>rpm|×g|xg|g-force)",
    r"(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>M|mM|µM|uM|%|w/v|v/v)",
]]
MAT_HINTS = re.compile(
    r"\b(buffer|solution|ethanol|methanol|sodium|chloride|NaCl|NaOH|HCl|Tris|EDTA|PBS|bleach|hypochlorite|yeast extract|glycerol|agar|AlCl3|PVP|catechin|gallic acid)\b",
    re.I)


def sentence_split(t: str) -> List[str]:
    return [p for p in re.split(r"(?<=[\.\!\?])\s+", (t or "").strip()) if p]


def extract_actions(s: str) -> List[str]:
    return [m.group(1).lower() for m in ACTION_RX.finditer(s or "")]


def extract_params(s: str) -> List[Dict[str, Any]]:
    out = []
    for rx in NUM_UNIT_RXS:
        for m in rx.finditer(s or ""):
            out.append({"value": m.group("value"), "unit": m.group("unit"), "span": m.group(0)})
    return out


def extract_materials(s: str) -> List[str]:
    return list({m.group(0) for m in MAT_HINTS.finditer(s or "")})


# --- ID loaders (STRICT) ---
def load_ids_from_csv(p: Path) -> Set[str]:
    ids = set()
    if not p: return ids
    with p.open("r", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    if not rows: return ids
    header = [h.strip().lower() for h in rows[0]]
    if "protocol_id" in header:
        idx = header.index("protocol_id")
        for r in rows[1:]:
            if len(r) > idx and r[idx].strip():
                ids.add(r[idx].strip())
    else:
        cands = [r[0].strip() for r in rows if r]
        if cands and cands[0].lower() in ("id", "protocol_id"):
            cands = cands[1:]
        ids.update([x for x in cands if x])
    return ids


def load_ids_from_gold_pairs(p: Path) -> Set[str]:
    ids = set()
    if not p: return ids
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            r = json.loads(line)
            pid = r.get("protocol_id", "").strip()
            if pid: ids.add(pid)
    return ids


def build_ir(proto_id: str, pmcid: str, domain: str, sections: Dict[str, str]) -> Dict[str, Any]:
    steps = [];
    mats = set();
    pars = [];
    order = [];
    sid = 0
    for sec_title, sec_text in (sections or {}).items():
        for sent in sentence_split(sec_text):
            acts = extract_actions(sent);
            ms = extract_materials(sent);
            ps = extract_params(sent)
            if not acts and not ms and not ps:
                continue
            sid += 1
            steps.append(
                {"step_id": sid, "section": sec_title, "actions": acts, "materials": ms, "parameters": ps, "raw": sent})
            mats.update([m.lower() for m in ms]);
            pars.extend(ps);
            order.append(sid)
    return {"protocol_id": proto_id, "pmcid": pmcid, "domain": domain, "steps": steps, "materials": sorted(mats),
            "parameters": pars, "order": order}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arts", required=True)
    ap.add_argument("--ids-from-gold", help="data/gold/gold_pairs_testset_top15.jsonl")
    ap.add_argument("--filter-ids", help="data/splits/test_biop_ids_top15.csv")
    ap.add_argument("--out", required=True)
    ap.add_argument("--trace", help="reports/s2_parser_trace_ids_top15.csv")
    args = ap.parse_args()

    ids = set()
    if args.ids_from_gold:
        ids |= load_ids_from_gold_pairs(Path(args.ids_from_gold))
    if args.filter_ids:
        ids |= load_ids_from_csv(Path(args.filter_ids))

    if not ids:
        raise SystemExit("No IDs provided. Use --ids-from-gold and/or --filter-ids")

    arts_p = Path(args.arts)
    out_p = Path(args.out);
    out_p.parent.mkdir(parents=True, exist_ok=True)
    trace_p = Path(args.trace) if args.trace else None
    if trace_p: trace_p.parent.mkdir(parents=True, exist_ok=True)

    total = 0;
    picked = 0;
    seen = set()
    trace_rows = []
    with arts_p.open("r", encoding="utf-8") as fin, out_p.open("w", encoding="utf-8") as fout:
        for ln in fin:
            total += 1
            if not ln.strip(): continue
            r = json.loads(ln)
            pid = (r.get("protocol_id") or "").strip()
            seen.add(pid)
            status = "skip_not_in_filter"
            if pid and pid in ids:
                pmcid = r.get("pmcid", "")
                domain = r.get("domain") or (r.get("classification", {}) or {}).get("primary_domain", "Unknown")
                sections = r.get("sections") or {}
                if isinstance(sections, list):
                    sections = {x.get("title", "Section"): x.get("text", "") for x in sections if isinstance(x, dict)}
                ir = build_ir(pid, pmcid, domain, sections)
                fout.write(json.dumps(ir, ensure_ascii=False) + "\n");
                picked += 1
                status = "picked"
            if trace_p:
                trace_rows.append([pid, status])

    print(
        f"[OK] wrote IR -> {out_p} (arts_lines={total}, ids_in_filter={len(ids)}, arts_ids={len(seen)}, picked={picked})")
    if trace_p:
        with trace_p.open("w", encoding="utf-8", newline="") as g:
            w = csv.writer(g);
            w.writerow(["protocol_id", "status"])
            for row in trace_rows: w.writerow(row)
        print(f"[TRACE] -> {trace_p}")


if __name__ == "__main__":
    main()
