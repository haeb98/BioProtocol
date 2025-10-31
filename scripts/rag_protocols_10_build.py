#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rag_protocols_10_build.py (v3)
- bio_protocol.json / protocol_exchange.json / protocols_io.json → RAG 코퍼스(JSONL)
- 테스트셋(exclude) 제외
- 의미 단위 청킹: --chunk-by section | step | none
- 짧은 청크 방지: --min-chars, --pack-steps
- IR 힌트(액션/재료/파라미터) 간단 추출
- 상세 로그 & 통계 출력
- 기본 출력: data/rag/corpus/protocols_wo_test25.jsonl
"""

import argparse
import csv
import json
import pathlib
import re
import sys
from typing import Dict, List, Tuple, Any

ACTIONS = {"incubate", "centrifuge", "mix", "add", "transfer", "pipette", "aliquot", "vortex",
           "resuspend", "pellet", "wash", "dry", "filter", "measure", "dilute", "prepare",
           "heat", "cool", "shake", "spin", "stain", "fix", "mount", "label", "load", "elute"}

MATERIAL_HINTS = {"buffer", "pbs", "ethanol", "methanol", "bleach", "nacl", "glycerol", "yeast",
                  "agar", "water", "h2o", "dmso", "triton", "edta", "mgso4", "kcl", "k2hpo4",
                  "kh2po4", "nh4", "medium", "media", "reagent", "solution"}

RE_PARAM = re.compile(r"\b(\d+(?:\.\d+)?)\s*(°c|degc|s|min|h|hr|hrs|ms|rpm|g|xg|µl|ul|ml|l|%)\b", re.I)
TOKSPLIT = re.compile(r"\W+")


def tokset(s: str) -> List[str]:
    return [t for t in TOKSPLIT.split((s or "").lower()) if t]


def extract_actions(text: str) -> List[str]:
    toks = set(tokset(text));
    return sorted(list(ACTIONS & toks))


def extract_materials(text: str) -> List[str]:
    toks = set(tokset(text));
    mats = set()
    for h in MATERIAL_HINTS:
        if h in toks: mats.add(h)
    for tt in list(toks):
        if re.match(r"^(nacl|kcl|pbs|h2o|ethanol|methanol|glycerol|triton|edta|dmso)$", tt):
            mats.add(tt)
    return sorted(list(mats))


def extract_params(text: str) -> List[Dict[str, Any]]:
    out = []
    for m in RE_PARAM.finditer(text or ""):
        try:
            val = float(m.group(1))
        except:
            continue
        out.append({"value": val, "unit": m.group(2)})
    return out


def load_json_array_or_jsonl(path: str) -> List[dict]:
    p = pathlib.Path(path)
    if not p.exists(): return []
    raw = p.read_text(encoding="utf-8", errors="ignore").strip()
    if not raw: return []
    # 1) JSON array 시도
    try:
        data = json.loads(raw)
        if isinstance(data, list): return data
    except Exception:
        pass
    # 2) JSONL 시도
    arr = []
    for line in raw.splitlines():
        line = line.strip()
        if not line: continue
        try:
            arr.append(json.loads(line))
        except Exception:
            continue
    return arr


def load_exclude_ids(csv_path: str) -> set:
    s = set()
    p = pathlib.Path(csv_path)
    if not p.exists(): return s
    with open(p, "r", encoding="utf-8") as fin:
        rdr = csv.DictReader(fin)
        for row in rdr:
            pid = (row.get("protocol_id") or row.get("id") or "").strip()
            if pid: s.add(pid)
    return s


def is_leaf(v: Any) -> bool:
    return isinstance(v, str)


def flatten_hier_to_lines(hier: dict) -> str:
    if not isinstance(hier, dict): return ""

    def keyer(k: str):
        return [int(p) if p.isdigit() else p for p in k.split(".")]

    lines = []
    for k in sorted(hier.keys(), key=keyer):
        v = hier[k]
        if isinstance(v, dict) and "title" in v:
            lines.append(v["title"])
        elif isinstance(v, str):
            lines.append(v)
    return "\n".join(lines)


def split_by_top_sections(hier: dict) -> List[Tuple[str, Dict[str, Any]]]:
    if not isinstance(hier, dict): return []

    def keyer(k: str):
        return [int(p) if p.isdigit() else p for p in k.split(".")]

    top_keys = sorted({k.split(".")[0] for k in hier.keys()},
                      key=lambda x: [int(x)] if x.isdigit() else [x])
    out = []
    for top in top_keys:
        title = None;
        steps = []
        if isinstance(hier.get(top), dict) and "title" in hier[top]:
            title = hier[top]["title"]
        for k in sorted(hier.keys(), key=keyer):
            if k == top or k.startswith(top + "."):
                v = hier[k]
                if is_leaf(v): steps.append(v)
        out.append((top, {"title": title, "steps": steps}))
    return out


def split_by_steps(hier: dict, pack_steps: int = 6, min_chars: int = 300) -> List[Tuple[str, List[str]]]:
    if not isinstance(hier, dict): return []

    def keyer(k: str):
        return [int(p) if p.isdigit() else p for p in k.split(".")]

    ordered = []
    for k in sorted(hier.keys(), key=keyer):
        v = hier[k]
        if is_leaf(v): ordered.append((k, v))
    chunks = [];
    i = 0;
    idx = 1
    while i < len(ordered):
        bag = [ordered[i][1]];
        chars = len(bag[0]);
        j = i + 1
        while j < len(ordered) and len(bag) < pack_steps:
            bag.append(ordered[j][1]);
            chars += len(ordered[j][1]);
            j += 1
        while chars < min_chars and j < len(ordered) and len(bag) < pack_steps:
            bag.append(ordered[j][1]);
            chars += len(ordered[j][1]);
            j += 1
        chunks.append((f"chunk{idx}", bag));
        idx += 1;
        i = j
    return chunks


def build_record(source_name: str, orig_id: str, title: str, section_title: str, domain: str,
                 text: str, url: str, keywords: str, emit_ir_hints: bool, section_id: str = "") -> dict:
    rec_id = f"protocols::{source_name}::{orig_id}"
    if section_id: rec_id += f"::sec:{section_id}"
    rec = {
        "id": rec_id,
        "doc_id": f"{source_name}::{orig_id}",
        "section_id": section_id or "document",
        "source": "protocols",
        "title": title or "",
        "section_title": section_title or "protocol",
        "domain": domain or "Unknown",
        "task_tags": ["protocol"],
        "text": (text or "").strip(),
        "meta": {"source_name": source_name, "orig_id": orig_id, "url": url or "", "keywords": keywords or ""},
        "stats": {"n_chars": len(text or ""), "n_tokens": len((text or "").split())},
        "ir_hint": {"actions": [], "materials": [], "params": []}
    }
    if emit_ir_hints:
        rec["ir_hint"] = {
            "actions": extract_actions(text or ""),
            "materials": extract_materials(text or ""),
            "params": extract_params(text or "")
        }
    return rec


def make_text_block(title: str, inputs: str, proto: str, extra: str = "") -> str:
    parts = [p for p in [title, inputs, proto, extra] if p and str(p).strip()]
    return "\n\n".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--biop", required=True)
    ap.add_argument("--protex", required=False)
    ap.add_argument("--pio", required=False)
    ap.add_argument("--exclude-ids", required=True)
    ap.add_argument("--chunk-by", choices=["section", "step", "none"], default="section")
    ap.add_argument("--min-chars", type=int, default=300)
    ap.add_argument("--pack-steps", type=int, default=6)
    ap.add_argument("--emit-ir-hints", action="store_true", default=True)
    ap.add_argument("--out", default="data/rag/corpus/protocols_wo_test25.jsonl")  # << 기본을 data/rag로
    ap.add_argument("--verbose", action="store_true", default=True)
    args = ap.parse_args()

    outp = pathlib.Path(args.out);
    outp.parent.mkdir(parents=True, exist_ok=True)

    # 입력 로드
    biop = load_json_array_or_jsonl(args.biop)
    protex = load_json_array_or_jsonl(args.protex) if args.protex else []
    pio = load_json_array_or_jsonl(args.pio) if args.pio else []

    excl = load_exclude_ids(args.exclude_ids)

    def filter_excl(arr, name):
        kept = [r for r in arr if str(r.get("id") or "").strip() not in excl]
        if args.verbose:
            print(f"[SRC] {name:17s} total={len(arr):5d}  excluded={len(arr) - len(kept):5d}  kept={len(kept):5d}")
        return kept

    biop = filter_excl(biop, "bio_protocol")
    protex = filter_excl(protex, "protocol_exchange")
    pio = filter_excl(pio, "protocols_io")

    sources = [("bio_protocol", biop), ("protocol_exchange", protex), ("protocols_io", pio)]

    n_docs = 0;
    n_chunks = 0
    with open(outp, "w", encoding="utf-8") as fout:
        for sname, arr in sources:
            for r in arr:
                n_docs += 1
                pid = str(r.get("id") or "").strip()
                title = r.get("title") or ""
                inputs = r.get("input") or ""
                proto = r.get("protocol") or ""
                hier = r.get("hierarchical_protocol") or {}
                domain = (r.get("classification") or {}).get("primary_domain", "Unknown")
                url = r.get("url") or ""
                keywords = r.get("keywords") or ""

                whole_text = make_text_block(title, inputs, proto, extra=flatten_hier_to_lines(hier))

                if args.chunk_by == "none" or not isinstance(hier, dict) or not hier:
                    rec = build_record(sname, pid, title, "protocol", domain, whole_text, url, keywords,
                                       args.emit_ir_hints, "")
                    fout.write(json.dumps(rec, ensure_ascii=False) + "\n");
                    n_chunks += 1;
                    continue

                if args.chunk_by == "section":
                    sections = split_by_top_sections(hier)
                    if not sections:
                        rec = build_record(sname, pid, title, "protocol", domain, whole_text, url, keywords,
                                           args.emit_ir_hints, "")
                        fout.write(json.dumps(rec, ensure_ascii=False) + "\n");
                        n_chunks += 1;
                        continue
                    for sec_key, sec_data in sections:
                        sec_title = sec_data.get("title") or f"Section {sec_key}"
                        steps_text = "\n".join(sec_data.get("steps") or [])
                        text = make_text_block(title, inputs, proto="", extra=steps_text)
                        if len(text) < args.min_chars:
                            text = whole_text
                            sec_title = f"{sec_title} (fallback: full)"
                        rec = build_record(sname, pid, title, sec_title, domain, text, url, keywords,
                                           args.emit_ir_hints, sec_key)
                        fout.write(json.dumps(rec, ensure_ascii=False) + "\n");
                        n_chunks += 1

                elif args.chunk_by == "step":
                    chunks = split_by_steps(hier, pack_steps=args.pack_steps, min_chars=args.min_chars)
                    if not chunks:
                        rec = build_record(sname, pid, title, "protocol", domain, whole_text, url, keywords,
                                           args.emit_ir_hints, "")
                        fout.write(json.dumps(rec, ensure_ascii=False) + "\n");
                        n_chunks += 1;
                        continue
                    for chunk_id, steps in chunks:
                        steps_text = "\n".join(steps)
                        text = steps_text if len(steps_text) >= args.min_chars else make_text_block(title, inputs,
                                                                                                    proto="",
                                                                                                    extra=steps_text)
                        rec = build_record(sname, pid, title, f"steps_{chunk_id}", domain, text, url, keywords,
                                           args.emit_ir_hints, chunk_id)
                        fout.write(json.dumps(rec, ensure_ascii=False) + "\n");
                        n_chunks += 1

    print(f"[OK] corpus -> {outp}")
    print(f"[STATS] docs_in={n_docs}, chunks_out={n_chunks}, chunk_by={args.chunk_by}")
    if n_chunks == 0:
        print("[WARN] 0 chunks written. Check --exclude-ids, input paths, or try '--chunk-by none' for sanity.",
              file=sys.stderr)


if __name__ == "__main__":
    main()
