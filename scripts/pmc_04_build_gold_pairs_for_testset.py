#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
pmc_04_build_gold_pairs_for_testset.py  (v2 – protocol_sim, soft coverage, null keywords)

- Build gold pairs between Bio-protocol items (b) and grouped article sections (a)

Changes:
  * action_coverage -> protocol_sim (b.protocol vs a.sec_text similarity)
    - Keep action_coverage for backward-compat but set equal to protocol_sim.
    - protocol_sim = max(word-jaccard, char-3gram-jaccard)
  * keywords_sim: if b.keywords empty -> null; else fraction contained in a.sec_text
  * materials_coverage / param_coverage:
    - report both strict and soft; final metric = max(strict, soft)
    - soft(materials): token subset overlap >= 0.6 against text token set
    - soft(params): tolerant regex (spaces/hyphens/μ/°C variants)

Also outputs length/token comparisons between a.sec_text and b.hierarchical_protocol.
"""

import argparse
import csv
import json
import re
from pathlib import Path

# -------------------------
# Tokenization & similarity
# -------------------------
TOKEN_RX = re.compile(r"[A-Za-z0-9]+", re.I)


def to_tokens(s: str) -> list[str]:
    return TOKEN_RX.findall((s or "").lower())


def to_tset(s: str) -> set[str]:
    return set(to_tokens(s))


def jaccard_set(A: set[str], B: set[str]) -> float:
    if not A and not B: return 1.0
    if not A or not B: return 0.0
    return len(A & B) / max(1, len(A | B))


def char_ngrams(s: str, n: int = 3) -> set[str]:
    s = (s or "").lower()
    # collapse spaces/punct a bit
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"[^a-z0-9 ]+", "", s)
    if len(s) < n: return {s} if s else set()
    return {s[i:i + n] for i in range(len(s) - n + 1)}


def jaccard_text(a: str, b: str) -> float:
    # word-level
    wj = jaccard_set(to_tset(a), to_tset(b))
    # char 3-gram
    cj = jaccard_set(char_ngrams(a, 3), char_ngrams(b, 3))
    return max(wj, cj)


def safe_len(s: str) -> int:
    return len(s or "")


def safe_tokens(s: str) -> int:
    return len(to_tokens(s or ""))


# -------------------------
# Bio-protocol helpers
# -------------------------
def get_keywords_list(brec) -> list[str]:
    kw = brec.get("keywords")
    out = []
    if not kw:
        return out
    if isinstance(kw, list):
        for k in kw:
            if isinstance(k, str) and k.strip():
                out.append(k.strip())
            elif isinstance(k, dict):
                v = (k.get("keyword") or k.get("name") or "").strip()
                if v: out.append(v)
    elif isinstance(kw, str):
        parts = re.split(r"[;,]", kw)
        out = [p.strip() for p in parts if p.strip()]
    # dedup lower
    seen = set();
    res = []
    for k in out:
        lk = k.lower()
        if lk not in seen:
            seen.add(lk);
            res.append(k)
    return res


def get_material_candidates(brec) -> list[str]:
    src = brec.get("input") or brec.get("materials") or ""
    items = []
    if isinstance(src, list):
        for x in src:
            if isinstance(x, str) and x.strip():
                items.extend(re.split(r"[,\n;]", x))
    elif isinstance(src, str) and src.strip():
        items = re.split(r"[,\n;]", src)
    items = [re.sub(r"\s+", " ", it).strip() for it in items if it and it.strip()]
    # very light filter
    ban = {"-", "—", "and", "or"}
    items = [it for it in items if it.lower() not in ban]
    # dedup
    seen = set();
    res = []
    for it in items:
        lk = it.lower()
        if lk not in seen:
            seen.add(lk);
            res.append(it)
    return res[:100]


STEP_NUMBER_RX = re.compile(r"^\s*\d+\.\s*")
UNIT_RX = re.compile(
    r"\b(\d+(?:\.\d+)?)\s?(mL|ml|µl|μl|L|g|mg|µg|μg|kg|M|mM|nM|µM|μM|%|°C|degC|min|h|hr|hrs|hours|sec|s|rpm|g|xg)\b",
    re.I
)


def get_param_candidates(brec) -> list[str]:
    prot = brec.get("protocol") or ""
    if not isinstance(prot, str) or not prot.strip():
        return []
    out = []
    for line in prot.splitlines():
        # strip leading numbering like "1. "
        line_ = STEP_NUMBER_RX.sub("", line)
        for m in UNIT_RX.finditer(line_):
            out.append(m.group(0).strip())
    # dedup keep order
    seen = set();
    res = []
    for s in out:
        lk = s.lower()
        if lk not in seen:
            seen.add(lk);
            res.append(s)
    return res[:150]


# -------------------------
# Coverage (strict/soft)
# -------------------------
def contains_word_boundary(needle: str, hay: str) -> bool:
    toks = needle.split()
    hay_l = (hay or "").lower()
    ndl = needle.lower().strip()
    if not ndl: return False
    if len(toks) == 1:
        return re.search(rf"\b{re.escape(ndl)}\b", hay_l) is not None
    return ndl in hay_l


def fraction_contains_strict(cands: list[str], text: str) -> tuple[float, int, int]:
    if not cands: return (0.0, 0, 0)
    hits = 0
    for c in cands:
        if contains_word_boundary(c.lower(), text):
            hits += 1
    return (hits / len(cands), hits, len(cands))


def normalize_unit_token(s: str) -> str:
    s = (s or "").strip()
    # unify micro symbol, degree
    s = s.replace("μl", "µl").replace("μg", "µg")
    s = s.replace("degc", "°c").replace("° C", "°C").replace("℃", "°C")
    s = re.sub(r"\s+", " ", s)
    return s


def param_pattern(needle: str) -> re.Pattern:
    # make tolerant regex for number+unit like "10 mL", "10mL", "10-mL", "10ml"
    n = normalize_unit_token(needle.lower())
    n = n.replace(" ", r"\s*").replace("-", r"[-\s]*")
    n = n.replace("°c", r"(?:°\s?c|degc)")
    # word boundary-ish
    return re.compile(rf"(?<!\d){n}(?!\d)", re.I)


def fraction_params_strict(cands: list[str], text: str) -> tuple[float, int, int]:
    if not cands: return (0.0, 0, 0)
    hay = (text or "")
    hits = 0
    for c in cands:
        pat = param_pattern(c)
        if pat.search(hay):
            hits += 1
    return (hits / len(cands), hits, len(cands))


def fraction_contains_soft(cands: list[str], text: str, thresh: float = 0.6) -> tuple[float, int, int]:
    """
    Soft: candidate tokens must be present in text tokens with ratio >= thresh.
    (order-free, global presence)
    """
    if not cands: return (0.0, 0, 0)
    text_toks = to_tset(text)
    hits = 0
    for c in cands:
        toks = set(to_tokens(c))
        toks = {t for t in toks if t}  # drop empties
        if not toks:
            continue
        cov = len(toks & text_toks) / len(toks)
        if cov >= thresh:
            hits += 1
    return (hits / len(cands), hits, len(cands))


# -------------------------
# IO helpers
# -------------------------
def read_ids_csv(p: Path) -> set[str]:
    s = set()
    with p.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            pid = (r.get("protocol_id") or "").strip()
            if pid: s.add(pid)
    return s


def load_bio(bio_json: Path) -> dict:
    js = json.loads(bio_json.read_text(encoding="utf-8"))
    idx = {}
    for r in js:
        pid = str(r.get("protocol_id") or r.get("id") or "").strip()
        if not pid: continue
        idx[pid] = r
    return idx


def load_arts(grouped_jsonl: Path) -> dict:
    by_pid = {}
    with grouped_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except:
                continue
            pid = rec.get("protocol_id")
            if not pid: continue
            by_pid.setdefault(pid, []).append(rec)
    return by_pid


def choose_sections(a: dict, prefer_regular: bool = True) -> tuple[str, list[str]]:
    sections = a.get("sections") or {}
    stats = a.get("stats") or {}
    titles = sorted(sections.keys())
    used = [];
    parts = []
    for t in titles:
        src = (stats.get(t) or {}).get("source", "")
        if prefer_regular and src == "heuristic":
            continue
        parts.append(sections[t]);
        used.append(t)
    if not parts and titles:
        for t in titles:
            parts.append(sections[t]);
            used.append(t)
    return ("\n\n".join(parts).strip(), used)


# -------------------------
# Metrics per pair
# -------------------------
def compute_metrics(b: dict, a: dict, prefer_regular: bool):
    # texts
    sec_text, section_list = choose_sections(a, prefer_regular=prefer_regular)
    b_title = b.get("title") or ""
    a_title = a.get("title") or ""

    # 1) title_sim
    title_sim = jaccard_text(b_title, a_title)

    # 2) keywords_sim (null if keywords empty)
    b_keywords = get_keywords_list(b)
    if not b_keywords:
        keywords_sim = None
        kw_hits = 0;
        kw_total = 0
    else:
        # fraction exact-contained with word boundary
        hits = 0
        for kw in b_keywords:
            k = kw.strip().lower()
            if not k: continue
            if contains_word_boundary(k, sec_text):
                hits += 1
        kw_hits = hits;
        kw_total = len(b_keywords)
        keywords_sim = (hits / max(1, len(b_keywords)))

    # 3) protocol_sim (b.protocol vs sec_text)
    b_protocol = b.get("protocol") or ""
    protocol_sim = jaccard_text(b_protocol, sec_text)

    # 4) materials coverage (strict & soft)
    mat_items = get_material_candidates(b)
    m_strict, m_hits_s, m_tot = fraction_contains_strict(mat_items, sec_text)
    m_soft, m_hits_sf, _ = fraction_contains_soft(mat_items, sec_text, 0.6)
    material_coverage = max(m_strict, m_soft)

    # 5) params coverage (strict tolerant regex & soft token)
    param_items = get_param_candidates(b)
    p_strict, p_hits_s, p_tot = fraction_params_strict(param_items, sec_text)
    p_soft, p_hits_sf, _ = fraction_contains_soft(param_items, sec_text, 0.6)
    param_coverage = max(p_strict, p_soft)

    # 6) length/token comparison
    b_hp = b.get("hierarchical_protocol") or ""
    b_hp_str = b_hp if isinstance(b_hp, str) else json.dumps(b_hp, ensure_ascii=False)

    metrics = {
        "title_sim": round(title_sim, 4),

        "keywords_sim": (None if keywords_sim is None else round(keywords_sim, 4)),
        "keywords_hits": kw_hits,
        "keywords_total": kw_total,

        "material_coverage": round(material_coverage, 4),
        "material_cov_strict": round(m_strict, 4),
        "material_cov_soft": round(m_soft, 4),

        # action_coverage kept for backward-compat but equals protocol_sim
        "protocol_sim": round(protocol_sim, 4),
        "action_coverage": round(protocol_sim, 4),

        "param_coverage": round(param_coverage, 4),
        "param_cov_strict": round(p_strict, 4),
        "param_cov_soft": round(p_soft, 4),

        "sec_chars": safe_len(sec_text),
        "sec_tokens": safe_tokens(sec_text),
        "hier_chars": safe_len(b_hp_str),
        "hier_tokens": safe_tokens(b_hp_str),

        "section_list": section_list,
    }

    details = {
        "materials_items": mat_items,
        "params_items": param_items,
    }

    return metrics, details, sec_text


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bio", required=True, help="data/raw/bio_protocol.json")
    ap.add_argument("--ids", required=True, help="CSV with protocol_id column (test set)")
    ap.add_argument("--arts", required=True, help="grouped article sections jsonl")
    ap.add_argument("--out", required=True, help="output jsonl (gold_pairs_testset.jsonl)")
    ap.add_argument("--report", default="data/gold/testset_consistency_report.csv", help="CSV report")
    ap.add_argument("--prefer-regular", action="store_true", help="ignore sections whose source is 'heuristic'")
    args = ap.parse_args()

    ids = read_ids_csv(Path(args.ids))
    bio_idx = load_bio(Path(args.bio))
    arts_by_pid = load_arts(Path(args.arts))

    out_rows = []
    rep_rows = []

    for pid in ids:
        b = bio_idx.get(pid)
        if not b: continue
        cand_arts = arts_by_pid.get(pid, [])
        if not cand_arts: continue

        # pick article with best title similarity to BioProtocol title
        best = None;
        best_sim = -1
        for a in cand_arts:
            sim = jaccard_text(b.get("title") or "", a.get("title") or "")
            if sim > best_sim:
                best_sim = sim;
                best = a
        a = best

        metrics, details, sec_text = compute_metrics(b, a, prefer_regular=args.prefer_regular)

        row = {
            "protocol_id": pid,
            "pmcid": a.get("pmcid"),
            "domain": a.get("domain") or b.get("classification", {}).get("primary_domain"),
            "bio": {
                "title": b.get("title"),
                "keywords": get_keywords_list(b),
                "hierarchical_protocol": b.get("hierarchical_protocol"),
                "protocol": b.get("protocol")
            },
            "article": {
                "title": a.get("title"),
                "meta": a.get("meta", {}),
                "sections": a.get("sections", {}),
                "section_list": metrics.get("section_list")
            },
            "sec_text": sec_text,
            "metrics": metrics,
            "details": details
        }
        out_rows.append(row)

        rep_rows.append({
            "protocol_id": pid,
            "pmcid": a.get("pmcid"),
            "title_sim": metrics["title_sim"],
            "keywords_sim": "" if metrics["keywords_sim"] is None else metrics["keywords_sim"],
            "materials_cov": metrics["material_coverage"],
            "materials_cov_strict": metrics["material_cov_strict"],
            "materials_cov_soft": metrics["material_cov_soft"],
            "protocol_sim": metrics["protocol_sim"],
            "param_cov": metrics["param_coverage"],
            "param_cov_strict": metrics["param_cov_strict"],
            "param_cov_soft": metrics["param_cov_soft"],
            "sec_tokens": metrics["sec_tokens"],
            "hier_tokens": metrics["hier_tokens"],
        })

    # write outputs
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as w:
        for r in out_rows:
            w.write(json.dumps(r, ensure_ascii=False) + "\n")

    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    with open(args.report, "w", newline="", encoding="utf-8") as f:
        fns = ["protocol_id", "pmcid", "title_sim", "keywords_sim",
               "materials_cov", "materials_cov_strict", "materials_cov_soft",
               "protocol_sim", "param_cov", "param_cov_strict", "param_cov_soft",
               "sec_tokens", "hier_tokens"]
        cw = csv.DictWriter(f, fieldnames=fns)
        cw.writeheader()
        for r in rep_rows:
            cw.writerow(r)

    print(f"[OK] wrote pairs: {args.out} (rows={len(out_rows)})")
    print(f"[OK] report: {args.report}")


if __name__ == "__main__":
    main()
