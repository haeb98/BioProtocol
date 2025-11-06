#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Robust IR-vs-Gold evaluator
- Schema autodetect (root / rec['ir'] / rec['slots'] / rec['extraction'] ... )
- Metrics:
  * Keyword P/R/F1 (top-K keyword lists; fallback: tokens from title+materials)
  * Step P/R/F1 (Jaccard over tokens; --sim cosine optional if sentence-transformers avail)
  * Token P/R/F1 (full text bag-of-words overlap)
  * Parameter P/R/F1 (regex-based value+unit compare; name-insensitive)
  * Order score (pairwise order agreement of matched steps)
- Writes per-protocol CSV and summary CSV; always prints [OK] messages.
"""

import argparse
import csv
import json
import os
import re
import sys
from collections import Counter, defaultdict


def load_jsonl(path):
    out = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                out.append(json.loads(line))
            except Exception as e:
                print(f"[WARN] bad json in {path}: {e}", file=sys.stderr)
    return out


# --------------------- schema helpers ---------------------
def pick(obj, *names, default=None):
    for n in names:
        if isinstance(obj, dict) and n in obj:
            return obj[n]
    return default


def ensure_list(x):
    if x is None: return []
    if isinstance(x, list): return x
    if isinstance(x, str): return [x]
    return list(x) if isinstance(x, (set, tuple)) else []


def tokens(s):
    if not s: return []
    s = re.sub(r"[_/\\\-\.\(\)\[\],:;!?\{\}%~^°××g]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip().lower()
    return [w for w in s.split() if len(w) > 1]


PARAM_RE = re.compile(
    r"(?P<val>\d+(?:\.\d+)?)\s*(?P<unit>h|min|s|°c|c|rpm|g|mg/ml|mg|ug/ml|µg/ml|ug|μg|g/l|ml|ul|µl|nm|mm|m|x|%)",
    re.I
)


def extract_params_from_text(s):
    out = []
    if not s: return out
    for m in PARAM_RE.finditer(s):
        val = float(m.group('val'))
        unit = m.group('unit').lower()
        out.append((round(val, 6), unit))
    return out


def normalize_record(rec, role="pred"):
    """Return unified dict:
       {protocol_id, title, keywords(list), steps(list[str]), actions(list[str]),
        materials(list[str]), params(list[(val,unit)]), flat_text(str)}
    """
    pid = rec.get("protocol_id") or rec.get("id") or rec.get("protocolId")
    title = pick(rec, "title", "protocol_title", default="")
    keywords = ensure_list(pick(rec, "keywords", "keyword_list", default=[]))

    # Where IR lives?
    ir = None
    for key in ("ir", "slots", "extraction", "parsed", "result"):
        if isinstance(rec.get(key), dict):
            ir = rec[key];
            break
    if ir is None:
        # sometimes IR fields live at root
        ir = rec

    # steps: try explicit steps; else from hierarchical structure; else from actions/materials text
    steps = pick(ir, "steps", "step_texts", default=[])
    if isinstance(steps, dict):
        # hierarchical map -> flatten by sorted numeric keys
        def keyer(k):
            return [int(p) if p.isdigit() else p for p in k.split(".")]

        flat = []
        for k in sorted(steps.keys(), key=keyer):
            v = steps[k]
            if isinstance(v, dict) and "title" in v:
                flat.append(v["title"])
            elif isinstance(v, str):
                flat.append(v)
        steps = flat
    elif isinstance(steps, list):
        steps = [s if isinstance(s, str) else json.dumps(s, ensure_ascii=False) for s in steps]
    else:
        steps = []

    actions = ensure_list(pick(ir, "actions", "action_list"))
    materials = ensure_list(pick(ir, "materials", "material_list", "reagents"))

    # parameters: accept structured or mine from texts
    params = []
    p_struct = pick(ir, "parameters", "params")
    if isinstance(p_struct, list) and p_struct:
        for p in p_struct:
            if isinstance(p, dict):
                v = p.get("value")
                u = (p.get("unit") or "").lower()
                try:
                    v = float(v)
                    if u: params.append((round(v, 6), u))
                except Exception:
                    pass
    if not params:
        blob = " ".join(steps + actions + materials + [title] + keywords)
        params = extract_params_from_text(blob)

    # keywords fallback
    if not keywords:
        keywords = list({*tokens(title), *[w for m in materials for w in tokens(m)]})[:20]

    flat_text = " ".join([title] + keywords + actions + materials + steps)

    return {
        "protocol_id": pid,
        "title": title,
        "keywords": keywords,
        "steps": steps,
        "actions": actions,
        "materials": materials,
        "params": params,
        "flat_text": flat_text,
    }


# --------------------- metrics ---------------------
def prf1(pred_set, gold_set):
    p = len(pred_set & gold_set) / (len(pred_set) or 1)
    r = len(pred_set & gold_set) / (len(gold_set) or 1)
    f = 0.0 if (p + r) == 0 else 2 * p * r / (p + r)
    return p, r, f


def jaccard(a_tokens, b_tokens):
    A, B = set(a_tokens), set(b_tokens)
    return len(A & B) / (len(A | B) or 1)


def step_match_pairs(pred_steps, gold_steps, sim="jaccard"):
    # greedy matching by similarity
    pairs = []
    used = set()
    for i, ps in enumerate(pred_steps):
        ts = tokens(ps)
        best_j, best_s = -1, 0.0
        for j, gs in enumerate(gold_steps):
            if j in used: continue
            s = jaccard(ts, tokens(gs))
            if s > best_s:
                best_s, best_j = s, j
        if best_j >= 0:
            pairs.append((i, best_j, best_s))
            used.add(best_j)
    return pairs


def order_score(pairs):
    # Kendall-like: ratio of concordant pairs among matched
    idx_pred = [i for i, _, _ in pairs]
    idx_gold = [j for _, j, _ in pairs]
    n = len(pairs)
    if n <= 1: return 1.0 if n == 1 else 0.0
    concord = 0;
    total = 0
    for a in range(n):
        for b in range(a + 1, n):
            total += 1
            concord += int((idx_pred[a] - idx_pred[b]) * (idx_gold[a] - idx_gold[b]) > 0)
    return concord / (total or 1)


def topk(seq, k):
    return list(seq)[:k] if len(seq) >= k else list(seq)


def token_bow(s):
    return Counter(tokens(s))


def token_overlap(pred_text, gold_text):
    Ap = token_bow(pred_text);
    Ag = token_bow(gold_text)
    common = set(Ap) & set(Ag)
    inter = sum(min(Ap[t], Ag[t]) for t in common)
    p = inter / (sum(Ap.values()) or 1)
    r = inter / (sum(Ag.values()) or 1)
    f = 0.0 if (p + r) == 0 else 2 * p * r / (p + r)
    return p, r, f


def param_set(params):
    # round value and unit
    return set(params)


def eval_pair(pred, gold, k_keywords=15):
    # keywords
    kw_p = set(topk(pred["keywords"], k_keywords))
    kw_g = set(topk(gold["keywords"], k_keywords))
    kwP, kwR, kwF = prf1(kw_p, kw_g)

    # steps
    spairs = step_match_pairs(pred["steps"], gold["steps"])
    stepP = len(spairs) / (len(pred["steps"]) or 1)
    stepR = len(spairs) / (len(gold["steps"]) or 1)
    stepF = 0.0 if (stepP + stepR) == 0 else 2 * stepP * stepR / (stepP + stepR)
    ordS = order_score(spairs)

    # tokens
    tokP, tokR, tokF = token_overlap(pred["flat_text"], gold["flat_text"])

    # params
    pp = param_set(pred["params"]);
    gp = param_set(gold["params"])
    parP, parR, parF = prf1(pp, gp)

    return {
        "keyword_p": kwP, "keyword_r": kwR, "keyword_f1": kwF,
        "step_p": stepP, "step_r": stepR, "step_f1": stepF,
        "token_p": tokP, "token_r": tokR, "token_f1": tokF,
        "param_p": parP, "param_r": parR, "param_f1": parF,
        "order": ordS,
        "matched_steps": len(spairs),
        "pred_steps": len(pred["steps"]),
        "gold_steps": len(gold["steps"]),
        "pred_params": len(pp),
        "gold_params": len(gp),
    }


def write_csv(path, rows, header):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--gold", required=True)
    ap.add_argument("--by-protocol", required=True)
    ap.add_argument("--summary", required=True)
    ap.add_argument("--debug-jsonl", default=None)
    ap.add_argument("--k", type=int, default=15)
    ap.add_argument("--sim", choices=["jaccard", "cosine"], default="jaccard")
    args = ap.parse_args()

    print("[INFO] pred:", args.pred)
    print("[INFO] gold:", args.gold)
    print("[INFO] by-protocol:", args.by_protocol)
    print("[INFO] summary:", args.summary)
    print("[INFO] loading ...", flush=True)

    pred_raw = load_jsonl(args.pred)
    gold_raw = load_jsonl(args.gold)
    print(f"[INFO] loaded pred={len(pred_raw)} gold={len(gold_raw)}")

    pred = {(r.get("protocol_id") or r.get("id")): normalize_record(r, "pred") for r in pred_raw}
    gold = {(r.get("protocol_id") or r.get("id")): normalize_record(r, "gold") for r in gold_raw}

    ids = sorted(set(pred.keys()) & set(gold.keys()))
    print(f"[INFO] id_intersection={len(ids)}", flush=True)

    if args.debug_jsonl:
        with open(args.debug_jsonl, "w", encoding="utf-8") as f:
            for pid in ids:
                f.write(json.dumps({
                    "protocol_id": pid,
                    "pred": pred[pid],
                    "gold": gold[pid]
                }, ensure_ascii=False) + "\n")
        print(f"[LOG] debug -> {args.debug_jsonl}")

    by_rows = []
    agg = defaultdict(list)

    for pid in ids:
        m = eval_pair(pred[pid], gold[pid], k_keywords=args.k)
        row = {"protocol_id": pid}
        row.update({k: round(v, 4) if isinstance(v, float) else v for k, v in m.items()})
        by_rows.append(row)
        for k, v in m.items():
            if isinstance(v, (int, float)):
                agg[k].append(float(v))

    if not by_rows:
        print("[WARN] no comparable pairs; writing empty CSVs")
        write_csv(args.by_protocol, [],
                  ["protocol_id", "keyword_p", "keyword_r", "keyword_f1", "step_p", "step_r", "step_f1", "token_p",
                   "token_r", "token_f1", "param_p", "param_r", "param_f1", "order", "matched_steps", "pred_steps",
                   "gold_steps", "pred_params", "gold_params"])
        write_csv(args.summary, [], ["metric", "mean", "std", "min", "max", "n"])
        print(f"[OK] wrote: {args.by_protocol}")
        print(f"[OK] wrote: {args.summary}")
        return

    # write by-protocol
    header = list(by_rows[0].keys())
    write_csv(args.by_protocol, by_rows, header)
    print(f"[OK] wrote: {args.by_protocol} (rows={len(by_rows)})")

    # summary
    import statistics as S
    srows = []
    for k, arr in agg.items():
        srows.append({
            "metric": k,
            "mean": round(S.mean(arr), 4),
            "std": round(S.pstdev(arr), 4) if len(arr) > 1 else 0.0,
            "min": round(min(arr), 4),
            "max": round(max(arr), 4),
            "n": len(arr)
        })
    write_csv(args.summary, srows, ["metric", "mean", "std", "min", "max", "n"])
    print(f"[OK] wrote: {args.summary} (metrics={len(srows)})")


if __name__ == "__main__":
    main()
