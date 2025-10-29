# eval/eval_s1_metrics.py
"""
Evaluate S1 generations against gold using Bioprobench-like metrics on hierarchical protocols.

Input :
  - eval/s1_generations.jsonl
    fields: protocol_id, gen_hier(dict|None), gold_hier(dict|None), gold_text(str)
Output:
  - eval/s1_metrics_report.json  (summary averages)
  - eval/s1_metrics_samples.tsv  (per-sample details)

Metrics:
  - Keyword Precision/Recall/F1 (from gold keywords if available, else TF-IDF fallback)
  - Step Precision/Recall/F1    (semantic step matching with sentence embeddings)
  - Token Precision/Recall/F1   (unigram overlap)
  - Param Precision/Recall/F1   (number+unit with tolerance)
  - Order                       (in-order pair ratio among matched steps)

Env:
  EMB_MODEL   (default: sentence-transformers/all-MiniLM-L6-v2)
  STEP_SIM    (default: 0.70)
  PARAM_TOL   (default: 0.10)   # ±10% tolerance
  TOP_KEYWORDS(default: 10)
  MIN_STEPLEN (default: 20)     # drop too-short steps
"""

import json
import os
import pathlib
import re
import statistics
from collections import Counter

import numpy as np
from sentence_transformers import SentenceTransformer, util as st_util
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

ROOT = pathlib.Path(".")
TAG = os.environ.get("OUT_TAG", "s1")
IN = ROOT / f"eval/{TAG}_generations.jsonl"
OUTJ = ROOT / f"eval/{TAG}_metrics_report.json"
OUTT = ROOT / f"eval/{TAG}_metrics_samples.tsv"

EMB_MODEL = os.environ.get("EMB_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
STEP_SIM = float(os.environ.get("STEP_SIM", "0.70"))
PARAM_TOL = float(os.environ.get("PARAM_TOL", "0.10"))
TOP_KEYW = int(os.environ.get("TOP_KEYWORDS", "10"))
MIN_STEPLEN = int(os.environ.get("MIN_STEPLEN", "20"))

WS = re.compile(r"\s+")
TOK = re.compile(r"[a-z0-9]+")
# number with optional scientific x10^exp; capture unit
NUM = re.compile(r"(?P<val>\d+(?:\.\d+)?)(?:\s*(?:×|x)\s*10\^?(?P<exp>[-+]?\d+))?")
UNIT = r"(?:°c|degc|°f|min|h|hr|hours?|s|sec|m?s|µl|ul|ml|l|mg|g|kg|ng|pg|mm|cm|µm|um|nm|mM|µM|uM|nM|%|rpm|g\/l|mg\/ml|m\/v|v\/v)"
PARAM_RE = re.compile(rf"{NUM.pattern}\s*({UNIT})", re.IGNORECASE)


def norm_text(s: str) -> str:
    return WS.sub(" ", (s or "")).strip()


def flatten_hier(h):
    """
    hierarchical_protocol dict -> list of step-like strings
    - Orders keys by dotted numeric order.
    - Emits "## title" for section titles, raw strings for leaf steps.
    """
    if not isinstance(h, dict): return []

    def keyer(k):
        return [int(p) if str(p).isdigit() else p for p in str(k).split(".")]

    out = []
    for k in sorted(h.keys(), key=keyer):
        v = h[k]
        if isinstance(v, dict) and "title" in v:
            t = f"## {v['title']}".strip()
            if len(t) >= MIN_STEPLEN:
                out.append(t)
            else:
                out.append(t)  # keep even if short title (optional)
        elif isinstance(v, str):
            s = v.strip()
            if s:
                out.append(s)
    # drop too-short step lines (except titles that start with '## ')
    out2 = []
    for s in out:
        if s.startswith("## "):
            out2.append(s)
        elif len(s) >= MIN_STEPLEN:
            out2.append(s)
    return out2


def sent_steps_free(s: str):
    """
    Fallback step splitter for plain text (when hierarchical missing).
    Conservative: keep sentences/lines longer than MIN_STEPLEN.
    """
    s = (s or "").replace("\r", "\n")
    lines = [ln.strip() for ln in s.split("\n") if ln.strip()]
    tmp = []
    for ln in lines:
        chunks = re.split(r"(?<=[\.\:;])\s+(?=[A-Z0-9\(])", ln)
        tmp.extend(chunks)
    steps = [norm_text(x) for x in tmp if len(norm_text(x)) >= MIN_STEPLEN]
    return steps


def to_tokens(s: str):
    return TOK.findall((s or "").lower())


def keyword_set_from_gold(gold_pair_like, fallback_text: str):
    # gold keywords may live in original gold pairs file; in s1_generations we typically don't store them.
    # Use TF-IDF fallback on gold text.
    vect = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1)
    try:
        X = vect.fit_transform([fallback_text])
        idx = np.argsort(-X.toarray()[0])[:TOP_KEYW]
        feats = np.array(vect.get_feature_names_out())[idx]
        return set([f.lower() for f in feats])
    except Exception:
        toks = to_tokens(fallback_text)
        cnt = Counter(toks)
        return set([w for w, _ in cnt.most_common(TOP_KEYW)])


def norm_unit(u: str) -> str:
    u = (u or "").lower().replace(" ", "")
    repl = {
        "ul": "µl", "μl": "µl", "µl": "µl",
        "ml": "ml", "l": "l",
        "degc": "°c", "°c": "°c", "°f": "°f",
        "hr": "h", "hour": "h", "hours": "h",
        "sec": "s", "ms": "ms", "s": "s", "min": "min", "h": "h",
        "um": "µm", "µm": "µm",
        "mm": "mm", "cm": "cm", "nm": "nm",
        "mmol": "mM", "millimolar": "mM",
        "mg/ml": "mg/ml", "g/l": "g/l",
        "%": "%", "rpm": "rpm",
    }
    return repl.get(u, u)


def parse_params(s: str):
    out = []
    for m in PARAM_RE.finditer(s or ""):
        val = float(m.group("val"))
        if m.group("exp"):
            val = val * (10 ** int(m.group("exp")))
        unit = norm_unit(m.group(2))
        out.append((unit, val))
    return out


def param_f1(gen_params, gold_params, tol=PARAM_TOL):
    used = set()
    tp = 0
    for (u1, v1) in gen_params:
        best_j = None
        for j, (u2, v2) in enumerate(gold_params):
            if j in used:
                continue
            if u1 == u2 and abs(v1 - v2) <= tol * max(1.0, abs(v2)):
                best_j = j
                break
        if best_j is not None:
            tp += 1
            used.add(best_j)
    fp = max(0, len(gen_params) - tp)
    fn = max(0, len(gold_params) - tp)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
    return prec, rec, f1


def token_f1(gen_text, gold_text):
    g = set(to_tokens(gen_text));
    h = set(to_tokens(gold_text))
    tp = len(g & h);
    fp = len(g - h);
    fn = len(h - g)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
    return prec, rec, f1


def keyword_metrics(gen_text, gold_text):
    gold_kw = keyword_set_from_gold(None, gold_text)
    gen_kw = set(to_tokens(gen_text))
    hit = gold_kw & gen_kw
    prec = len(hit) / len(gen_kw) if gen_kw else 0.0
    rec = len(hit) / len(gold_kw) if gold_kw else 0.0
    f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
    return prec, rec, f1, sorted(hit)


def match_steps(gen_steps, gold_steps, model):
    if not gen_steps or not gold_steps:
        return [], 0.0, 0.0, 0.0
    gen_emb = model.encode(gen_steps, convert_to_tensor=True, normalize_embeddings=True, show_progress_bar=False)
    gold_emb = model.encode(gold_steps, convert_to_tensor=True, normalize_embeddings=True, show_progress_bar=False)
    sim = st_util.cos_sim(gen_emb, gold_emb).cpu().numpy()  # [G x H]
    matches = [];
    used_g = set();
    used_h = set()
    while True:
        i, j = np.unravel_index(np.argmax(sim, axis=None), sim.shape)
        if sim[i, j] < STEP_SIM:
            break
        if i in used_g or j in used_h:
            sim[i, j] = -1.0;
            continue
        matches.append((i, j, float(sim[i, j])))
        used_g.add(i);
        used_h.add(j)
        sim[i, :] = -1.0;
        sim[:, j] = -1.0
    tp = len(matches)
    prec = tp / len(gen_steps) if gen_steps else 0.0
    rec = tp / len(gold_steps) if gold_steps else 0.0
    f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
    return matches, prec, rec, f1


def order_score(matches):
    # matches: (i_gen, j_gold, sim)
    if len(matches) < 2: return 1.0
    m_sorted = sorted(matches, key=lambda x: x[1])  # by gold order
    gen_seq = [m[0] for m in m_sorted]
    in_order = 0;
    total = 0
    for a in range(len(gen_seq)):
        for b in range(a + 1, len(gen_seq)):
            total += 1
            if gen_seq[a] < gen_seq[b]:
                in_order += 1
    return in_order / total if total > 0 else 1.0


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def main():
    assert IN.exists(), f"not found: {IN}"
    model = SentenceTransformer(EMB_MODEL)

    rows = []
    for rec in tqdm(read_jsonl(IN), desc="eval-s1-hier"):
        pid = str(rec.get("protocol_id"))
        # --- gold steps ---
        gold_hier = rec.get("gold_hier")
        gold_text = norm_text(rec.get("gold_text", ""))
        gold_steps = flatten_hier(gold_hier) if gold_hier else sent_steps_free(gold_text)

        # --- gen steps ---
        gen_hier = rec.get("gen_hier")
        gen_text = norm_text(rec.get("gen", ""))  # only used when gen_hier is missing
        gen_steps = flatten_hier(gen_hier) if gen_hier else sent_steps_free(gen_text)

        # --- metrics ---
        # step-level semantic
        matches, sp, sr, s_f1 = match_steps(gen_steps, gold_steps, model)
        # token-level (concatenate steps)
        gen_join = " ".join(gen_steps)
        gold_join = " ".join(gold_steps)
        t_p, t_r, t_f1 = token_f1(gen_join, gold_join)
        # parameter (number+unit)
        gp = [p for s in gen_steps for p in parse_params(s)]
        hp = [p for s in gold_steps for p in parse_params(s)]
        p_p, p_r, p_f1 = param_f1(gp, hp, tol=PARAM_TOL)
        # keywords (TF-IDF from gold)
        k_p, k_r, k_f1, k_hit = keyword_metrics(gen_join, gold_join)
        # order
        ord_sc = order_score(matches)

        rows.append({
            "protocol_id": pid,
            "keyword_P": k_p, "keyword_R": k_r, "keyword_F1": k_f1,
            "step_P": sp, "step_R": sr, "step_F1": s_f1,
            "token_P": t_p, "token_R": t_r, "token_F1": t_f1,
            "param_P": p_p, "param_R": p_r, "param_F1": p_f1,
            "order": ord_sc,
            "n_gen_steps": len(gen_steps),
            "n_gold_steps": len(gold_steps),
            "has_gen_hier": bool(gen_hier),
            "has_gold_hier": bool(gold_hier),
            "hit_keywords": k_hit,
        })

    def avg(name):
        arr = [r[name] for r in rows if r[name] == r[name]]
        return round(float(statistics.mean(arr)), 4) if arr else 0.0

    report = {
        "n": len(rows),
        "EMB_MODEL": EMB_MODEL,
        "STEP_SIM": STEP_SIM,
        "PARAM_TOL": PARAM_TOL,
        "TOP_KEYWORDS": TOP_KEYW,
        "avg": {
            "keyword_P": avg("keyword_P"), "keyword_R": avg("keyword_R"), "keyword_F1": avg("keyword_F1"),
            "step_P": avg("step_P"), "step_R": avg("step_R"), "step_F1": avg("step_F1"),
            "token_P": avg("token_P"), "token_R": avg("token_R"), "token_F1": avg("token_F1"),
            "param_P": avg("param_P"), "param_R": avg("param_R"), "param_F1": avg("param_F1"),
            "order": avg("order"),
        }
    }
    OUTJ.write_text(json.dumps(report, indent=2), encoding="utf-8")

    # TSV
    cols = ["protocol_id", "keyword_P", "keyword_R", "keyword_F1", "step_P", "step_R", "step_F1",
            "token_P", "token_R", "token_F1", "param_P", "param_R", "param_F1", "order",
            "n_gen_steps", "n_gold_steps", "has_gen_hier", "has_gold_hier", "hit_keywords"]
    with open(OUTT, "w", encoding="utf-8") as f:
        f.write("\t".join(cols) + "\n")
        for r in rows:
            f.write("\t".join([str(r[c]) for c in cols]) + "\n")

    print(f"[OK] report: {OUTJ}")
    print(f"[OK] samples: {OUTT}")


if __name__ == "__main__":
    main()
