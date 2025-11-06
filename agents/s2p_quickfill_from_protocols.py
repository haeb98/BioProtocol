# agents/s2p_quickfill_from_protocols.py
import argparse
import json
import random
import re
from collections import defaultdict, Counter


# ---- 간단 TF-IDF / BM25 검색기 ----
def build_retriever(docs):
    try:
        from rank_bm25 import BM25Okapi
        tok = lambda t: re.findall(r"[A-Za-z0-9µμ%/.\-]+", t.lower())
        corpus = [tok(d["text"]) for d in docs]
        bm25 = BM25Okapi(corpus)

        def search(q, k=8):
            qtok = tok(q)
            scores = bm25.get_scores(qtok)
            idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
            return [(i, float(scores[i])) for i in idx]

        return search
    except Exception:
        # TF-IDF fallback
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        vec = TfidfVectorizer(max_features=50000)
        X = vec.fit_transform([d["text"] for d in docs])

        def search(q, k=8):
            qv = vec.transform([q])
            sc = cosine_similarity(qv, X)[0]
            idx = sc.argsort()[::-1][:k]
            return [(int(i), float(sc[i])) for i in idx]

        return search


UNITS_MAP = {
    "℃": "°C", "c": "°C", "°c": "°C",
    "ul": "µL", "ul.": "µL", "uL": "µL", "μl": "µL", "μL": "µL",
    "ml": "mL", "l": "L",
    "hr": "h", "hrs": "h", "hour": "h", "hours": "h",
    "min": "min", "mins": "min", "s": "s", "sec": "s", "secs": "s",
    "g": "g", "mg": "mg", "µg": "µg", "ug": "µg", "ng": "ng",
    "rpm": "rpm", "×g": "×g", "xg": "×g",
    "m": "M", "mm": "mM", "µm": "µM", "um": "µM", "nm": "nM",
    "%": "%"
}

# 값(숫자) + 단위 캡쳐
PARAM_PAT = re.compile(
    r"(?P<val>\d+(?:\.\d+)?)\s*(?P<unit>°C|℃|C|c|h|hr|hrs|hour|hours|min|mins|s|sec|secs|mL|ml|L|l|µL|μL|ul|uL|g|mg|µg|ug|ng|rpm|×g|xg|M|mM|µM|uM|nM|%)\b"
)

COMMON_STOP = set("""
the and or of to for with in on at by from as a an is are be was were this that these those it its their into over under using use used solution buffer medium media plate tubes water ddh2o pbs agar ethanol methanol acetic acid sodium chloride
""".split())


def normalize_unit(u):
    if not isinstance(u, str):
        return u
    u = u.strip()
    ul = u.lower()
    return UNITS_MAP.get(ul, u)


def extract_params(text):
    out = []
    for m in PARAM_PAT.finditer(text or ""):
        v = m.group("val")
        u = normalize_unit(m.group("unit"))
        out.append({"value": v, "unit": u, "span": m.group(0)})
    return out


def extract_materials(text):
    toks = re.findall(r"[A-Za-z0-9\-\+\(\)µμ%/\.]+", text or "")
    cands = []
    for t in toks:
        if len(t) < 2:
            continue
        if t.lower() in COMMON_STOP:
            continue
        if t[0].isupper() or any(c.isdigit() for c in t):
            cands.append(t)
    cnt = Counter(cands)
    mats = [w for w, _ in cnt.most_common(10)]
    return mats


def best_query_for_step(step):
    q_parts = []
    for k in ("text", "raw", "sentence", "desc", "step_text"):
        v = step.get(k)
        if isinstance(v, str) and v.strip():
            q_parts.append(v.strip())
    for arrk in ("actions", "materials"):
        arr = step.get(arrk)
        if isinstance(arr, list) and arr:
            q_parts.extend([str(x) for x in arr if x])
    if "parameters" in step and isinstance(step["parameters"], list):
        q_parts.extend([f'{p.get("value", "")} {p.get("unit", "")}'.strip()
                        for p in step["parameters"] if isinstance(p, dict)])
    q = " ".join(q_parts).strip()
    return q[:1000] if q else ""


def fill_slots_from_docs(step, docs, hits,
                         need_params=True, need_mats=True,
                         topn_per_hit=3, conf_w=0.8, min_conf=0.55):
    filled = {"parameters": [], "materials": []}
    evid = {"parameters": [], "materials": []}
    sco = defaultdict(float)

    if need_params:
        for idx, score in hits:
            score = float(score)
            ps = extract_params(docs[idx]["text"])[:topn_per_hit]
            for p in ps:
                key = (p["value"], p["unit"])
                sco[("param", key)] += conf_w * score + (1.0 - conf_w)
                evid["parameters"].append({
                    "value": p["value"], "unit": p["unit"],
                    "doc_id": docs[idx].get("doc_id", idx),
                    "score": score, "span": p["span"]
                })
        chosen = []
        for (tag, key), sc in sorted(sco.items(), key=lambda x: x[1], reverse=True):
            if tag != "param":
                continue
            if sc < min_conf:
                continue
            v, u = key
            chosen.append({"value": v, "unit": u, "confidence": float(sc)})
            if len(chosen) >= 5:
                break
        filled["parameters"] = chosen

    if need_mats:
        for idx, score in hits:
            score = float(score)
            ms = extract_materials(docs[idx]["text"])[:topn_per_hit]
            for m in ms:
                sco[("mat", m)] += conf_w * score + (1.0 - conf_w)
                evid["materials"].append({
                    "name": m, "doc_id": docs[idx].get("doc_id", idx), "score": score
                })
        mats = []
        for (tag, key), sc in sorted(sco.items(), key=lambda x: x[1], reverse=True):
            if tag != "mat":
                continue
            if sc < min_conf:
                continue
            mats.append({"name": key, "confidence": float(sc)})
            if len(mats) >= 10:
                break
        filled["materials"] = mats

    return filled, evid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", default="runs/s2_llm_top15.ir.jsonl")
    ap.add_argument("--corpus", default="data/rag/corpus/protocols_wo_test25.jsonl")
    ap.add_argument("--out", default="runs/s2p_quickfill_top15.ir.jsonl")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min_conf", type=float, default=0.55)
    args = ap.parse_args()
    random.seed(args.seed)

    # 코퍼스 로드
    docs = []
    with open(args.corpus, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            txt = r.get("text") or r.get("chunk") or r.get("content") or ""
            if not txt:
                continue
            docs.append({
                "doc_id": r.get("doc_id") or r.get("id"),
                "text": txt,
                "meta": {k: r.get(k) for k in ("source", "protocol_id", "section_title", "step_idx")}
            })
    if not docs:
        raise SystemExit(f"empty corpus: {args.corpus}")

    search = build_retriever(docs)

    # IR 로드 & 보강
    out = open(args.out, "w", encoding="utf-8")
    n = 0;
    updated = 0
    for line in open(args.pred, "r", encoding="utf-8"):
        if not line.strip():
            continue
        rec = json.loads(line)

        # steps 키 이름 다양성 방어
        steps = rec.get("steps")
        if not isinstance(steps, list):
            steps = rec.get("ir")
        if not isinstance(steps, list):
            steps = []
        changed = False

        for s in steps:
            # 필드 초기화 방어
            if "parameters" not in s or not isinstance(s.get("parameters"), list):
                s["parameters"] = []
            if "materials" not in s or not isinstance(s.get("materials"), list):
                s["materials"] = []

            q = best_query_for_step(s)
            if not q:
                continue
            hits = search(q, k=args.k)

            need_params = (not s.get("parameters")) or len(s["parameters"]) < 2
            need_mats = (not s.get("materials")) or len(s["materials"]) < 2

            if not (need_params or need_mats):
                continue

            filled, evid = fill_slots_from_docs(
                s, docs, hits, need_params, need_mats, min_conf=args.min_conf
            )

            # 원본 백업 & 적용
            if need_params:
                s.setdefault("parameters_raw", list(s.get("parameters") or []))
                if filled["parameters"]:
                    s["parameters"] = (s.get("parameters") or []) + filled["parameters"]
                    s["parameters_evidence"] = (s.get("parameters_evidence") or []) + evid["parameters"]
                    changed = True
            if need_mats:
                s.setdefault("materials_raw", list(s.get("materials") or []))
                if filled["materials"]:
                    # materials는 문자열 리스트 유지
                    s["materials"] = (s.get("materials") or []) + [m["name"] for m in filled["materials"]]
                    s["materials_evidence"] = (s.get("materials_evidence") or []) + evid["materials"]
                    changed = True

        out.write(json.dumps(rec, ensure_ascii=False) + "\n")
        n += 1
        if changed:
            updated += 1
    out.close()
    print(f"[OK] wrote -> {args.out} (records={n}, updated={updated})")


if __name__ == "__main__":
    main()
