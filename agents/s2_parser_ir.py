# agents/s2_parser_ir.py
import json
import os
import pathlib
import random
import re
import time

import orjson
from tqdm import tqdm

# ---- retriever (RAG) ----
from rag.retriever_light import HybridRetrieverLight

# ---- OpenAI client ----
try:
    from openai import OpenAI

    _HAS_NEW = True
except Exception:
    import openai

    _HAS_NEW = False

ROOT = pathlib.Path(".")
GOLD = ROOT / "data/gold/gold_pairs_pmc_bioprotocol_balanced.jsonl"
OUT = ROOT / "eval/s2_ir.jsonl"
DBG = ROOT / "eval/_debug_s2_ir"
OUT.parent.mkdir(parents=True, exist_ok=True)
DBG.mkdir(parents=True, exist_ok=True)

# ---- Retrieval knobs ----
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
K_LEX = int(os.environ.get("K_LEX", "200"))
K_SEM = int(os.environ.get("K_SEM", "300"))
TOPN = int(os.environ.get("TOPN", "50"))
TOPK = int(os.environ.get("TOPK", "12"))

USE_RERANK = bool(int(os.environ.get("USE_RERANK", "1")))
RERANK_MODEL = os.environ.get("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
USE_MMR = bool(int(os.environ.get("USE_MMR", "1")))
MMR_LAMBDA = float(os.environ.get("MMR_LAMBDA", "0.5"))
USE_HYDE = bool(int(os.environ.get("USE_HYDE", "1")))
NUM_QUERIES = int(os.environ.get("NUM_QUERIES", "2"))

# ---- Self-evidence knobs ----
USE_SELF_EVIDENCE = bool(int(os.environ.get("USE_SELF_EVIDENCE", "0")))
FALLBACK_SELF = bool(int(os.environ.get("FALLBACK_SELF", "1")))
SELF_SECTIONS = os.environ.get("SELF_SECTIONS", "methods").split(",")  # methods,abstract,intro,full
SELF_WIN = int(os.environ.get("SELF_WIN", "1200"))
SELF_STEP = int(os.environ.get("SELF_STEP", "1000"))
EVID_TRUNC = int(os.environ.get("EVID_TRUNC", "1800"))

SEED = 42
random.seed(SEED)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# ---- Optional reranker / embedder ----
from sentence_transformers import SentenceTransformer, util as st_util

_reranker = None
if USE_RERANK:
    from sentence_transformers import CrossEncoder

    _reranker = CrossEncoder(RERANK_MODEL)

_emb = None


def emb():
    global _emb
    if _emb is None:
        _emb = SentenceTransformer(os.environ.get("EMB_MODEL", "sentence-transformers/all-MiniLM-L6-v2"))
    return _emb


def mmr_select(query_text, evidences, k=TOPK, lamb=MMR_LAMBDA):
    if len(evidences) <= k:
        return list(range(len(evidences)))
    q = emb().encode([query_text], convert_to_tensor=True, normalize_embeddings=True)
    D = emb().encode([e["preview"] for e in evidences], convert_to_tensor=True, normalize_embeddings=True)
    sim_q = st_util.cos_sim(q, D)[0]
    sim_dd = st_util.cos_sim(D, D)
    import torch
    selected = []
    rest = list(range(len(evidences)))
    for _ in range(k):
        best, best_score = None, -1e9
        for i in rest:
            div = 0 if not selected else torch.max(sim_dd[i, selected]).item()
            score = lamb * sim_q[i].item() - (1 - lamb) * div
            if score > best_score:
                best, best_score = i, score
        selected.append(best)
        rest.remove(best)
    return selected


# ---- Guard rails for IR ----
HEADER_TOKENS = {"meta", "materials", "solutions", "equipment", "steps", "dependencies", "outputs"}
ALLOWED_ACTIONS = [
    "add", "mix", "incubate", "centrifuge", "wash", "filter", "resuspend", "dissolve",
    "measure", "read", "inject", "prepare", "stain", "vortex", "sonicate", "pipette",
    "aliquot", "store", "dry", "heat", "cool", "spin", "pellet", "separate", "equilibrate",
    "calibrate", "label", "rinse", "mount", "transfer", "load", "elute", "dilute", "prewarm",
    "precool", "thaw", "freeze", "count", "seed", "harvest"
]
ALLOWED_ACTIONS_SET = set(ALLOWED_ACTIONS)


def nonempty(x):
    return x is not None and (x != "") and (x != []) and (x != {})


def is_valid_step(step: dict) -> bool:
    if not isinstance(step, dict):
        return False
    action = (step.get("action") or "").strip().lower()
    if not action or action in HEADER_TOKENS or action not in ALLOWED_ACTIONS_SET:
        return False
    # payload check: 최소 하나의 파라미터/입력/조건/출력이 있어야 함
    params = step.get("params") or {}
    inputs = step.get("inputs") or []
    outputs = step.get("outputs") or []
    has_payload = any([
        (isinstance(params, dict) and any(nonempty(v) for v in params.values())),
        (isinstance(inputs, list) and any(nonempty(i) for i in inputs)),
        (isinstance(outputs, list) and any(nonempty(o) for o in outputs))
    ])
    return has_payload


def clean_ir(ir: dict) -> dict:
    """ 스키마를 보존하되, 가짜/빈 헤더 스텝 제거 + 빈 섹션 삭제 """
    if not isinstance(ir, dict):
        return {}
    out = {}

    # meta/materials/solutions/equipment: 증거가 없으면 생략
    for k in ["meta", "materials", "solutions", "equipment", "outputs", "dependencies"]:
        v = ir.get(k)
        if k == "meta" and isinstance(v, dict) and any(nonempty(v.get(x)) for x in v.keys()):
            out[k] = v
        elif k in {"materials", "solutions", "equipment", "outputs"} and isinstance(v, list):
            vv = [x for x in v if isinstance(x, dict) and any(nonempty(x.get(kk)) for kk in x.keys())]
            if vv:
                out[k] = vv
        elif k == "dependencies" and isinstance(v, list):
            edges = []
            for e in v:
                if isinstance(e, (list, tuple)) and len(e) == 2 and all(isinstance(s, str) and s for s in e):
                    edges.append([e[0], e[1]])
            if edges:
                out[k] = edges

    # steps: 오직 유효 action만 남김
    raw_steps = ir.get("steps") or []
    steps = []
    for st in raw_steps:
        if is_valid_step(st):
            steps.append(st)
    if steps:
        out["steps"] = steps

    return out


# ---- Prompts ----
SYS_IR_STRICT = f"""
You are a strict biomedical METHODS extractor.
Goal: build a SINGLE JSON object for an EXECUTABLE protocol IR using ONLY the EVIDENCE provided.

Hard constraints:
- DO NOT output section headers or placeholders as steps. Forbidden tokens (anywhere in 'action'): {sorted(list(HEADER_TOKENS))}
- Each step MUST be an atomic lab action whose 'action' is one of: {ALLOWED_ACTIONS}
- Fill ONLY fields supported by evidence; DO NOT hallucinate.
- Normalize units to: µL/mL/°C/min/rpm/x g/mM/µM/% when possible.
- Return ONLY JSON. No markdown, no comments, no extra text.

Target JSON schema (keys exactly):
{{
  "meta": {{"title":"", "organism":"", "assay":"", "domain":""}},
  "materials": [{{"name":"", "role":"", "amount":{{"value":0,"unit":""}}, "concentration":{{"value":0,"unit":""}}, "catalog":"", "supplier":""}}],
  "solutions": [{{"name":"", "recipe":[{{"name":"", "amount":{{"value":0,"unit":""}}}}], "final_pH": null}}],
  "equipment": [{{"name":"", "model":"", "rpm_max": null}}],
  "steps": [
    {{
      "id":"1.1",
      "action":"<one of allowed actions>",
      "inputs": ["<materials/solutions/equipment refs>"],
      "outputs": [],
      "params": {{
        "time": {{ "value": 0, "unit":"min" }},
        "temperature": {{ "value": 0, "unit":"°C" }},
        "volume": {{ "value": 0, "unit":"µL" }},
        "speed": {{ "value": 0, "unit":"rpm" }},
        "g": {{ "value": 0, "unit":"x g" }},
        "concentration": {{ "value": 0, "unit":"mM" }}
      }},
      "notes": "",
      "evidence": [1]
    }}
  ],
  "dependencies": [["1.1","1.2"]],
  "outputs": [{{"name":"", "type":""}}]
}}
"""

SYS_IR_RELAX = """
Relaxed mode: use the same JSON schema. If a field is unknown, omit it.
Keep the constraints: steps must be real lab actions (no headers), and return ONLY JSON.
"""

SYS_QGEN = (
    "You read biomedical METHODS and produce concise search queries. "
    "Return 2-3 queries, each 8-16 words, ending with 'protocol'. One per line, no numbering."
)


def openai_client():
    return OpenAI() if _HAS_NEW else openai.OpenAI()


def chat(client, system, user, max_tokens=1500, temperature=0.0):
    r = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=temperature, max_tokens=max_tokens
    )
    return r.choices[0].message.content


def gen_queries(client, methods, n=NUM_QUERIES):
    txt = chat(client, SYS_QGEN, f"METHODS:\n{methods[:8000]}\nReturn {n} queries:", max_tokens=300, temperature=0.2)
    out = []
    for line in (txt or "").splitlines():
        s = line.strip().rstrip(".")
        if not s:
            continue
        if not s.lower().endswith("protocol"):
            s += " protocol"
        out.append(s)
    return out[:max(1, n)] or ["experimental protocol"]


def hyde_query(client, methods):
    hyp = chat(client, "You generate hypothetical protocols.",
               f"Write a concise hypothetical protocol paragraph (5-7 sentences) from METHODS:\n{methods[:4000]}")
    q = chat(client, "You compress queries.",
             "Compress into a single 12-16 word search query ending with 'protocol':\n" + (hyp or "")[:1500],
             max_tokens=80, temperature=0.2)
    q = (q or "").strip().rstrip(".")
    if not q.lower().endswith("protocol"):
        q += " protocol"
    return q


def parse_json(s):
    try:
        return json.loads(s)
    except Exception:
        m = re.search(r"\{[\s\S]*\}", s or "")
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                return None
        return None


def valid_ir(ir):
    if not isinstance(ir, dict):
        return False
    steps = ir.get("steps") or []
    return isinstance(steps, list) and any(is_valid_step(s) for s in steps)


def save_debug(pid, blob):
    (DBG / f"{pid}.json").write_text(json.dumps(blob, ensure_ascii=False, indent=2), encoding="utf-8")


# ---- self-evidence builder ----
def windows(s, w=SELF_WIN, step=SELF_STEP):
    s = s or ""
    if len(s) <= w:
        return [s]
    out, i = [], 0
    while i < len(s):
        out.append(s[i:i + w])
        i += step
    return out


def pick_sections(pair):
    txts = []
    art = pair.get("article") or {}
    if "full" in SELF_SECTIONS:
        for key in ["title", "abstract_text", "intro_text", "methods_text", "results_text",
                    "discussion_text", "conclusion_text", "ack_text"]:
            t = art.get(key, "")
            if t:
                txts.append(t)
    else:
        if "methods" in SELF_SECTIONS:
            t = art.get("methods_text", "");
            txts.append(t or "")
        if "abstract" in SELF_SECTIONS:
            t = art.get("abstract_text", "");
            txts.append(t or "")
        if "intro" in SELF_SECTIONS:
            t = art.get("intro_text", "");
            txts.append(t or "")
    return "\n\n".join([t for t in txts if t]).strip()


def build_self_evidence(pair, topk=TOPK):
    base = pick_sections(pair)
    if not base:
        return []
    chunks = windows(base, w=SELF_WIN, step=SELF_STEP)
    ev = []
    for i, ch in enumerate(chunks[:topk]):
        ev.append({"doc_id": f"self:{i + 1}", "preview": ch[:EVID_TRUNC]})
    return ev


def main():
    cli = openai_client()
    R = HybridRetrieverLight()

    with open(GOLD, "r", encoding="utf-8") as fin, open(OUT, "w", encoding="utf-8") as fou:
        for line in tqdm(fin, desc="s2-parse-IR"):
            pair = orjson.loads(line)
            pid = str(pair["protocol_id"])
            title = pair.get("protocol", {}).get("title", "")
            methods = pair.get("article", {}).get("methods_text", "") or ""

            # --- evidence collection ---
            evid = []
            used_self = False

            if USE_SELF_EVIDENCE:
                evid = build_self_evidence(pair, topk=TOPK)
                used_self = True
            else:
                # 1) query gen
                queries = gen_queries(cli, methods, n=NUM_QUERIES)
                if USE_HYDE:
                    try:
                        qh = hyde_query(cli, methods)
                        queries = list(dict.fromkeys(queries + [qh]))
                    except Exception:
                        pass
                # 2) retrieve
                cand, seen = [], set()
                for q in queries:
                    hits = R.search(q, k_lex=K_LEX, k_sem=K_SEM, topn=TOPN) or []
                    for h in hits:
                        did = h.get("doc_id") or h.get("id")
                        if not did or did in seen:
                            continue
                        seen.add(did)
                        txt = h.get("preview") or h.get("text") or h.get("content") or ""
                        if not txt:
                            continue
                        cand.append({"doc_id": did, "preview": txt})
                # 3) rerank / mmr / topk / fallback
                if not cand and FALLBACK_SELF:
                    evid = build_self_evidence(pair, topk=TOPK)
                    used_self = True
                else:
                    if _reranker and USE_RERANK and len(cand) > 3:
                        pairs = [[queries[0], c["preview"]] for c in cand]
                        scores = _reranker.predict(pairs)
                        cand = [c for _, c in sorted(zip(scores, cand), key=lambda x: -x[0])]
                    if USE_MMR and len(cand) > TOPK:
                        sel = mmr_select(" ".join(queries), cand, k=TOPK, lamb=MMR_LAMBDA)
                        cand = [cand[i] for i in sel]
                    else:
                        cand = cand[:TOPK]
                    evid = cand

            for c in evid:
                c["preview"] = (c.get("preview") or "")[:EVID_TRUNC]

            if not evid:
                save_debug(pid, {"reason": "no_evidence", "used_self": used_self, "sections": SELF_SECTIONS})
                fou.write(orjson.dumps({"protocol_id": pid, "domain": pair.get("domain", "Unknown"),
                                        "ir": None, "evidence_ids": []}).decode() + "\n")
                continue

            # --- LLM call ---
            ev_txt = "\n\n".join([f"[EVIDENCE {i + 1}] {c['preview']}" for i, c in enumerate(evid)])
            user = f"STUDY TITLE: {title}\n\nEVIDENCE:\n{ev_txt}"

            raw1 = chat(cli, SYS_IR_STRICT, user, max_tokens=1800, temperature=0.0)
            ir1 = parse_json(raw1)
            ir1 = clean_ir(ir1) if isinstance(ir1, dict) else {}

            attempt = "strict"
            if not valid_ir(ir1):
                raw2 = chat(cli, SYS_IR_RELAX, user, max_tokens=1800, temperature=0.0)
                ir2 = parse_json(raw2)
                ir2 = clean_ir(ir2) if isinstance(ir2, dict) else {}
                # 두 시도 중 더 나은 쪽 선택(유효 step 수 우선)
                s1 = len((ir1.get("steps") or [])) if isinstance(ir1, dict) else 0
                s2 = len((ir2.get("steps") or [])) if isinstance(ir2, dict) else 0
                ir = ir2 if s2 >= s1 else ir1
                attempt = "relax" if ir is ir2 else "strict"
            else:
                ir = ir1

            # debug dump
            save_debug(pid, {
                "pid": pid,
                "used_self": used_self,
                "evidence_count": len(evid),
                "evidence_ids": [e["doc_id"] for e in evid],
                "attempt": attempt,
                "raw_len_1": len(raw1 or ""),
                "raw_len_2": len(raw2 or "") if 'raw2' in locals() else 0
            })

            rec = {
                "protocol_id": pid,
                "domain": pair.get("domain", "Unknown"),
                "ir": ir if valid_ir(ir) else None,
                "evidence_ids": [e["doc_id"] for e in evid]
            }
            fou.write(orjson.dumps(rec).decode() + "\n")
            time.sleep(0.05)

    print(f"[OK] saved IR -> {OUT}")


if __name__ == "__main__":
    main()
