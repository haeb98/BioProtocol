# agents/s1_writer_baseline.py
"""
S1 (Writer-only) baseline with hierarchical protocol JSON.

Input  : data/gold/gold_pairs_pmc_bioprotocol_balanced.jsonl
         (each line has: protocol.hierarchical_protocol (if available),
                          protocol.text (optional),
                          article.methods_text (query source), etc.)
Output : eval/s1_generations.jsonl
         fields: protocol_id, domain, query, gen_hier (dict|None),
                 gold_hier (dict|None), gold_text (str|""), evidence_ids (list)

Env:
  OPENAI_API_KEY  (required)
  OPENAI_MODEL    (default: gpt-4o-mini)
  TOPK            (default: 5)
  K_LEX           (default: 150)
  K_SEM           (default: 200)
  TOPN            (default: 20)
  NUM_QUERIES     (default: 2)
  MAX_TOK         (default: 1200)
"""

import json
import os
import pathlib
import random
import re

import orjson
from tqdm import tqdm

# ✅ Hybrid retriever (returns dict hits)
from rag.retriever_light import HybridRetrieverLight

# OpenAI client (new SDK first, fallback to old)
try:
    from openai import OpenAI

    _HAS_NEW_OPENAI = True
except Exception:
    import openai

    _HAS_NEW_OPENAI = False

ROOT = pathlib.Path(".")
GOLD = ROOT / "data/gold/gold_pairs_pmc_bioprotocol_balanced.jsonl"
OUT_TAG = os.environ.get("OUT_TAG", "s1")  # ← 기본 s1
OUT = ROOT / f"eval/{OUT_TAG}_generations.jsonl"
OUT.parent.mkdir(parents=True, exist_ok=True)

OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
TOPK = int(os.environ.get("TOPK", "5"))
K_LEX = int(os.environ.get("K_LEX", "150"))
K_SEM = int(os.environ.get("K_SEM", "200"))
TOPN = int(os.environ.get("TOPN", "20"))
NUM_QUERIES = int(os.environ.get("NUM_QUERIES", "2"))
MAX_TOK = int(os.environ.get("MAX_TOK", "1200"))
SEED = 42

USE_RERANK = bool(int(os.environ.get("USE_RERANK", "0")))
USE_HYDE = bool(int(os.environ.get("USE_HYDE", "0")))
USE_MMR = bool(int(os.environ.get("USE_MMR", "0")))
MMR_LAMBDA = float(os.environ.get("MMR_LAMBDA", "0.5"))  # 0~1, 0=유사도만, 1=다양성↑

if USE_RERANK:
    from sentence_transformers import CrossEncoder

    RERANK_MODEL = os.environ.get("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    reranker = CrossEncoder(RERANK_MODEL)
else:
    reranker = None

# === System prompts ===
SYS_QGEN = (
    "You are an expert at reading biomedical Methods sections and formulating search intents.\n"
    "From the given METHODS text, extract 2-3 concise task queries that would retrieve similar protocols.\n"
    "Each query should be 8-16 words, include core assay/organism/material keywords, and end with the word 'protocol'.\n"
    "Return one query per line, no numbering."
)

# 계층형 JSON만 출력하도록 강제
SYS_WRITER = (
    "You are a biomedical methods expert. Use ONLY the provided EVIDENCE to output a hierarchical protocol "
    "in the exact JSON schema below. Do not add any extra keys.\n\n"
    "Schema (JSON object):\n"
    "{\n"
    '  "hierarchical_protocol": {\n'
    '    "1": {"title": "Section title"},\n'
    '    "1.1": "First step sentence...",\n'
    '    "1.2": "Second step...",\n'
    '    "2": {"title": "Next section"},\n'
    '    "2.1": "Step...",\n'
    '    "2.2": "Step..."\n'
    "  }\n"
    "}\n\n"
    "Rules:\n"
    "- Keep keys as dotted numeric strings (e.g., 1, 1.1, 1.2, 2, 2.1...).\n"
    "- Use concise, executable steps with parameters (time, temperature, volumes, concentrations) when present.\n"
    "- If materials are needed, include them as early steps (e.g., 'Prepare ...').\n"
    "- Cite sources inline in step text as [EVIDENCE i] when appropriate.\n"
    "- Return ONLY valid JSON. No markdown fences, no explanations."
)


def hyde_query(client, methods_text):
    hyp = call_chat(
        client,
        "You generate hypothetical protocol paragraphs.",
        "Write a concise hypothetical protocol paragraph (5-7 sentences) capturing the key tasks from METHODS:\n"
        + methods_text[:4000]
    )
    # 이 문단을 질의로 쓰면 너무 길어서 압축
    q = call_chat(
        client,
        "You compress queries.",
        f"Compress the following into a single 12-16 word search query ending with 'protocol':\n{hyp}"
    )
    q = q.strip().rstrip(".")
    if not q.lower().endswith("protocol"):
        q += " protocol"
    return q


def openai_client():
    if _HAS_NEW_OPENAI:
        return OpenAI()
    return openai.OpenAI()


def call_chat(client, system_prompt, user_prompt, max_tokens=MAX_TOK, temperature=0.2):
    if _HAS_NEW_OPENAI:
        r = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "system", "content": system_prompt},
                      {"role": "user", "content": user_prompt}],
            temperature=temperature, max_tokens=max_tokens
        )
        return r.choices[0].message.content
    else:
        r = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "system", "content": system_prompt},
                      {"role": "user", "content": user_prompt}],
            temperature=temperature, max_tokens=max_tokens
        )
        return r.choices[0].message.content


def llm_generate_queries(client, methods_text, n=2):
    """Generate n search queries from METHODS text."""
    prompt = f"METHODS:\n{methods_text}\n\nReturn {n} queries:"
    out = (call_chat(client, SYS_QGEN, prompt, max_tokens=300, temperature=0.2) or "").strip().splitlines()
    qs = []
    for s in out:
        s = s.strip().rstrip(".")
        if not s:
            continue
        if not s.lower().endswith("protocol"):
            s = s + " protocol"
        qs.append(s)
    if len(qs) < n and qs:
        while len(qs) < n:
            qs.append(qs[0])
    return qs[:n] if qs else []


# --- hit dict helpers ---
def hit_doc_id(hit: dict):
    return hit.get("doc_id") or hit.get("id") or hit.get("docId") or hit.get("pk")


def hit_title(hit: dict):
    return hit.get("title") or hit.get("doc_title") or ""


def hit_preview(hit: dict):
    return hit.get("preview") or hit.get("text") or hit.get("content") or hit.get("snippet") or ""


def fuse_evidence_for_writer(R: HybridRetrieverLight, queries, topk=TOPK):
    """Hybrid search over multiple queries, dedup, take topk."""
    evidences = []
    seen = set()
    for q in queries:
        hits = R.search(q, k_lex=K_LEX, k_sem=K_SEM, topn=TOPN) or []
        for h in hits:
            did = hit_doc_id(h)
            if did is None or did in seen:
                continue
            seen.add(did)
            evidences.append({
                "doc_id": did,
                "title": hit_title(h),
                "preview": hit_preview(h),
            })
            if reranker and evidences:
                pairs = [[queries[0], e["preview"]] for e in evidences]  # 첫 질의 기준
                scores = reranker.predict(pairs)  # numpy array
                evidences = [e for _, e in sorted(zip(scores, evidences), key=lambda x: -x[0])]
            if len(evidences) >= topk:
                break
        if len(evidences) >= topk:
            break
    # enumerate
    return [{"i": i + 1, **e} for i, e in enumerate(evidences)]


def build_writer_prompt(title, queries, evidences):
    qtxt = "\n".join([f"- {q}" for q in queries])
    ctx = "\n\n".join([f"[EVIDENCE {e['i']}] {e['preview']}" for e in evidences])
    return (
        f"STUDY TITLE: {title}\n\n"
        f"SEARCH INTENTS (derived from Methods):\n{qtxt}\n\n"
        f"EVIDENCE snippets (use only this info; cite as [EVIDENCE i]):\n{ctx}\n\n"
        "Produce the JSON object as specified in the system message."
    )


# --- robust JSON parse (handles ```json blocks, extra text) ---
_BRACE = re.compile(r"\{[\s\S]*\}")


def parse_hier_json(gen_str):
    # 1) direct parse
    try:
        obj = json.loads(gen_str)
        hp = obj.get("hierarchical_protocol")
        if isinstance(hp, dict):
            return hp
    except Exception:
        pass
    # 2) extract first JSON object block
    m = _BRACE.search(gen_str or "")
    if m:
        try:
            obj = json.loads(m.group(0))
            hp = obj.get("hierarchical_protocol")
            if isinstance(hp, dict):
                return hp
        except Exception:
            pass
    return None


def main():
    random.seed(SEED)
    client = openai_client()
    R = HybridRetrieverLight()

    with open(GOLD, "r", encoding="utf-8") as fin, open(OUT, "w", encoding="utf-8") as fou:
        for line in tqdm(fin, desc="s1-write"):
            pair = orjson.loads(line)
            pid = pair["protocol_id"]
            title = pair["protocol"].get("title", "")
            gold_hier = pair["protocol"].get("hierarchical_protocol")  # dict|None
            gold_text = pair["protocol"].get("text", "")  # optional
            methods_text = pair["article"].get("methods_text", "")

            # 1) Methods → LLM queries
            queries = llm_generate_queries(client, methods_text, n=NUM_QUERIES)
            if USE_HYDE:
                q_hyde = hyde_query(client, methods_text)
                queries = list(dict.fromkeys(queries + [q_hyde]))
            if not queries:
                queries = [f"{title} protocol"]

            # 2) Retrieve evidences
            evidences = fuse_evidence_for_writer(R, queries, topk=TOPK)

            # 3) Writer prompt & call
            prompt = build_writer_prompt(title, queries, evidences)
            gen_raw = call_chat(client, SYS_WRITER, prompt, max_tokens=MAX_TOK, temperature=0.2)

            # 4) Parse hierarchical JSON
            gen_hier = parse_hier_json(gen_raw)

            # 5) Save
            rec = {
                "protocol_id": pid,
                "domain": pair.get("domain", "Unknown"),
                "query": queries,
                "gen_hier": gen_hier,  # dict or None
                "gold_hier": gold_hier,  # dict or None
                "gold_text": gold_text,  # string (fallback)
                "evidence_ids": [e["doc_id"] for e in evidences]
            }
            fou.write(orjson.dumps(rec).decode() + "\n")

    print(f"[OK] saved generations -> {OUT}")


if __name__ == "__main__":
    main()
