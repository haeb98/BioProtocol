# eval/eval_s0_retrieval_llmqueries.py
"""
LLM 기반 질의 생성 → 유사 프로토콜 검색 평가(S0 변형, 도메인 제한 없음)

입력:
  - data/gold/gold_pairs.jsonl  (article.text, protocol.text, domain, protocol_id)
  - data/rag/{corpus_fts.sqlite, faiss_by_id.index, doc_ids.npy}
환경:
  - OPENAI_API_KEY  (OpenAI 사용)
  - OPENAI_MODEL (기본: gpt-4o-mini)
  - SIM_THRESH (기본: 0.60)  # 유사도 히트 판단 임계치
  - MAX_TOK (기본: 300)      # LLM 응답 토큰 상한

출력:
  - eval/s0_llm_report.json   (SimHit@k, AvgMaxSim, 분위수 등)
  - eval/s0_llm_samples.tsv   (샘플별 최대 유사도, 사용 질의, 상위 결과 미리보기)
"""
import os
import pathlib
import re

import numpy as np
import orjson
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from rag.retriever_light import HybridRetrieverLight  # 도메인 제한 없이 사용

ROOT = pathlib.Path(".")
GOLD = ROOT / "data/gold/gold_pairs_pmc_bioprotocol_balanced.jsonl"
OUT_DIR = ROOT / "eval";
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_JSON = OUT_DIR / "s0_llm_report.json"
OUT_SAMPLES = OUT_DIR / "s0_llm_samples.tsv"

# 검색 파라미터
K_LEX = 50
K_SEM = 50
TOPN = 10
SIM_THRESH = float(os.environ.get("SIM_THRESH", "0.60"))  # 히트 판단 임계치

OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
MAX_TOK = int(os.environ.get("MAX_TOK", "300"))  # LLM 응답 토큰 상한


def load_gold(path):
    arr = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            arr.append(orjson.loads(line))
    return arr


def build_prompt(domain: str, article_text: str) -> list[dict]:
    # 입력 텍스트는 너무 길면 컷오프(2500자)
    ctx = (article_text or "").strip().replace("\r", " ")
    if len(ctx) > 2500:
        ctx = ctx[:2500]
    sys = f"You are a senior experimental protocol expert in {domain}."
    user = f"""Read the following paper text and list 3–6 concise experimental TASK NAMES needed to reproduce the core experiments.
    Each task should be a short English noun phrase (3–6 words). Do not write steps; write the names of the tasks.
    For each task, also include 2–3 key keywords in parentheses (e.g., organism, assay, reagent) to narrow search space.

    Respond with a simple numbered list only (no extra commentary).

    --- Paper text (excerpt) ---
    {ctx}
    """
    return [{"role": "system", "content": sys}, {"role": "user", "content": user}]


def parse_tasks(text: str) -> list[str]:
    lines = [l.strip() for l in (text or "").splitlines() if l.strip()]
    tasks = []
    for l in lines:
        l2 = l
        for pref in ["- ", "* "]:
            if l2.startswith(pref): l2 = l2[len(pref):]
        l2 = l2.lstrip("0123456789. )(").strip(" -–—\t")
        if 2 <= len(l2.split()) <= 10:
            tasks.append(l2)
    uniq = []
    for t in tasks:
        if t and t not in uniq:
            uniq.append(t)
    return uniq[:8]  # 최대 8개까지만 사용


def quantiles(xs):
    if not xs:
        return {"p10": 0.0, "p25": 0.0, "p50": 0.0, "p75": 0.0, "p90": 0.0}
    arr = np.array(xs, dtype=float)
    return {
        "p10": float(np.quantile(arr, 0.10)),
        "p25": float(np.quantile(arr, 0.25)),
        "p50": float(np.quantile(arr, 0.50)),
        "p75": float(np.quantile(arr, 0.75)),
        "p90": float(np.quantile(arr, 0.90)),
    }


def main():
    # 준비
    assert os.getenv("OPENAI_API_KEY"), "OPENAI_API_KEY가 필요합니다."
    client = OpenAI()
    R = HybridRetrieverLight()  # 도메인 제한 인자 사용 안 함
    emb = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    gold = load_gold(GOLD)
    sims = [];
    hits = 0

    with open(OUT_SAMPLES, "w", encoding="utf-8") as fw:
        fw.write("protocol_id\tdomain\tmax_sim\tbest_query\tretrieved_top3\n")

        for p in tqdm(gold, desc="eval-llm(nodomain)"):
            pid = p["protocol_id"]
            domain = p.get("domain", "Unknown")
            art = p["article"].get("methods_text") or ""
            proto_text = p["protocol"].get("text") or ""

            # 1) LLM 질의 생성
            msgs = build_prompt(domain, art)
            resp = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=msgs,
                temperature=0.2,
                max_tokens=MAX_TOK
            )
            raw = resp.choices[0].message.content or ""
            tasks = parse_tasks(raw)
            if not tasks:
                bq = p["protocol"].get("title") or p["protocol"].get("keywords") or ""
                tasks = [bq] if bq else []

            # 2) 검색(도메인 제한 없음)
            max_sim = 0.0
            best_q = ""
            top3_preview = ""
            proto_vec = emb.encode([proto_text], normalize_embeddings=True, convert_to_numpy=True)[0]

            for q in tasks:
                q2 = re.sub(r"[()]", " ", q).strip() + " protocol"  # 쿼리 강화 토큰
                res = R.search(q2, k_lex=K_LEX, k_sem=K_SEM, topn=TOPN)  # ← domain 파라미터 제거
                cand_texts = [r["text_preview"] for r in res]
                if not cand_texts:
                    continue
                cand_vecs = emb.encode(cand_texts, normalize_embeddings=True, convert_to_numpy=True)
                sims_vec = cand_vecs @ proto_vec
                local_max = float(sims_vec.max())
                if local_max > max_sim:
                    max_sim = local_max
                    best_q = q2
                    top3_preview = "; ".join(
                        [f"{res[i]['doc_id']}|{res[i]['title'][:60]}" for i in range(min(3, len(res)))])

            sims.append(max_sim)
            if max_sim >= SIM_THRESH:
                hits += 1

            fw.write(f"{pid}\t{domain}\t{max_sim:.3f}\t{best_q}\t{top3_preview}\n")

    n = len(gold)
    report = {
        "num_samples": n,
        "TOPN": TOPN,
        "SIM_THRESH": SIM_THRESH,
        "SimHit@TOPN": hits / n if n else 0.0,
        "AvgMaxSim": float(np.mean(sims)) if sims else 0.0,
        "Sim@quantiles": quantiles(sims)
    }
    OUT_JSON.write_text(orjson.dumps(report, option=orjson.OPT_INDENT_2).decode(), encoding="utf-8")
    print("[OK] report:", OUT_JSON)
    print(report)


if __name__ == "__main__":
    main()
