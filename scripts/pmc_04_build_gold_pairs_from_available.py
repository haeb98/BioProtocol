# scripts/pmc_04_build_gold_pairs_from_available.py
"""
methods 가 추출된 기사(gold_articles_methods_pmc.jsonl)에서
도메인 상위 N개(기본 12개)만 선택하고, 각 도메인당 최대 K개(기본 5개) 샘플링하여
bioprotocol 원문과 매핑한 gold 페어를 생성합니다.

환경변수:
- TOP_K_DOMAINS: 상위 도메인 개수 (기본 12)
- TAKE_PER_DOMAIN: 도메인당 샘플 개수 (기본 5)

출력:
- data/gold/gold_pairs_pmc_bioprotocol_balanced.jsonl
  (article.methods_text + protocol.text 묶음)
- 실행 로그에 도메인별 선택 개수 요약 출력
"""
import orjson
import os
import pathlib
import random
from collections import Counter, defaultdict

ROOT = pathlib.Path(".")
BIO = ROOT / "data/raw/bio_protocol.json"
ART = ROOT / "data/gold/gold_articles_methods_pmc.jsonl"
OUT = ROOT / "data/gold/gold_pairs_pmc_bioprotocol_balanced.jsonl"
OUT.parent.mkdir(parents=True, exist_ok=True)

SEED = 42
TOP_K_DOMAINS = int(os.environ.get("TOP_K_DOMAINS", "12"))
TAKE_PER_DOMAIN = int(os.environ.get("TAKE_PER_DOMAIN", "5"))


def flatten_hier(h):
    if not isinstance(h, dict): return ""

    def keyer(k):
        return [int(p) if p.isdigit() else p for p in k.split(".")]

    lines = []
    for k in sorted(h.keys(), key=keyer):
        v = h[k]
        if isinstance(v, dict) and "title" in v:
            lines.append(f"## {v['title']}")
        elif isinstance(v, str):
            lines.append(v)
    return "\n".join(lines)


def build_protocol_text(r):
    title = r.get("title") or ""
    inputs = r.get("input") or ""
    proto = r.get("protocol") or ""
    hier = flatten_hier(r.get("hierarchical_protocol") or {})
    parts = [p for p in [title, "# Materials & Inputs", inputs, "# Protocol", proto, "# Outline", hier] if
             p and str(p).strip()]
    return "\n\n".join(parts)


def main():
    random.seed(SEED)

    # 0) 로드
    bio = orjson.loads(BIO.read_bytes())
    bio_idx = {str(x.get("id")): x for x in bio}

    methods = []
    with open(ART, "r", encoding="utf-8") as f:
        for line in f:
            methods.append(orjson.loads(line))

    if not methods:
        raise SystemExit("No methods records found. Run pmc_03_extract_methods_from_jats.py first.")

    # 1) methods가 존재하는 도메인 상위 N개 선정
    dom_cnt = Counter([m.get("domain", "Unknown") for m in methods])
    top_domains = [d for d, _ in dom_cnt.most_common(TOP_K_DOMAINS)]

    # 2) 버킷 구성(상위 도메인만)
    buckets = defaultdict(list)
    for m in methods:
        d = m.get("domain", "Unknown")
        if d in top_domains:
            buckets[d].append(m)

    # 3) 샘플링(도메인당 최대 K개)
    total = 0
    picked_counts = {}
    with open(OUT, "w", encoding="utf-8") as fout:
        for d in top_domains:
            arr = buckets.get(d, [])
            random.shuffle(arr)
            pick = arr[:TAKE_PER_DOMAIN]
            picked_counts[d] = len(pick)
            for a in pick:
                br = bio_idx.get(a["protocol_id"])
                if not br:
                    continue
                pair = {
                    "protocol_id": a["protocol_id"],
                    "domain": d,
                    "article": {
                        "id": a["article_id"],
                        "pmid": a.get("pmid", ""),
                        "pmcid": a.get("pmcid", ""),
                        "methods_text": a["methods_text"],
                        "xml_path": a.get("xml_path", "")
                    },
                    "protocol": {
                        "title": br.get("title", ""),
                        "text": build_protocol_text(br),
                        "keywords": br.get("keywords", ""),
                        "url": br.get("url", "")
                    }
                }
                fout.write(orjson.dumps(pair).decode() + "\n")
                total += 1

    print(f"[OK] {OUT} (pairs={total})")
    print("== Selected per-domain counts ==")
    for d in top_domains:
        print(f"{d:40s} {picked_counts.get(d, 0)}")


if __name__ == "__main__":
    main()
