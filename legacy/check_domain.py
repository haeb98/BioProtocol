# scripts/gold_00_summarize_methods_domains.py
import pathlib
from collections import Counter

import orjson

ROOT = pathlib.Path("../scripts")
BIO = ROOT / "data/raw/bio_protocol.json"
ART = ROOT / "data/gold/gold_articles_methods_pmc.jsonl"


def main():
    bio = orjson.loads(BIO.read_bytes())
    # bioprotocol 전체 도메인 카운트(eligibility 판단 기준)
    dom_all = [(x.get("classification") or {}).get("primary_domain", "Unknown") for x in bio]
    all_cnt = Counter(dom_all)

    # 이번에 methods 추출된 기사들의 도메인 카운트
    have = []
    with open(ART, "r", encoding="utf-8") as f:
        for line in f:
            r = orjson.loads(line)
            have.append(r.get("domain", "Unknown"))
    have_cnt = Counter(have)

    print("== bioprotocol 전체 도메인 카운트(상위 30) ==")
    for d, c in all_cnt.most_common(30):
        print(f"{d:40s} {c}")

    print("\n== 이번에 methods 추출된 도메인 카운트(상위 30) ==")
    for d, c in have_cnt.most_common(30):
        print(f"{d:40s} {c}")

    # eligibility: bioprotocol 기준 count>=30
    elig = {d for d, c in all_cnt.items() if c >= 30}
    print(f"\n[INFO] eligible domains (>=30 in bioprotocol): {len(elig)}개")
    hit = [(d, have_cnt[d]) for d in sorted(elig, key=lambda x: -have_cnt[x])]
    print("eligible 교집합(현재 methods 보유 수):")
    for d, c in hit:
        print(f"{d:40s} {c}")


if __name__ == "__main__":
    main()
