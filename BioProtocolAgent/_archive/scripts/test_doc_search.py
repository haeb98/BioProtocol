# scripts/test_doc_search.py
import json
import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
sys.path.append(str(PROJECT_ROOT))

from src.tools.doc_search import doc_search_fulltext

GOLD_PAIRS_PATH = Path("data/gold_pairs_testset_v2.jsonl")


def get_any_pmcid_and_pid():
    """
    gold_pairs_testset_v2.jsonl에서 protocol_id, pmcid 하나만 가져오는 헬퍼.
    실제 파일 구조에 맞게 key 이름만 맞춰주면 됨.
    """
    with GOLD_PAIRS_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)

            # 🔴 실제 구조에 맞게 조정 필요!
            # 예시 A)
            #   rec["protocol_id"], rec["pmcid"]
            # 예시 B)
            #   rec["bio"]["id"], rec["article"]["pmcid"]
            protocol_id = rec.get("protocol_id") or rec["bio"]["id"]
            pmcid = rec.get("pmcid") or rec["article"]["pmcid"]

            return protocol_id, pmcid

    raise RuntimeError("No records found in gold_pairs_testset_v2.jsonl")


def main():
    pid, pmcid = get_any_pmcid_and_pid()
    print(f"Using protocol_id={pid}, pmcid={pmcid}")

    query = "changing the medium or passaging the cells"
    print(f"\n[Query] {query}\n")

    hits = doc_search_fulltext(pmcid, query, top_k=5)

    for i, h in enumerate(hits, 1):
        print(f"### Hit {i}")
        print("Section :", h["section"])
        print("Score   :", h["score"])
        print("Text    :", h["text"][:400], "...")
        print()


if __name__ == "__main__":
    main()
