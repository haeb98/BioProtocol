# scripts/10b_make_splits_by_domain_threshold.py
# 정책:
# - 세 JSON(bio_protocol, protocol_io, protocol_exchange)에서
#   전체 도메인 카운트를 합산하여 count >= THRESH(기본 100)인 도메인만 사용
# - 각 '사용 도메인'에 대해 source별로 랜덤 N(기본 10)개를 테스트로 샘플
# - 나머지는 RAG 코퍼스
# - 도메인 < THRESH 는 아예 제외
#
# 출력:
#   data/splits/eligible_domains.txt
#   data/splits/test_ids.txt
#   data/splits/corpus_ids.txt
#   data/splits/test_ids_by_source.csv
#   data/splits/corpus_ids_by_source.csv
#   data/splits/split_summary.json

import csv
import os
import pathlib
import random
from collections import Counter, defaultdict

import ijson
import orjson
from tqdm import tqdm

ROOT = pathlib.Path(".")
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "splits"
OUT.mkdir(parents=True, exist_ok=True)

# 파일명 후보(언더스코어/하이픈 모두 허용)
CANDS = [
    ("bio_protocol", ["bio_protocol.json"]),
    ("protocol_io", ["protocol_io.json", "protocol-io.json"]),
    ("protocol_exchange", ["protocol_exchange.json", "protocol-exchange.json"]),
]

THRESH = int(os.environ.get("DOMAIN_COUNT_THRESHOLD", "100"))  # 도메인 최소 개수
N_TEST = int(os.environ.get("N_TEST_PER_DOMAIN_PER_SOURCE", "10"))
SEED = int(os.environ.get("SEED", "42"))
random.seed(SEED)


def find_file(names):
    for nm in names:
        p = RAW / nm
        if p.exists(): return p
    return None


def stream_array(path: pathlib.Path):
    with open(path, "rb") as f:
        for obj in ijson.items(f, "item"):
            yield obj


def rec_id(r):
    return str(r.get("id") or r.get("_id") or r.get("uid") or "")


def rec_domain(r):
    c = r.get("classification") or {}
    d = c.get("primary_domain")
    if not d or not str(d).strip(): return "Unknown"
    return str(d).strip()


def main():
    # 0) 파일 존재 확인
    sources = []
    for key, names in CANDS:
        p = find_file(names)
        if not p:
            print(f"[WARN] missing for {key}: tried {names}")
            continue
        sources.append((key, p))
    if not sources:
        raise SystemExit("원본 JSON을 찾지 못했습니다. data/raw 확인 요망.")

    # 1) 1차 패스: 전체 도메인 카운트
    overall = Counter()
    total_records = 0
    for key, path in sources:
        for r in tqdm(stream_array(path), desc=f"count {key}"):
            total_records += 1
            overall[rec_domain(r)] += 1

    eligible = sorted([d for d, c in overall.items() if c >= THRESH])
    (OUT / "eligible_domains.txt").write_text("\n".join(eligible), encoding="utf-8")
    print(f"[OK] eligible domains (>= {THRESH}): {len(eligible)} 도메인")

    # 2) 2차 패스: eligible 도메인에 한해 source별 id 버킷 수집
    #    buckets[source][domain] = list of ids
    buckets = defaultdict(lambda: defaultdict(list))
    seen_ids = set()  # 전 소스 통틀어 id 중복 방지(일반적으로 겹치지 않겠지만 안전하게)
    per_source_total = Counter()
    for key, path in sources:
        for r in tqdm(stream_array(path), desc=f"collect {key}"):
            _id = rec_id(r)
            if not _id or _id in seen_ids:
                continue
            d = rec_domain(r)
            if d in eligible:
                buckets[key][d].append(_id)
                per_source_total[key] += 1
                seen_ids.add(_id)

    # 3) 도메인×소스별 랜덤 샘플 → 테스트 / 코퍼스 분리
    test_rows = []  # list of (source, domain, id)
    corpus_rows = []

    TEST_ONLY_SOURCES = {"bio_protocol"}

    # for d in eligible:
    #     for key, _ in sources:
    #         ids = buckets[key].get(d, [])
    #         if not ids:
    #             continue
    #         random.shuffle(ids)
    #         k = min(N_TEST, len(ids))
    #         test_ids = ids[:k]
    #         rest_ids = ids[k:]
    #         test_rows.extend([(key, d, i) for i in test_ids])
    #         corpus_rows.extend([(key, d, i) for i in rest_ids])

    for d in eligible:
        for key, _ in sources:
            ids = buckets[key].get(d, [])
            if not ids:
                continue
            random.shuffle(ids)
            k = min(N_TEST, len(ids)) if key in TEST_ONLY_SOURCES else 0
            test_ids = ids[:k]
            rest_ids = ids[k:]
            test_rows.extend([(key, d, i) for i in test_ids])
            corpus_rows.extend([(key, d, i) for i in rest_ids])

    # 4) 파일로 저장
    # 4-1) by_source CSV
    test_csv = OUT / "test_ids_by_source.csv"
    corpus_csv = OUT / "corpus_ids_by_source.csv"
    with open(test_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f);
        w.writerow(["source", "domain", "id"]);
        w.writerows(test_rows)
    with open(corpus_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f);
        w.writerow(["source", "domain", "id"]);
        w.writerows(corpus_rows)

    # 4-2) 단일 리스트(txt)
    (OUT / "test_ids.txt").write_text("\n".join([r[2] for r in test_rows]), encoding="utf-8")
    (OUT / "corpus_ids.txt").write_text("\n".join([r[2] for r in corpus_rows]), encoding="utf-8")

    # 5) 요약 저장
    summary = {
        "params": {
            "threshold": THRESH,
            "n_test_per_domain_per_source": N_TEST,
            "seed": SEED
        },
        "stats": {
            "total_records_scanned": total_records,
            "eligible_domains_count": len(eligible),
            "eligible_domains": eligible,
            "per_source_total_in_eligible": dict(per_source_total),
            "n_test": len(test_rows),
            "n_corpus": len(corpus_rows)
        }
    }
    (OUT / "split_summary.json").write_text(
        orjson.dumps(summary, option=orjson.OPT_INDENT_2).decode(),
        encoding="utf-8"
    )

    print(f"[OK] saved: {test_csv}")
    print(f"[OK] saved: {corpus_csv}")
    print(f"[OK] saved: {OUT / 'test_ids.txt'}")
    print(f"[OK] saved: {OUT / 'corpus_ids.txt'}")
    print(f"[OK] saved: {OUT / 'eligible_domains.txt'}")
    print(f"[OK] saved: {OUT / 'split_summary.json'}")
    print("※ 임계치 미만 도메인 자료는 테스트/코퍼스 어디에도 포함되지 않습니다.")


if __name__ == "__main__":
    main()
