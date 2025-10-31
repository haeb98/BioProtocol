# scripts/09_domain_distribution.py
# 목적:
# - 세 원본 JSON에서 classification.primary_domain을 스트리밍 집계
# - 파일별 분포, 전체 합계 분포 저장
# 출력:
#   data/splits/domain_counts_by_source.csv
#   data/splits/domain_counts_overall.csv
#   data/splits/domain_summary.json

import csv
import pathlib
from collections import Counter, defaultdict

import ijson
import orjson

ROOT = pathlib.Path("../scripts")
RAW = ROOT / "data" / "raw"
OUTDIR = ROOT / "data" / "splits"
OUTDIR.mkdir(parents=True, exist_ok=True)

# 파일명 호환(언더스코어/하이픈 혼용 대비)
CANDIDATES = [
    ("bio_protocol", ["bio_protocol.json"]),
    ("protocol_io", ["protocols_io.json", "protocol-io.json"]),
    ("protocol_exchange", ["protocol_exchange.json", "protocol-exchange.json"]),
]


def find_file(name_candidates):
    for nm in name_candidates:
        p = RAW / nm
        if p.exists():
            return p
    return None


def stream_array(path: pathlib.Path):
    # 대형 리스트 JSON을 스트리밍으로 순회
    with open(path, "rb") as f:
        for obj in ijson.items(f, "item"):
            yield obj


def get_id(rec):
    return str(rec.get("id") or rec.get("_id") or rec.get("uid") or "")


def get_domain(rec):
    c = rec.get("classification") or {}
    d = c.get("primary_domain")
    if not d or not str(d).strip():
        return "Unknown"
    return str(d).strip()


def main():
    per_source = defaultdict(Counter)  # {source: Counter({domain: count})}
    overall = Counter()
    empty_domain_examples = defaultdict(int)

    sources = []
    for key, names in CANDIDATES:
        p = find_file(names)
        if not p:
            print(f"[WARN] missing file for {key}: tried {names}")
            continue
        sources.append((key, p))

    if not sources:
        raise SystemExit("원본 JSON을 찾지 못했습니다. data/raw/ 경로를 확인하세요.")

    for src_key, path in sources:
        n_total = 0
        for rec in stream_array(path):
            n_total += 1
            dom = get_domain(rec)
            per_source[src_key][dom] += 1
            overall[dom] += 1
            if dom == "Unknown":
                empty_domain_examples[src_key] += 1
        print(f"[OK] {src_key}: {n_total} records")

    # 파일별 CSV
    by_source_csv = OUTDIR / "domain_counts_by_source.csv"
    with open(by_source_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["source", "domain", "count"])
        for src_key, counter in per_source.items():
            for dom, cnt in sorted(counter.items(), key=lambda x: (-x[1], x[0])):
                w.writerow([src_key, dom, cnt])

    # 전체 합계 CSV
    overall_csv = OUTDIR / "domain_counts_overall.csv"
    with open(overall_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["domain", "count"])
        for dom, cnt in sorted(overall.items(), key=lambda x: (-x[1], x[0])):
            w.writerow([dom, cnt])

    # 요약 JSON
    summary = {
        "files_processed": [str(p) for _, p in sources],
        "per_source": {k: dict(v) for k, v in per_source.items()},
        "overall": dict(overall),
        "unknown_counts_by_source": dict(empty_domain_examples),
        "notes": "counts are raw (no dedup across files)."
    }
    (OUTDIR / "domain_summary.json").write_text(
        orjson.dumps(summary, option=orjson.OPT_INDENT_2).decode(),
        encoding="utf-8"
    )

    print(f"[OK] saved: {by_source_csv}")
    print(f"[OK] saved: {overall_csv}")
    print(f"[OK] saved: {OUTDIR / 'domain_summary.json'}")
    print("참고: raw 합계이며, 파일 간 중복 id 제거는 하지 않았습니다.")
    print("    (원하면 dedup 버전도 만들어 드릴 수 있어요.)")


if __name__ == "__main__":
    main()
