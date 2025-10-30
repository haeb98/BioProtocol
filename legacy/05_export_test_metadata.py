# scripts/10c_export_test_protocol_metadata.py
# 목적:
# - data/splits/test_ids.txt에 들어있는 id들만 세 원본 JSON에서 찾아
#   source, id, doi_url, title, keywords 를 추출해 CSV/JSONL로 저장
# - 중복 id 등장 시 source 우선순위(bio_protocol > protocol_io > protocol_exchange)로 1개만 채택
#
# 출력:
#   data/splits/test_protocol_metadata.csv
#   data/splits/test_protocol_metadata.jsonl
#   data/splits/test_ids_missing.txt      (원본에서 못 찾은 id 목록)

import csv
import pathlib

import ijson
import orjson
from tqdm import tqdm

ROOT = pathlib.Path("../scripts")
RAW = ROOT / "data" / "raw"
SPLITS = ROOT / "data" / "splits"

TEST_IDS_PATH = SPLITS / "test_ids.txt"
OUT_CSV = SPLITS / "test_protocol_metadata.csv"
OUT_JSON = SPLITS / "test_protocol_metadata.jsonl"
OUT_MISS = SPLITS / "test_ids_missing.txt"

# 파일명 후보(언더스코어/하이픈 모두 허용)
SOURCES = [
    ("bio_protocol", ["bio_protocol.json"]),  # 1순위
    ("protocol_io", ["protocol_io.json"]),  # 2순위
    ("protocol_exchange", ["protocol_exchange.json"]),  # 3순위
]


def find_file(names):
    for nm in names:
        p = RAW / nm
        if p.exists():
            return p
    return None


def stream_array(path: pathlib.Path):
    with open(path, "rb") as f:
        for obj in ijson.items(f, "item"):
            yield obj


def rec_id(r):
    return str(r.get("id") or r.get("_id") or r.get("uid") or "")


def norm_keywords(v):
    # 문자열/리스트 모두 수용 → 세미콜론으로 합침
    if v is None:
        return ""
    if isinstance(v, str):
        # 줄바꿈/중복 스페이스 정리
        s = " ".join(v.replace("\n", " ").split())
        return s
    if isinstance(v, (list, tuple)):
        items = []
        for x in v:
            if x is None:
                continue
            xs = str(x).strip()
            if xs:
                items.append(xs)
        return "; ".join(items)
    # 예외: dict 등은 문자열화
    return str(v)


def get_doi_url(rec):
    """
    가능한 필드에서 DOI/URL을 유연하게 추출.
    기본적으로 bio_protocol.json은 'url'이 DOI URL인 경우가 많음.
    기타 케이스도 최대한 대응.
    """
    # 1) 흔한 케이스
    for k in ["url", "doi_url", "doi"]:
        v = rec.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    # 2) links.* 안에 있는 경우
    links = rec.get("links") or rec.get("link") or {}
    if isinstance(links, dict):
        for k, v in links.items():
            if isinstance(v, str) and "doi.org" in v:
                return v.strip()
    # 3) 텍스트 안에서 doi.org 패턴 찾기 (last resort)
    import re
    pat = re.compile(r"https?://doi\.org/\S+", re.I)
    for k in ["protocol", "input", "abstract", "description"]:
        s = rec.get(k)
        if isinstance(s, str):
            m = pat.search(s)
            if m:
                return m.group(0)
    return ""


def main():
    assert TEST_IDS_PATH.exists(), f"not found: {TEST_IDS_PATH}"
    test_ids = [x.strip() for x in TEST_IDS_PATH.read_text().splitlines() if x.strip()]
    test_set = set(test_ids)
    picked = {}  # id -> row (source 우선순위대로 먼저 채택된 것 유지)

    # source 우선순위대로 순회
    for src_key, candidates in SOURCES:
        p = find_file(candidates)
        if not p:
            print(f"[WARN] missing file for {src_key}: tried {candidates}")
            continue
        for rec in tqdm(stream_array(p), desc=f"scan {src_key}"):
            _id = rec_id(rec)
            # 대상 id이고 아직 미채택이면 기록
            if _id in test_set and _id not in picked:
                row = {
                    "source": src_key,
                    "id": _id,
                    "doi_url": get_doi_url(rec),
                    "title": (rec.get("title") or "").strip(),
                    "keywords": norm_keywords(rec.get("keywords"))
                }
                picked[_id] = row

    # 누락 id 리포트
    missing = [i for i in test_ids if i not in picked]
    if missing:
        OUT_MISS.write_text("\n".join(missing), encoding="utf-8")
        print(f"[INFO] missing ids: {len(missing)} -> {OUT_MISS}")

    # 저장 (테스트 id의 원래 순서 유지)
    rows = [picked[i] for i in test_ids if i in picked]

    # CSV
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["source", "id", "doi_url", "title", "keywords"])
        w.writeheader()
        w.writerows(rows)

    # JSONL
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(orjson.dumps(r).decode() + "\n")

    print(f"[OK] saved: {OUT_CSV} (n={len(rows)})")
    print(f"[OK] saved: {OUT_JSON} (n={len(rows)})")


if __name__ == "__main__":
    main()
