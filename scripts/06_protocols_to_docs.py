# scripts/11_protocols_to_docs_filtered.py
"""
입력:
  - data/splits/test_ids.txt, data/splits/corpus_ids.txt
  - data/raw/*.json (bio_protocol / protocol_io / protocol_exchange)
출력:
  - data/processed/protocol_corpus_docs.jsonl
  - data/processed/protocol_test_docs.jsonl

설명:
  - ID 리스트 기반 필터링(스플릿 반영)
  - 프로토콜 1건 = 1문서 원칙
  - 메모리 절약을 위해 ijson 스트리밍
"""
import ijson
import orjson
import pathlib

ROOT = pathlib.Path(".")
RAW = ROOT / "data" / "raw"
SPL = ROOT / "data" / "splits"
OUT = ROOT / "data" / "processed"
OUT.mkdir(parents=True, exist_ok=True)

IN_FILES = [
    RAW / "bio_protocol.json",
    RAW / "protocol_io.json", RAW / "protocol-io.json",
    RAW / "protocol_exchange.json", RAW / "protocol-exchange.json"
]

TEST_IDS = set((SPL / "test_ids.txt").read_text().splitlines())
CORPUS_IDS = set((SPL / "corpus_ids.txt").read_text().splitlines())

OUT_CORP = OUT / "protocol_corpus_docs.jsonl"
OUT_TEST = OUT / "protocol_test_docs.jsonl"


def exists(p: pathlib.Path): return p.exists() and p.is_file()


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


def build_text(r):
    t = r.get("title") or ""
    ab = r.get("abstract") or ""
    inputs = r.get("input") or ""
    proto = r.get("protocol") or ""
    hier = flatten_hier(r.get("hierarchical_protocol") or {})
    parts = [p for p in [t, "# Abstract", ab, "# Materials & Inputs", inputs, "# Protocol", proto, "# Outline", hier] if
             p and str(p).strip()]
    return "\n\n".join(parts)


def stream_array(path: pathlib.Path):
    with open(path, "rb") as f:
        for obj in ijson.items(f, "item"):
            yield obj


def rec_id(r):  return str(r.get("id") or r.get("_id") or r.get("uid") or "")


def dom(r):     return (r.get("classification") or {}).get("primary_domain") or "Unknown"


def main():
    fc = open(OUT_CORP, "w", encoding="utf-8")
    ft = open(OUT_TEST, "w", encoding="utf-8")
    seen = set();
    n_c = n_t = 0

    targets = TEST_IDS | CORPUS_IDS
    files = [p for p in IN_FILES if exists(p)]
    assert files, "data/raw에 원본 JSON이 없습니다."

    for p in files:
        for r in stream_array(p):
            _id = rec_id(r)
            if not _id or _id in seen:
                continue
            if _id not in targets:
                continue
            seen.add(_id)
            rec = {
                "doc_id": _id,
                "title": r.get("title") or "",
                "text": build_text(r),
                "meta": {
                    "source_file": p.name,
                    "primary_domain": dom(r),
                }
            }
            line = orjson.dumps(rec).decode() + "\n"
            if _id in TEST_IDS:
                ft.write(line);
                n_t += 1
            else:
                fc.write(line);
                n_c += 1

    fc.close();
    ft.close()
    print(f"[OK] corpus docs: {OUT_CORP} (n={n_c})")
    print(f"[OK] test docs:   {OUT_TEST} (n={n_t})")


if __name__ == "__main__":
    main()
