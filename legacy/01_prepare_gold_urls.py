# scripts/01_prepare_gold_urls.py
import csv
import json
import pathlib
import re

ROOT = pathlib.Path("../scripts")
IN = ROOT / "data/gold/bio_protocol.json"
OUT_URLS = ROOT / "data/gold/bio_protocol_urls.txt"
OUT_IDX = ROOT / "data/gold/bio_protocol_index.csv"
OUT_MAP = ROOT / "data/gold/gold_id_map.csv"


def get_num(pid: str) -> int:
    # 'Bio-protocol-123' -> 123
    m = re.search(r'(\d+)$', pid or "")
    return int(m.group(1)) if m else -1


def main():
    data = json.loads(IN.read_text())
    # 정렬(번호순) 보장
    data.sort(key=lambda r: get_num(r.get("id", "")))
    # urls
    with open(OUT_URLS, "w", encoding="utf-8") as fw:
        for r in data:
            u = r.get("url", "")
            if u: fw.write(u.strip() + "\n")
    # index
    with open(OUT_IDX, "w", newline="", encoding="utf-8") as fw:
        w = csv.writer(fw, delimiter="\t")
        w.writerow(["id", "url", "title", "keywords"])
        for r in data:
            w.writerow([r.get("id", ""), r.get("url", ""), r.get("title", ""), (r.get("keywords") or "").strip()])
    # id map
    with open(OUT_MAP, "w", newline="", encoding="utf-8") as fw:
        w = csv.writer(fw)
        w.writerow(["protocol_id", "article_expected_filename"])
        for r in data:
            n = get_num(r.get("id", ""))
            if n >= 0:
                w.writerow([f"Bio-protocol-{n}", f"Bio-article-{n}.pdf"])
    print("[OK]", OUT_URLS, OUT_IDX, OUT_MAP)


if __name__ == "__main__":
    main()
