# 입력: data/splits/test_ids.txt, data/raw/bio_protocol.json
# 출력:
#   data/gold/test_original_urls.csv  (id, title, doi_url)
#   data/gold/test_original_urls.txt  (원문 URL만 라인별)
import csv
import orjson
import pathlib

ROOT = pathlib.Path(".")
TEST_IDS = (ROOT / "data/splits/test_ids.txt").read_text().splitlines()
BIO = ROOT / "data/raw/bio_protocol.json"
OUT_CSV = ROOT / "data/gold/test_original_urls.csv"
OUT_TXT = ROOT / "data/gold/test_original_urls.txt"
(OUT_CSV.parent).mkdir(parents=True, exist_ok=True)


def get_doi_url(rec: dict) -> str:
    for k in ["url", "doi_url", "doi"]:
        v = rec.get(k)
        if isinstance(v, str) and v.strip(): return v.strip()
    links = rec.get("links") or {}
    if isinstance(links, dict):
        for v in links.values():
            if isinstance(v, str) and "doi.org" in v: return v.strip()
    import re
    pat = re.compile(r"https?://doi\.org/\S+", re.I)
    for k in ["protocol", "input", "abstract", "description"]:
        s = rec.get(k)
        if isinstance(s, str):
            m = pat.search(s)
            if m: return m.group(0)
    return ""


def main():
    data = orjson.loads(BIO.read_bytes())
    idx = {str(x.get("id")): x for x in data}
    rows = [];
    urls = []
    for tid in TEST_IDS:
        rec = idx.get(tid)
        if not rec: continue
        url = get_doi_url(rec)
        rows.append({"id": tid, "title": (rec.get("title") or "").strip(), "doi_url": url})
        if url: urls.append(url)
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id", "title", "doi_url"])
        w.writeheader();
        w.writerows(rows)
    (OUT_TXT).write_text("\n".join(urls), encoding="utf-8")
    print("[OK]", OUT_CSV);
    print("[OK]", OUT_TXT)


if __name__ == "__main__": main()
