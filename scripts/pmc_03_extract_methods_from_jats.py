import csv
import orjson
import pathlib
import re

from lxml import etree
from tqdm import tqdm

ROOT = pathlib.Path(".")
MAP = ROOT / "data/gold/pmc_map_from_urls.csv"
JATS = ROOT / "data/gold/pmc_jats"
BIO = ROOT / "data/raw/bio_protocol.json"  # ← 추가: 바이오프로토콜에서 domain/title 보강
OUT = ROOT / "data/gold/gold_articles_methods_pmc.jsonl"
OUT.parent.mkdir(parents=True, exist_ok=True)

TITLE_PAT = re.compile(r"""
^\s*(
  materials?\s*(and|&)\s*methods |
  methods |
  experimental\s+procedures |
  methodology
)\s*$""", re.I | re.X)


def text_of(node):
    return " ".join("".join(node.itertext()).split())


def find_methods(root):
    parts = []
    for sec in root.findall(".//sec"):
        te = sec.find("./title")
        title = (te.text or "").strip() if te is not None else ""
        if TITLE_PAT.match(title):
            txt = text_of(sec)
            if len(txt) > 100:
                parts.append(txt)
    if not parts:
        for s in root.findall(".//subsec"):
            te = s.find("./title")
            title = (te.text or "").strip() if te is not None else ""
            if TITLE_PAT.match(title):
                txt = text_of(s)
                if len(txt) > 100:
                    parts.append(txt)
    return "\n\n".join(parts)


def main():
    # 1) pmc_map: id ↔ pmcid (CSV)
    map_rows = list(csv.DictReader(open(MAP, "r", encoding="utf-8")))
    pmc_idx = {r["pmcid"]: r for r in map_rows if r.get("pmcid")}

    # 2) bio_protocol.json 로드 → id 기반 보강 인덱스
    bio = orjson.loads(BIO.read_bytes())
    # id는 문자열로 정규화
    bio_idx = {str(x.get("id")): x for x in bio}

    n = 0
    with open(OUT, "w", encoding="utf-8") as fout:
        for xmlp in tqdm(sorted(JATS.glob("PMC*.xml")), desc="extract-methods"):
            pmcid = xmlp.stem
            meta = pmc_idx.get(pmcid)
            if not meta:
                continue

            proto_id = meta.get("id", "")
            bio_rec = bio_idx.get(str(proto_id), {})

            # domain/title 보강 우선순위: bio_protocol.json > map.csv
            domain = ((bio_rec.get("classification") or {}).get("primary_domain")
                      or meta.get("domain") or "Unknown")
            title = (bio_rec.get("title") or meta.get("title") or "")

            try:
                tree = etree.parse(str(xmlp))
            except Exception:
                continue
            methods = find_methods(tree.getroot())
            if not methods or len(methods) < 200:
                continue

            rec = {
                "article_id": pmcid,
                "protocol_id": proto_id,  # Bio-protocol-*
                "domain": domain,
                "title": title,
                "pmid": meta.get("pmid", ""),
                "pmcid": pmcid,
                "methods_text": methods,
                "xml_path": str(xmlp)  # 원문 경로 보관
            }
            fout.write(orjson.dumps(rec).decode() + "\n");
            n += 1
    print(f"[OK] {OUT} (rows={n})")


if __name__ == "__main__":
    main()
