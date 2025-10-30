# 입력: data/gold/gold_article_pdf/Bio-article-*.pdf
# 출력: data/gold/gold_articles.jsonl (article_id, protocol_id, text, meta)
import pathlib

import fitz
import orjson
from tqdm import tqdm

ROOT = pathlib.Path("../scripts")
IN = ROOT / "data/gold/gold_article_pdf"
OUT = ROOT / "data/gold/gold_articles.jsonl"
OUT.parent.mkdir(parents=True, exist_ok=True)


def pdf_to_text(p):
    doc = fitz.open(p);
    meta = doc.metadata or {}
    return "\n".join([pg.get_text("text") for pg in doc]), meta


def main():
    files = sorted(IN.glob("Bio-article-*.pdf"))
    n = 0
    with open(OUT, "w", encoding="utf-8") as f:
        for pdf in tqdm(files):
            txt, meta = pdf_to_text(pdf)
            pid = pdf.stem.replace("article", "protocol")
            rec = {"article_id": pdf.stem, "protocol_id": pid, "filename": pdf.name, "text": txt, "meta": meta}
            f.write(orjson.dumps(rec).decode() + "\n");
            n += 1
    print("[OK]", OUT, "(rows=", n, ")")


if __name__ == "__main__": main()
