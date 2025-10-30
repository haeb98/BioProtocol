# scripts/06_collect_original_articles.py
import csv
import pathlib
import re
import sys
import time

import requests
from bs4 import BeautifulSoup
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm import tqdm

ROOT = pathlib.Path("../scripts")
IN_TXT = ROOT / "data/gold/bio_protocol_urls.txt"
OUT_CSV = ROOT / "data/gold/bio_protocol_original_articles.csv"
OUT_URLS = ROOT / "data/gold/original_articles_urls.txt"
PDF_DIR = ROOT / "data/gold/gold_article_pdf"

HEADERS_HTML = {
    "User-Agent": "Mozilla/5.0 (compatible; biopro-rag/0.1)",
    "Accept": "text/html,application/xhtml+xml"
}
HEADERS_PDF = {
    # 일부 DOI는 이 헤더로 PDF 직링크로 리다이렉트됨
    "User-Agent": "Mozilla/5.0 (compatible; biopro-rag/0.1)",
    "Accept": "application/pdf"
}

TIMEOUT = 30
SLEEP_BETWEEN = 0.3


def proto_id_from_bioprotocol_doi(doi_url: str) -> str:
    # https://doi.org/10.21769/BioProtoc.5130 -> Bio-protocol-5130
    m = re.search(r"BioProtoc\.(\d+)$", doi_url)
    return f"Bio-protocol-{m.group(1)}" if m else ""


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, max=6))
def resolve_doi(doi_url: str) -> str:
    r = requests.get(doi_url, headers=HEADERS_HTML, allow_redirects=True, timeout=TIMEOUT)
    r.raise_for_status()
    return r.url  # 최종 도착 URL (bioprotocol 페이지)


def extract_original_link(html: str) -> str:
    """
    Bioprotocol 페이지에서 'Original Research Article' 박스의 링크를 파싱.
    견고하게 여러 전략을 시도.
    """
    soup = BeautifulSoup(html, "lxml")

    # 1) 'Original Research Article' 텍스트 인근의 a[href]
    box = None
    for tag in soup.find_all(text=re.compile(r"Original Research Article", re.I)):
        box = tag.parent
        break
    if box:
        a = box.find_next("a", href=True)
        if a and a["href"]:
            return a["href"].strip()

    # 2) 페이지 하단 블루 박스(사이트에 따라 class가 다름): 'article_show' / 'protocol_tag' 등
    #    a[href^="https://doi.org/"] 우선 수집
    for a in soup.select('a[href^="https://doi.org/"], a[href^="http://doi.org/"]'):
        # 저널명이나 "article_title" 클래스를 가진 앵커가 주로 해당
        return a["href"].strip()

    # 3) 안전 장치: 'a' 중 DOI 패턴을 모두 스캔
    for a in soup.find_all("a", href=True):
        if re.search(r"doi\.org/\S+", a["href"]):
            return a["href"].strip()

    return ""


@retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, max=6))
def fetch_bioprotocol_page(url: str) -> str:
    r = requests.get(url, headers=HEADERS_HTML, timeout=TIMEOUT)
    r.raise_for_status()
    return r.text


def try_download_pdf_from_doi(original_doi_url: str, out_path: pathlib.Path) -> (bool, str):
    """
    DOI에 Accept: application/pdf로 요청하여 바로 PDF가 열리는 경우만 저장.
    (출판사에 따라 HTML 랜딩/로그인 요구 시 False 반환)
    """
    try:
        with requests.get(original_doi_url, headers=HEADERS_PDF, allow_redirects=True,
                          timeout=TIMEOUT, stream=True) as r:
            ctype = r.headers.get("Content-Type", "")
            if "application/pdf" in ctype.lower():
                out_path.parent.mkdir(parents=True, exist_ok=True)
                with open(out_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                return True, "downloaded_via_accept_pdf"
            # 일부는 최종 URL이 pdf로 끝남(.pdf)
            final = r.url or ""
            if re.search(r"\.pdf($|\?)", final, re.I):
                # 직접 재요청(일반 GET)로 저장
                pdf = requests.get(final, headers=HEADERS_HTML, timeout=TIMEOUT)
                pdf.raise_for_status()
                if "application/pdf" in pdf.headers.get("Content-Type", "").lower():
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    out_path.write_bytes(pdf.content)
                    return True, "downloaded_via_direct_pdf_url"
    except requests.RequestException as e:
        return False, f"pdf_error:{e}"
    return False, "no_direct_pdf"


def main(download_pdf=False):
    assert IN_TXT.exists(), f"not found: {IN_TXT}"
    rows_out = []
    urls_out = []

    with open(IN_TXT, "r", encoding="utf-8") as fr:
        doi_urls = [line.strip() for line in fr if line.strip()]

    for doi in tqdm(doi_urls):
        proto_id = proto_id_from_bioprotocol_doi(doi)
        status, notes = "ok", ""
        biopro_page = ""
        original_url = ""
        saved_pdf = ""

        try:
            biopro_page = resolve_doi(doi)
            html = fetch_bioprotocol_page(biopro_page)
            original_url = extract_original_link(html)
            if not original_url:
                status, notes = "no_original_link", "could_not_find_original_article_anchor"
            else:
                urls_out.append(original_url)

            if download_pdf and original_url:
                n = re.search(r"(\d+)$", proto_id)  # Bio-protocol-<n> → n
                if n:
                    pdf_path = PDF_DIR / f"Bio-article-{n.group(1)}.pdf"
                    ok, why = try_download_pdf_from_doi(original_url, pdf_path)
                    if ok:
                        saved_pdf = str(pdf_path)
                    else:
                        notes = (notes + ";" + why).strip(";")

        except requests.HTTPError as e:
            status, notes = "http_error", str(e)
        except Exception as e:
            status, notes = "error", str(e)

        rows_out.append({
            "protocol_id": proto_id,
            "bioprotocol_doi": doi,
            "bioprotocol_page": biopro_page,
            "original_article_url": original_url,
            "pdf_path": saved_pdf,
            "status": status,
            "notes": notes
        })
        time.sleep(SLEEP_BETWEEN)

    # CSV
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as fw:
        w = csv.DictWriter(fw, fieldnames=list(rows_out[0].keys()))
        w.writeheader()
        w.writerows(rows_out)

    # URL list
    with open(OUT_URLS, "w", encoding="utf-8") as fu:
        fu.write("\n".join([u for u in urls_out if u]))

    print(f"[OK] saved: {OUT_CSV}")
    print(f"[OK] urls:  {OUT_URLS}")
    if download_pdf:
        print(f"[INFO] PDFs (if any) in: {PDF_DIR}")


if __name__ == "__main__":
    # CLI: python scripts/06_collect_original_articles.py [--pdf]
    dl = len(sys.argv) > 1 and sys.argv[1] in ("--pdf", "--download-pdf")
    main(download_pdf=dl)
