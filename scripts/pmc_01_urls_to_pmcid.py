import csv
import os
import pathlib
import re
import requests
import time
from urllib.parse import urlparse

from tqdm import tqdm

ROOT = pathlib.Path(".")
IN = ROOT / "data/gold/bio_protocol_original_articles.csv"  # 너의 기존 CSV
OUT = ROOT / "data/gold/pmc_map_from_urls.csv"
OUT.parent.mkdir(parents=True, exist_ok=True)

NCBI_API_KEY = os.environ.get("NCBI_API_KEY", "")
IDCONV = "https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/"

RE_PMCID = re.compile(r'/pmc/articles/(PMC\d+)', re.I)
RE_PMID = re.compile(r'/pubmed/(\d+)|pubmed\.ncbi\.nlm\.nih\.gov/(\d+)', re.I)


def is_pubmedish(u: str) -> bool:
    if not u: return False
    h = urlparse(u).netloc.lower()
    return ("ncbi.nlm.nih.gov" in h) or ("pubmed.ncbi.nlm.nih.gov" in h)


def extract_pmcid(url: str) -> str:
    m = RE_PMCID.search(url)
    return m.group(1) if m else ""


def extract_pmid(url: str) -> str:
    m = RE_PMID.search(url)
    if not m: return ""
    g1, g2 = m.groups()
    return (g1 or g2 or "").strip()


def idconv_pmid_to_pmcid(pmid: str) -> str:
    params = {"ids": pmid, "format": "json"}
    if NCBI_API_KEY: params["api_key"] = NCBI_API_KEY
    r = requests.get(IDCONV, params=params, timeout=30)
    r.raise_for_status()
    j = r.json()
    recs = j.get("records", [])
    if not recs: return ""
    pmcid = recs[0].get("pmcid", "")
    return pmcid or ""


def main():
    assert IN.exists(), f"not found: {IN}"
    rows = list(csv.DictReader(open(IN, "r", encoding="utf-8")))
    out = open(OUT, "w", newline="", encoding="utf-8")
    w = csv.writer(out)
    # 입력 CSV의 열 이름이 다를 수 있으니 안전하게 가져옴
    w.writerow(["id", "title", "domain", "original_article_url", "pmid", "pmcid", "status"])

    for r in tqdm(rows, desc="url-scan"):
        pid = (r.get("id") or r.get("protocol_id") or "").strip()
        title = (r.get("title") or "").strip()
        dom = (r.get("domain") or r.get("primary_domain") or "").strip()
        url = (r.get("original_article_url") or r.get("original_url") or "").strip()
        if not url or not is_pubmedish(url):
            w.writerow([pid, title, dom, url, "", "", "skip_non_pubmed"])
            continue

        pmcid = extract_pmcid(url)
        pmid = extract_pmid(url)
        status = ""

        if pmcid:
            status = "pmcid_from_url"
        elif pmid:
            # pmid → pmcid 변환 시도
            try:
                pmcid = idconv_pmid_to_pmcid(pmid)
                status = "pmcid_from_idconv" if pmcid else "no_pmc_for_pmid"
                time.sleep(0.35)
            except requests.HTTPError as e:
                status = f"idconv_error:{e}"
        else:
            status = "no_pmid_or_pmcid_in_url"

        w.writerow([pid, title, dom, url, pmid, pmcid, status])

    out.close()
    print(f"[OK] saved: {OUT}")


if __name__ == "__main__":
    main()
