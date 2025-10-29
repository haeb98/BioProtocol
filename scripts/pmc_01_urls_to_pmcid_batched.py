# scripts/pmc_01_urls_to_pmcid_batched.py
import csv
import os
import pathlib
import re
import requests
import shutil
import tempfile
import time
from urllib.parse import urlparse

from requests.adapters import HTTPAdapter
from tqdm import tqdm
from urllib3.util.retry import Retry

ROOT = pathlib.Path(".")
IN = ROOT / "data/gold/bio_protocol_original_articles.csv"  # 이미 있는 입력
OUT = ROOT / "data/gold/pmc_map_from_urls.csv"  # 출력 (재개 가능)
OUT.parent.mkdir(parents=True, exist_ok=True)

NCBI_API_KEY = os.environ.get("NCBI_API_KEY", "")
IDCONV = "https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/"

# 설정 (필요 시 환경변수로 조절)
BATCH = int(os.environ.get("IDCONV_BATCH", "150"))  # idconv는 최대 200까지OK
SLEEP = float(os.environ.get("IDCONV_SLEEP", "0.35"))
TIMEOUT = float(os.environ.get("HTTP_TIMEOUT", "60"))
RESUME = os.environ.get("RESUME", "0") == "1"

RE_PMCID = re.compile(r'/pmc/articles/(PMC\d+)', re.I)
RE_PMID = re.compile(r'/pubmed/(\d+)|pubmed\.ncbi\.nlm\.nih\.gov/(\d+)', re.I)


def make_session():
    sess = requests.Session()
    retry = Retry(
        total=6, read=6, connect=6,
        backoff_factor=1.0,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"]
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=10, pool_maxsize=10)
    sess.mount("https://", adapter)
    sess.mount("http://", adapter)
    return sess


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


def load_existing_map(path: pathlib.Path):
    """RESUME 용: 기존 OUT이 있으면 id→pmcid/pmid/status 맵으로 로드"""
    done = {}
    if not path.exists(): return done
    with open(path, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            done[r["id"]] = r  # 전체 행 보관
    return done


def idconv_lookup_batch(pmids: list[str], session: requests.Session) -> dict:
    """
    pmids를 배치로 idconv 조회 -> {pmid: pmcid or ""} 반환
    """
    out = {}
    for i in range(0, len(pmids), BATCH):
        chunk = pmids[i:i + BATCH]
        params = {"ids": ",".join(chunk), "format": "json"}
        if NCBI_API_KEY: params["api_key"] = NCBI_API_KEY
        try:
            r = session.get(IDCONV, params=params, timeout=TIMEOUT)
            r.raise_for_status()
            j = r.json()
            recs = j.get("records", [])
            for rec in recs:
                pmid = str(rec.get("pmid", "")).strip()
                pmcid = rec.get("pmcid", "") or ""
                if pmid:
                    out[pmid] = pmcid
        except requests.RequestException as e:
            # 배치 실패 시 각 pmid 개별 재시도로 보완
            for pmid in chunk:
                try:
                    rr = session.get(IDCONV, params={"ids": pmid, "format": "json",
                                                     **({"api_key": NCBI_API_KEY} if NCBI_API_KEY else {})},
                                     timeout=TIMEOUT)
                    rr.raise_for_status()
                    jj = rr.json();
                    recs2 = jj.get("records", [])
                    pmcid = (recs2[0].get("pmcid", "") if recs2 else "") or ""
                    out[pmid] = pmcid
                    time.sleep(SLEEP)
                except requests.RequestException:
                    out[pmid] = ""
        time.sleep(SLEEP)
    return out


def atomic_write_csv(path: pathlib.Path, fieldnames, rows: list[dict]):
    tmpfd, tmppath = tempfile.mkstemp(prefix="pmcmap_", suffix=".csv", dir=str(path.parent))
    os.close(tmpfd)
    with open(tmppath, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    shutil.move(tmppath, path)


def main():
    assert IN.exists(), f"not found: {IN}"
    sess = make_session()

    # 입력 로드
    in_rows = list(csv.DictReader(open(IN, "r", encoding="utf-8")))
    # 기존 결과(재개)
    existing = load_existing_map(OUT) if RESUME else {}

    # 1차 스캔: URL에서 pmcid/pmid 추출
    out_rows = []
    need_idconv_pmids = []
    for r in tqdm(in_rows, desc="scan-urls"):
        pid = (r.get("id") or r.get("protocol_id") or "").strip()
        title = (r.get("title") or "").strip()
        dom = (r.get("domain") or r.get("primary_domain") or "").strip()
        url = (r.get("original_article_url") or r.get("original_url") or "").strip()

        # RESUME: 이미 처리된 id면 그대로 사용
        if RESUME and pid in existing:
            out_rows.append(existing[pid])
            continue

        if not url or not is_pubmedish(url):
            out_rows.append({
                "id": pid, "title": title, "domain": dom, "original_article_url": url,
                "pmid": "", "pmcid": "", "status": "skip_non_pubmed"
            })
            continue

        pmcid = extract_pmcid(url)
        pmid = extract_pmid(url)
        status = ""
        if pmcid:
            status = "pmcid_from_url"
        elif pmid:
            status = "need_idconv"
            need_idconv_pmids.append(pmid)
        else:
            status = "no_pmid_or_pmcid_in_url"

        out_rows.append({
            "id": pid, "title": title, "domain": dom, "original_article_url": url,
            "pmid": pmid, "pmcid": pmcid, "status": status
        })

    # 2차: idconv로 pmid→pmcid 일괄 변환
    need_idconv_pmids = sorted(set([p for p in need_idconv_pmids if p]))
    pmid_to_pmcid = {}
    if need_idconv_pmids:
        pmid_to_pmcid = idconv_lookup_batch(need_idconv_pmids, sess)

    # 3차: 병합 & 저장
    fieldnames = ["id", "title", "domain", "original_article_url", "pmid", "pmcid", "status"]
    final_rows = []
    for r in out_rows:
        if r["status"] == "need_idconv":
            pmid = r["pmid"]
            pmcid = pmid_to_pmcid.get(pmid, "")
            r["pmcid"] = pmcid
            r["status"] = "pmcid_from_idconv" if pmcid else "no_pmc_for_pmid"
        final_rows.append(r)

    atomic_write_csv(OUT, fieldnames, final_rows)
    print(f"[OK] saved: {OUT} (rows={len(final_rows)})")
    # 간단 통계
    n_oa = sum(1 for r in final_rows if r["pmcid"])
    n_non = sum(1 for r in final_rows if not r["pmcid"] and r["status"].startswith("no_"))
    print(f"[INFO] PMCID found: {n_oa} | missing: {n_non}")


if __name__ == "__main__":
    main()
