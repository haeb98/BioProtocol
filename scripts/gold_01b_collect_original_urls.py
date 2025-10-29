# scripts/gold_01b_collect_original_urls_fast.py
# 목적: test_ids.txt + bio_protocol.json 기반으로
#       Bioprotocol 페이지에서 "원문 논문 URL"만 빠르게 수집 (PDF 다운로드 X)
# 특징: httpx + HTTP/2 + 멀티스레드, BeautifulSoup 없이 정규식만으로 DOI URL 탐색
#
# 출력:
#   data/gold/test_original_urls.csv (id,title,biopro_doi,original_article_url,status,notes)
#   data/gold/test_original_urls.txt (원문 URL 라인 리스트)

import csv
import orjson
import os
import pathlib
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

import httpx
from tqdm import tqdm

ROOT = pathlib.Path(".")
BIO = ROOT / "data/raw/bio_protocol.json"
TEST_IDS = ROOT / "data/splits/test_ids.txt"
OUT_DIR = ROOT / "data/gold"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV = OUT_DIR / "test_original_urls.csv"
OUT_TXT = OUT_DIR / "test_original_urls.txt"

# 병렬/타임아웃 파라미터 (필요시 환경변수로 조절)
MAX_WORKERS = int(os.environ.get("MAX_WORKERS", "16"))
TIMEOUT = float(os.environ.get("TIMEOUT", "20"))
HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; biopro-rag/0.2; +https://example.local)",
    "Accept": "text/html,application/xhtml+xml"
}

# 정규식
RE_DOI_LINK = re.compile(r'https?://doi\.org/[^\s"\'<>]+', re.I)
RE_BIOPROTOC_DOI = re.compile(r'https?://doi\.org/10\.21769/BioProtoc\.\d+', re.I)
RE_ORIGINAL_NEAR = re.compile(r'Original\s+Research\s+Article', re.I)


def load_bio_index():
    data = orjson.loads(BIO.read_bytes())
    return {str(x.get("id")): x for x in data}


def get_biopro_doi_url(rec: dict) -> str:
    # bio_protocol.json 안의 DOI URL (보통 rec["url"])
    for k in ("url", "doi_url", "doi"):
        v = rec.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    links = rec.get("links") or {}
    if isinstance(links, dict):
        for v in links.values():
            if isinstance(v, str) and "doi.org" in v:
                return v.strip()
    return ""


def pick_original_link(html: str, fallback_to_any=True) -> str:
    """
    Bioprotocol 페이지 HTML에서 '원문 논문' DOI/URL을 고릅니다.
    1) 페이지 내 모든 doi.org 링크 수집
    2) BioProtoc DOI(자기 자신)는 제외
    3) 남은 후보가 여러 개인 경우:
       - 'Original Research Article' 근처(±2000자)에 있는 링크 우선
       - 그 외 첫 번째 링크
    """
    if not html:
        return ""
    all_dois = RE_DOI_LINK.findall(html)
    if not all_dois:
        return ""

    # 자기 자신(BioProtoc) DOI 제외
    doi_candidates = [u for u in all_dois if not RE_BIOPROTOC_DOI.search(u)]
    if not doi_candidates:
        # 다른 DOI가 없으면, 필요 시 아무거나 리턴 (off by default)
        return doi_candidates[0] if fallback_to_any and all_dois else ""

    # "Original Research Article" 근처에 있는 링크를 우선
    m = RE_ORIGINAL_NEAR.search(html)
    if m:
        center = m.start()
        window = 2000  # 가볍게 근접 후보 보기
        segment = html[max(0, center - window): center + window]
        near = [u for u in doi_candidates if u in segment]
        if near:
            return near[0]

    # 그 외 첫 번째 후보
    return doi_candidates[0]


def fetch_original_for_one(client: httpx.Client, biopro_doi: str) -> tuple[str, str]:
    """
    Bioprotocol DOI를 받아 최종 랜딩 페이지 HTML을 가져와
    원문 기사 DOI/URL을 리턴합니다.
    반환: (original_article_url, notes)
    """
    try:
        # DOI -> Bioprotocol 랜딩 페이지로 리다이렉트 추적
        r = client.get(biopro_doi, headers=HEADERS, follow_redirects=True, timeout=TIMEOUT)
        r.raise_for_status()
        html = r.text
        orig = pick_original_link(html)
        if orig:
            return orig, "ok"
        else:
            return "", "no_original_link_found"
    except httpx.HTTPError as e:
        return "", f"http_error:{e}"


def main():
    assert BIO.exists(), f"not found: {BIO}"
    assert TEST_IDS.exists(), f"not found: {TEST_IDS}"

    bio_idx = load_bio_index()
    test_ids = [x.strip() for x in TEST_IDS.read_text().splitlines() if x.strip()]

    rows = []
    urls_only = []

    limits = httpx.Limits(max_connections=MAX_WORKERS, max_keepalive_connections=MAX_WORKERS)
    transport = httpx.HTTPTransport(retries=2)

    with httpx.Client(http2=True, limits=limits, transport=transport) as client:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            futs = {}
            for pid in test_ids:
                rec = bio_idx.get(pid)
                if not rec:
                    rows.append({
                        "id": pid, "title": "", "biopro_doi": "", "original_article_url": "",
                        "status": "missing_in_bio_protocol", "notes": ""
                    })
                    continue
                biopro_doi = get_biopro_doi_url(rec)
                title = (rec.get("title") or "").strip()
                if not biopro_doi:
                    rows.append({
                        "id": pid, "title": title, "biopro_doi": "",
                        "original_article_url": "", "status": "no_biopro_doi", "notes": ""
                    })
                    continue
                # 병렬 제출
                fut = ex.submit(fetch_original_for_one, client, biopro_doi)
                futs[fut] = (pid, title, biopro_doi)

            for fut in tqdm(as_completed(futs), total=len(futs), desc="collect"):
                pid, title, biopro_doi = futs[fut]
                try:
                    orig, notes = fut.result()
                    status = "ok" if orig else "not_found"
                    if orig:
                        urls_only.append(orig)
                    rows.append({
                        "id": pid, "title": title, "biopro_doi": biopro_doi,
                        "original_article_url": orig, "status": status, "notes": notes
                    })
                except Exception as e:
                    rows.append({
                        "id": pid, "title": title, "biopro_doi": biopro_doi,
                        "original_article_url": "", "status": "error", "notes": str(e)
                    })

    # 저장
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id", "title", "biopro_doi", "original_article_url", "status", "notes"])
        w.writeheader();
        w.writerows(rows)
    OUT_TXT.write_text("\n".join(urls_only), encoding="utf-8")

    print(f"[OK] saved: {OUT_CSV} (rows={len(rows)})")
    print(f"[OK] saved: {OUT_TXT} (urls={len(urls_only)})")
    missing = [r for r in rows if r["status"] != "ok"]
    if missing:
        miss_p = OUT_DIR / "test_original_urls_missing.csv"
        with open(miss_p, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["id", "title", "biopro_doi", "original_article_url", "status", "notes"])
            w.writeheader();
            w.writerows(missing)
        print(f"[INFO] missing or errors: {len(missing)} -> {miss_p}")


if __name__ == "__main__":
    main()
