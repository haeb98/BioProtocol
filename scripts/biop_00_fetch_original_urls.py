#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

UA = "BioProtocol-MappingBot/1.3 (strict-original-card-only; +contact: your_email)"

PUBMED_PATTERNS = [
    re.compile(r"https?://(?:www\.)?ncbi\.nlm\.nih\.gov/pubmed/(\d+)", re.I),
    re.compile(r"https?://pubmed\.ncbi\.nlm\.nih\.gov/(\d+)/?", re.I),
]
PMCID_PAT = re.compile(r"PMC(\d+)", re.I)
DOI_PAT = re.compile(r"https?://doi\.org/(.+)", re.I)
TITLE_RX = re.compile(r"\boriginal\s+research\s+article\b", re.I)


def read_index(path, id_col=None, url_col=None):
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        # TSV일 수도 있으므로 python 엔진으로 시도
        header = f.readline()
        sep = "\t" if ("\t" in header and "," not in header) else ","
        f.seek(0)
        r = csv.DictReader(f, delimiter=sep)
        # 컬럼 자동 추정
        if id_col is None:
            for cand in ("protocol_id", "id", "ProtocolID", "bp_id"):
                if cand in r.fieldnames: id_col = cand; break
        if url_col is None:
            for cand in ("biop_url", "url", "detail_url", "bp_url"):
                if cand in r.fieldnames: url_col = cand; break
        if not id_col or not url_col:
            raise SystemExit(f"[read_index] 컬럼명을 찾지 못했습니다. id_col={id_col}, url_col={url_col}, headers={r.fieldnames}")

        for rec in r:
            pid = rec.get(id_col);
            url = rec.get(url_col)
            if pid and url:
                rows.append({"protocol_id": str(pid), "biop_url": url})
    return rows


def session_with_retries():
    s = requests.Session()
    adapter = requests.adapters.HTTPAdapter(max_retries=3, pool_connections=50, pool_maxsize=50)
    s.mount("http://", adapter);
    s.mount("https://", adapter)
    s.headers.update({
        "User-Agent": UA,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9,ko;q=0.8",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
    })
    return s


def cache_path(base, pid): return base / f"{pid}.html"


def safe_get(s, url, timeout=25):
    try:
        return s.get(url, timeout=timeout, allow_redirects=True)
    except requests.RequestException:
        return None


def has_original_phrase(soup: BeautifulSoup) -> bool:
    txt = soup.get_text(" ", strip=True)
    return bool(TITLE_RX.search(txt))


def find_original_box(soup: BeautifulSoup):
    """
    스샷 기준 구조:
      <div class="right_box">
        ...
        <div class="article_box">
          <p class="article_type">Original research article</p>
          <p> The authors used this protocol in: </p>
          <div>
            <a href="...">...</a>
          </div>
          <p>Dec 2015</p>
        </div>
      </div>
    """
    # 1) p.article_type 텍스트 매칭 → 조상 중 class에 'article_box'가 포함된 컨테이너
    for p in soup.select("p.article_type"):
        text = " ".join(p.stripped_strings).lower()
        if "original" in text and "research" in text and "article" in text:
            # 가장 가까운 조상 중 class에 article_box 포함
            parent = p
            for _ in range(8):
                parent = parent.parent
                if not parent: break
                classes = " ".join(parent.get("class", [])).lower()
                if "article_box" in classes:
                    if parent.select_one("a[href]"):
                        return parent
                    break  # article_box는 찾았지만 링크 없으면 하위는 의미 없음

    # 2) 텍스트 기반 백업(아주 보수적으로)
    for el in soup.find_all(string=TITLE_RX):
        parent = el.parent
        for _ in range(8):
            if not parent: break
            classes = " ".join(parent.get("class", [])).lower()
            if ("article_box" in classes) and parent.select_one("a[href]"):
                return parent
            parent = parent.parent

    return None


def extract_from_original_box(box, base_url):
    urls = []
    for a in box.select("a[href]"):
        href = a.get("href", "").strip()
        if not href: continue
        url = urljoin(base_url, href)
        urls.append(url)
    # 중복 제거
    seen, uniq = set(), []
    for u in urls:
        if u not in seen:
            seen.add(u);
            uniq.append(u)
    return uniq


def normalize_pubmed(url):
    for pat in PUBMED_PATTERNS:
        m = pat.match(url)
        if m:
            pmid = m.group(1)
            return f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/", pmid
    return None, None


def extract_pmcid(url):
    m = PMCID_PAT.search(url)
    return (f"PMC{m.group(1)}" if m else None)


def choose_in_box(urls):
    # pubmed > pmc > doi > 기타
    def score(u):
        u_low = u.lower()
        if "pubmed" in u_low: return (3, "original_box_pubmed")
        if "/pmc/articles/" in u_low: return (2, "original_box_pmc")
        if "doi.org/" in u_low: return (1, "original_box_doi")
        return (0, "original_box_other")

    scored = [(score(u)[0], score(u)[1], u) for u in urls]
    if not scored: return None, None
    best = sorted(scored, key=lambda x: (-x[0], len(x[2])))[0]
    return best[1], best[2]


def worker(rec, sess, cache_dir, sleep=0.35, refresh=False):
    pid, url = rec["protocol_id"], rec["biop_url"]
    cpath = cache_path(cache_dir, pid)

    html = None
    from_cache = False
    if cpath.exists() and not refresh:
        html = cpath.read_text(encoding="utf-8", errors="ignore")
        from_cache = True
    else:
        resp = safe_get(sess, url)
        if not resp or resp.status_code >= 400 or not resp.text:
            return {"protocol_id": pid, "biop_url": url, "status": "fetch_fail",
                    "pick_url": None, "pick_source": None,
                    "pubmed_url": None, "pmid": None, "pmcid_hint": None, "doi_hint": None,
                    "candidates": "[]", "cache": "miss", "has_phrase": None,
                    "box_class": None, "num_links_in_box": 0}
        html = resp.text
        cpath.parent.mkdir(parents=True, exist_ok=True)
        cpath.write_text(html, encoding="utf-8")
        from_cache = False

    soup = BeautifulSoup(html, "lxml")
    phrase = has_original_phrase(soup)
    box = find_original_box(soup)
    if not box:
        return {"protocol_id": pid, "biop_url": url,
                "status": ("no_original_box_phrase" if phrase else "no_phrase_in_page"),
                "pick_url": None, "pick_source": None,
                "pubmed_url": None, "pmid": None, "pmcid_hint": None, "doi_hint": None,
                "candidates": "[]", "cache": "hit" if from_cache else "miss",
                "has_phrase": phrase, "box_class": None, "num_links_in_box": 0}

    urls_in_box = extract_from_original_box(box, url)
    pick_source, pick_url = choose_in_box(urls_in_box)

    pubmed_url, pmid = (None, None)
    pmcid = None;
    doi = None
    if pick_url:
        pubmed_url, pmid = normalize_pubmed(pick_url)
        if not pubmed_url and "pubmed" in pick_url.lower():
            pubmed_url = pick_url  # 구형 형태면 raw 유지
        pmcid = extract_pmcid(pick_url)
        m = DOI_PAT.match(pick_url)
        if m: doi = m.group(1)

    cand_json = json.dumps([{"url": u} for u in urls_in_box], ensure_ascii=False)
    time.sleep(sleep)

    classes = " ".join(box.get("class", []))
    return {
        "protocol_id": pid, "biop_url": url,
        "status": "ok" if pick_url else "box_link_unusable",
        "pick_url": pick_url, "pick_source": pick_source,
        "pubmed_url": pubmed_url, "pmid": pmid, "pmcid_hint": pmcid, "doi_hint": doi,
        "candidates": cand_json, "cache": "hit" if from_cache else "miss",
        "has_phrase": phrase, "box_class": classes, "num_links_in_box": len(urls_in_box)
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True)
    ap.add_argument("--out", default="data/gold/biop_original_candidates.csv")
    ap.add_argument("--cache-dir", default="data/cache/biop_html")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--sleep", type=float, default=0.35)
    ap.add_argument("--id-col", default=None)
    ap.add_argument("--url-col", default=None)
    ap.add_argument("--refresh", action="store_true", help="Ignore cache and refetch HTML")
    args = ap.parse_args()

    Path(args.cache_dir).mkdir(parents=True, exist_ok=True)
    rows = read_index(args.index, id_col=args.id_col, url_col=args.url_col)
    print(f"[info] rows from index: {len(rows)}", file=sys.stderr)

    sess = session_with_retries()
    outp = Path(args.out)
    with outp.open("w", newline="", encoding="utf-8") as w:
        fns = ["protocol_id", "biop_url", "status", "pick_url", "pick_source",
               "pubmed_url", "pmid", "pmcid_hint", "doi_hint",
               "candidates", "cache", "has_phrase", "box_class", "num_links_in_box"]
        cw = csv.DictWriter(w, fieldnames=fns);
        cw.writeheader()
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(worker, rec, sess, Path(args.cache_dir), args.sleep, args.refresh)
                    for rec in rows]
            for fut in as_completed(futs):
                cw.writerow(fut.result())

    # 요약
    import collections
    cnt = collections.Counter()
    with outp.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            cnt[r["status"]] += 1
    print("[summary]", dict(cnt), file=sys.stderr)


if __name__ == "__main__":
    main()
