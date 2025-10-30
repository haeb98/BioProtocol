#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pmc_01_urls_to_pmcid_batched.py

Purpose
- From a list of Bio-protocol original-article URLs, map each protocol to PubMed/PMC identifiers.
- Robust join with data/raw/bio_protocol.json to enrich domain/title/etc. (for downstream eval).
- Batch-convert PMID -> PMCID via NCBI idconv API.
- (Light) OA/JATS availability probe via PMC OAI-PMH GetRecord, and fetch article-title for title similarity.
- Compute a confidence score and keep the best candidate per protocol_id (deduplicate).
- Emit a single, analysis-friendly CSV.

Inputs
- --urls: CSV with columns [id or protocol_id, original_article_url, (optional title/domain...)]
- --biop-ids: CSV with columns [protocol_id,...] (optional; if given, restrict mapping to these IDs)
- --bio: JSON array file at data/raw/bio_protocol.json (default). Required for domain/title/metadata join.

Outputs
- data/gold/pmc_map_from_urls.csv  (protocol_id, pmid, pmcid, domain/title/..., confidence, status, reason, ...)
- runs/YYYY-MM-DD/pmc_01_errors.jsonl         (network/low_confidence/no_pmc etc.)
- runs/YYYY-MM-DD/pmc_01_dropped.jsonl        (duplicates dropped by lower confidence)
"""

import argparse
import csv
import datetime as dt
import json
import os
import pathlib
import re
import sys
import time
from urllib.parse import urlparse

import orjson
import requests
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from urllib3.util.retry import Retry

# -------------------------------
# Config & Constants
# -------------------------------
ROOT = pathlib.Path(".")
DEF_URLS = ROOT / "data/gold/bio_protocol_original_articles.csv"
DEF_BIO = ROOT / "data/raw/bio_protocol.json"
DEF_OUT = ROOT / "data/gold/pmc_map_from_urls.csv"

IDCONV = "https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/"
OAI = "https://www.ncbi.nlm.nih.gov/pmc/oai/oai.cgi"

PUBMED_HOSTS = {"ncbi.nlm.nih.gov", "pubmed.ncbi.nlm.nih.gov"}

RE_PMCID = re.compile(r"/pmc/articles/(PMC\d+)", re.I)
RE_PMID = re.compile(r"/pubmed/(\d+)|pubmed\.ncbi\.nlm\.nih\.gov/(\d+)", re.I)

NCBI_API_KEY = os.environ.get("NCBI_API_KEY", "")

CSV_HEADER = [
    "protocol_id",
    "biop_domain",
    "biop_title",
    "original_article_url",
    "pmid",
    "pmcid",
    "source",  # pmcid_from_url | pmcid_from_idconv | skip_non_pubmed | ...
    "status",  # OK | low_confidence | no_pmc | idconv_error | skip_non_pubmed | ...
    "reason",
    "title_sim",  # 0~1
    "oa_check",  # yes | no | unknown
    "confidence",  # 0~1
    # helpful for downstream eval:
    "biop_url",
    "biop_keywords",
    "biop_hier_len",
    "biop_abstract",
]


# -------------------------------
# HTTP Session with Retry
# -------------------------------
def make_session():
    s = requests.Session()
    retry = Retry(
        total=5,
        connect=5,
        read=5,
        backoff_factor=0.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET", "POST"]),
    )
    s.headers.update({"User-Agent": "BioProtocol/1.0 (contact: you@example.org)"})
    s.mount("https://", HTTPAdapter(max_retries=retry))
    s.mount("http://", HTTPAdapter(max_retries=retry))
    return s


SESSION = make_session()


def safe_get(url, params=None, timeout=30):
    params = {} if params is None else dict(params)
    if "ncbi.nlm.nih.gov" in url and NCBI_API_KEY:
        params["api_key"] = NCBI_API_KEY
    r = SESSION.get(url, params=params, timeout=timeout)
    r.raise_for_status()
    time.sleep(0.34)  # Respect rate limits
    return r


# -------------------------------
# Helpers
# -------------------------------
def today_run_dir():
    d = ROOT / "runs" / dt.date.today().isoformat()
    d.mkdir(parents=True, exist_ok=True)
    return d


def write_jsonl(path: pathlib.Path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def norm_id(s: str) -> str:
    return (s or "").strip()


def is_pubmedish(u: str) -> bool:
    if not u:
        return False
    try:
        h = urlparse(u).netloc.lower()
    except Exception:
        return False
    return any(host in h for host in PUBMED_HOSTS)


def extract_pmcid(url: str) -> str:
    m = RE_PMCID.search(url or "")
    if not m:
        return ""
    pmcid = m.group(1).upper()
    if not pmcid.startswith("PMC"):
        pmcid = f"PMC{pmcid}"
    return pmcid


def extract_pmid(url: str) -> str:
    m = RE_PMID.search(url or "")
    if not m:
        return ""
    g1, g2 = m.groups()
    pmid = (g1 or g2 or "").strip()
    return pmid


def tokenize_title(t: str):
    return set(re.findall(r"[A-Za-z0-9]+", (t or "").lower()))


def jaccard(a: str, b: str) -> float:
    A = tokenize_title(a)
    B = tokenize_title(b)
    if not A and not B:
        return 0.0
    return len(A & B) / max(1, len(A | B))


# -------------------------------
# Loaders & Join
# -------------------------------
def load_bio_index(bio_path: pathlib.Path):
    if not bio_path.exists():
        print(f"[ERR] not found: {bio_path}", file=sys.stderr)
        return {}
    try:
        data = orjson.loads(bio_path.read_bytes())  # JSON array
    except Exception as e:
        print(f"[ERR] parse bio_protocol.json: {e}", file=sys.stderr)
        return {}
    idx = {}
    for r in data:
        pid = norm_id(r.get("id"))
        if not pid:
            continue
        cls = r.get("classification") or {}
        dom = cls.get("primary_domain") or "Unknown"
        meta = {
            "biop_id": pid,
            "biop_title": (r.get("title") or "").strip(),
            "biop_domain": dom,
            "biop_keywords": (r.get("keywords") or "").strip(),
            "biop_url": (r.get("url") or "").strip(),
            "biop_hier_len": len((r.get("hierarchical_protocol") or {})),
            "biop_abstract": (r.get("abstract") or "").strip(),
        }
        idx[pid] = meta
    return idx


def load_urls_csv(path: pathlib.Path):
    if not path.exists():
        print(f"[ERR] not found: {path}", file=sys.stderr)
        return []
    rows = list(csv.DictReader(open(path, "r", encoding="utf-8")))
    return rows


def load_test_ids(path: pathlib.Path):
    if not path or not path.exists():
        return None
    rows = list(csv.DictReader(open(path, "r", encoding="utf-8")))
    ids = {norm_id(r.get("protocol_id")) for r in rows if r.get("protocol_id")}
    return ids


def join_urls_with_bio(url_rows, bio_idx, restrict_ids=None):
    joined = []
    miss = 0
    for r in url_rows:
        pid = norm_id(r.get("id") or r.get("protocol_id"))
        if not pid:
            continue
        if restrict_ids and pid not in restrict_ids:
            continue
        url = (r.get("original_article_url") or r.get("original_url") or "").strip()
        if not url:
            continue
        meta = bio_idx.get(pid)
        if not meta:
            miss += 1
            meta = {
                "biop_id": pid,
                "biop_title": "",
                "biop_domain": "",
                "biop_keywords": "",
                "biop_url": "",
                "biop_hier_len": "",
                "biop_abstract": "",
            }
        joined.append({
            "protocol_id": pid,
            "original_article_url": url,
            **meta
        })
    if miss:
        print(f"[WARN] bio join missed {miss} ids (check id mismatch/whitespace).")
    return joined


# -------------------------------
# NCBI idconv (PMID -> PMCID)
# -------------------------------
def idconv_batch(pmids, batch_size=100):
    pmid2pmcid = {}
    if not pmids:
        return pmid2pmcid
    pmids = [p for p in pmids if p]
    for i in range(0, len(pmids), batch_size):
        chunk = pmids[i:i + batch_size]
        params = {"ids": ",".join(chunk), "format": "json"}
        r = safe_get(IDCONV, params=params, timeout=60)
        j = r.json()
        recs = j.get("records", [])
        for rec in recs:
            pmid = str(rec.get("pmid") or "").strip()
            pmcid = str(rec.get("pmcid") or "").strip()
            if pmid and pmcid:
                pmid2pmcid[pmid] = pmcid if pmcid.upper().startswith("PMC") else f"PMC{pmcid}"
    return pmid2pmcid


# -------------------------------
# OAI-PMH probe & title fetch
# -------------------------------
def probe_oai_and_title(pmcid: str, timeout=30):
    """
    Returns: (oa_check: str in {'yes','no','unknown'}, article_title: str)
    """
    if not pmcid:
        return "unknown", ""
    try:
        pmcnum = pmcid.upper().replace("PMC", "")
        params = {
            "verb": "GetRecord",
            "identifier": f"oai:pubmedcentral.nih.gov:{pmcnum}",
            "metadataPrefix": "pmc",
        }
        r = safe_get(OAI, params=params, timeout=timeout)
        xml = r.text
        # Try to extract <article-title> (simple heuristic)
        # Avoid heavy XML deps; a light regex fallback:
        # (If lxml installed, you can switch to etree parsing.)
        title = ""
        m = re.search(r"<article-title[^>]*>(.*?)</article-title>", xml, flags=re.I | re.S)
        if m:
            # Remove tags in-between
            raw = m.group(1)
            title = re.sub(r"<[^>]+>", "", raw)
            title = " ".join(title.split())
        return "yes", title
    except requests.HTTPError as e:
        if e.response is not None and e.response.status_code in (403, 404):
            return "no", ""
        return "unknown", ""
    except Exception:
        return "unknown", ""


# -------------------------------
# Confidence
# -------------------------------
def compute_confidence(title_sim: float, source: str, oa_check: str):
    base = 0.7 * title_sim
    src_score = 1.0 if source == "pmcid_from_url" else (0.8 if source == "pmcid_from_idconv" else 0.0)
    base += 0.3 * src_score
    if oa_check == "yes":
        base += 0.05
    return min(1.0, max(0.0, base))


# -------------------------------
# Main
# -------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--urls", type=str, default=str(DEF_URLS),
                    help="CSV with columns: id or protocol_id, original_article_url, ...")
    ap.add_argument("--biop-ids", type=str, default="",
                    help="Optional CSV to restrict to test IDs (protocol_id column).")
    ap.add_argument("--bio", type=str, default=str(DEF_BIO),
                    help="JSON (array) data/raw/bio_protocol.json")
    ap.add_argument("--out", type=str, default=str(DEF_OUT))
    ap.add_argument("--min-conf", type=float, default=0.6,
                    help="Confidence threshold for status=OK")
    ap.add_argument("--batch-size", type=int, default=100)
    ap.add_argument("--probe-oai", action="store_true", default=True)
    ap.add_argument("--no-probe-oai", dest="probe_oai", action="store_false")
    args = ap.parse_args()

    urls_path = pathlib.Path(args.urls)
    bio_path = pathlib.Path(args.bio)
    out_path = pathlib.Path(args.out)
    ids_path = pathlib.Path(args.biop_ids) if args.biop_ids else None

    out_path.parent.mkdir(parents=True, exist_ok=True)
    rundir = today_run_dir()
    err_log = rundir / "pmc_01_errors.jsonl"
    drop_log = rundir / "pmc_01_dropped.jsonl"

    # 1) Load & Join
    bio_idx = load_bio_index(bio_path)
    url_rows = load_urls_csv(urls_path)
    restrict_ids = load_test_ids(ids_path) if ids_path else None
    joined = join_urls_with_bio(url_rows, bio_idx, restrict_ids)

    # 2) PubMed-ish filter & initial extraction
    candidates = []
    for rec in joined:
        url = rec["original_article_url"]
        if not is_pubmedish(url):
            candidates.append({**rec,
                               "pmid": "", "pmcid": "",
                               "source": "skip_non_pubmed",
                               "status": "skip_non_pubmed",
                               "reason": "non_pubmed_url",
                               "title_sim": "",
                               "oa_check": "",
                               "confidence": ""})
            continue
        pmcid = extract_pmcid(url)
        pmid = extract_pmid(url)
        src = "pmcid_from_url" if pmcid else ("pmid_found" if pmid else "")
        candidates.append({**rec,
                           "pmid": pmid, "pmcid": pmcid,
                           "source": src, "status": "", "reason": "",
                           "title_sim": "", "oa_check": "", "confidence": ""})

    # 3) Batch idconv for those with pmid but no pmcid
    pmids_need = [c["pmid"] for c in candidates if (c["pmid"] and not c["pmcid"])]
    pmid2pmcid = idconv_batch(pmids_need, batch_size=args.batch_size)

    for c in candidates:
        if not c["pmcid"] and c["pmid"]:
            pmcid = pmid2pmcid.get(c["pmid"], "")
            if pmcid:
                c["pmcid"] = pmcid
                c["source"] = "pmcid_from_idconv"

    # 4) OA probe + title similarity + confidence
    errors = []
    for c in tqdm(candidates, desc="oai-probe & confidence"):
        if c["source"] == "skip_non_pubmed":
            # leave as-is
            continue
        pmcid = c["pmcid"]
        if args.probe_oai:
            oa_check, art_title = probe_oai_and_title(pmcid) if pmcid else ("unknown", "")
        else:
            oa_check, art_title = ("unknown", "")

        # If article-title Not found via OAI, title_sim = jaccard(biop_title, "")
        title_sim = jaccard(c.get("biop_title", ""), art_title) if art_title else 0.0
        conf = compute_confidence(title_sim, c.get("source", ""), oa_check)

        c["title_sim"] = f"{title_sim:.3f}"
        c["oa_check"] = oa_check
        c["confidence"] = f"{conf:.3f}"

        # status / reason decision
        if not c["pmcid"]:
            c["status"] = "no_pmc"
            c["reason"] = "no_pmcid_after_idconv"
            errors.append({**c})
        else:
            if conf >= args.min_conf:
                c["status"] = "OK"
            else:
                c["status"] = "low_confidence"
                c["reason"] = "title_mismatch_or_low_conf"
                errors.append({**c})

    # 5) Deduplicate by protocol_id (keep highest confidence)
    best_by_id = {}
    dropped = []
    for c in candidates:
        pid = c["protocol_id"]
        key_conf = float(c["confidence"] or 0.0) if c["confidence"] != "" else (-1.0)
        prev = best_by_id.get(pid)
        if prev is None:
            best_by_id[pid] = c
        else:
            prev_conf = float(prev["confidence"] or 0.0) if prev["confidence"] != "" else (-1.0)
            if key_conf > prev_conf:
                dropped.append(prev)
                best_by_id[pid] = c
            else:
                dropped.append(c)

    # 6) Write outputs
    rows = list(best_by_id.values())
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(CSV_HEADER)
        for r in rows:
            w.writerow([
                r.get("protocol_id", ""),
                r.get("biop_domain", ""),
                r.get("biop_title", ""),
                r.get("original_article_url", ""),
                r.get("pmid", ""),
                r.get("pmcid", ""),
                r.get("source", ""),
                r.get("status", ""),
                r.get("reason", ""),
                r.get("title_sim", ""),
                r.get("oa_check", ""),
                r.get("confidence", ""),
                r.get("biop_url", ""),
                r.get("biop_keywords", ""),
                r.get("biop_hier_len", ""),
                r.get("biop_abstract", ""),
            ])

    write_jsonl(err_log, errors)
    write_jsonl(drop_log, dropped)

    # 7) Summary
    ok = sum(1 for r in rows if r["status"] == "OK")
    low = sum(1 for r in rows if r["status"] == "low_confidence")
    nop = sum(1 for r in rows if r["status"] == "no_pmc")
    skp = sum(1 for r in rows if r["status"] == "skip_non_pubmed")
    print(
        f"[DONE] wrote: {out_path}  (total unique={len(rows)}, OK={ok}, low_conf={low}, no_pmc={nop}, skip_non_pubmed={skp})")
    print(f"[LOGS] errors -> {err_log}")
    print(f"[LOGS] dropped -> {drop_log}")


if __name__ == "__main__":
    main()
