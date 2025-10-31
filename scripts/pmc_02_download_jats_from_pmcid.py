#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pmc_02_download_jats_from_pmcid.py

Purpose
- Download PMC JATS XML per PMCID from the mapping CSV (pmc_01 output).
- Accept rows by status (OK, low_conf, ...) OR accept any PMCID regardless of status.
- Try OAI-PMH GetRecord first; if it fails (400/403/404 or sanity fail), fall back to Entrez EFetch (db=pmc).
- Resume-friendly (skip existing unless --overwrite).
- Robust network handling (retry/backoff) + audit logs.

Inputs (CSV columns from pmc_01):
  protocol_id, biop_domain, biop_title, original_article_url,
  pmid, pmcid, source, status, reason,
  title_sim, oa_check, confidence,
  biop_url, biop_keywords, biop_hier_len, biop_abstract

Outputs
- data/gold/pmc_jats/PMCxxxxxx.xml
- runs/YYYY-MM-DD/pmc_02_success.jsonl
- runs/YYYY-MM-DD/pmc_02_failures.jsonl
- runs/YYYY-MM-DD/pmc_02_skipped.jsonl

Notes
- OAI-PMH: https://www.ncbi.nlm.nih.gov/pmc/oai/oai.cgi
- EFetch  : https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pmc&id=PMCxxxxxx
- Set NCBI_API_KEY in env for better rate-limits.
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
from typing import Dict, List, Tuple

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

ROOT = pathlib.Path(".")
DEF_MAP = ROOT / "data/gold/pmc_map_from_urls.csv"
DEF_OUTD = ROOT / "data/gold/pmc_jats"

OAI_URL = "https://www.ncbi.nlm.nih.gov/pmc/oai/oai.cgi"
EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
NCBI_API_KEY = os.environ.get("NCBI_API_KEY", "")


def make_session():
    s = requests.Session()
    retry = Retry(
        total=5, connect=5, read=5, backoff_factor=0.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET", "POST"]),
    )
    s.headers.update({"User-Agent": "BioProtocol/1.0 (contact: you@example.org)"})
    s.mount("https://", HTTPAdapter(max_retries=retry))
    s.mount("http://", HTTPAdapter(max_retries=retry))
    return s


SESSION = make_session()


def safe_get(url, params=None, timeout=60):
    params = dict(params or {})
    if "ncbi.nlm.nih.gov" in url and NCBI_API_KEY:
        params["api_key"] = NCBI_API_KEY
    r = SESSION.get(url, params=params, timeout=timeout)
    r.raise_for_status()
    time.sleep(0.34)
    return r


def today_run_dir() -> pathlib.Path:
    d = ROOT / "runs" / dt.date.today().isoformat()
    d.mkdir(parents=True, exist_ok=True)
    return d


def write_jsonl(path: pathlib.Path, rows: List[Dict]):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def to_pmc_number(pmcid: str) -> str:
    return (pmcid or "").upper().replace("PMC", "")


def quick_jats_sanity(xml_text: str) -> bool:
    return bool(re.search(r"<article[\s>]", xml_text, flags=re.I)) and ("</article>" in xml_text)


def load_map_csv(path: pathlib.Path) -> List[Dict]:
    if not path.exists():
        print(f"[ERR] not found: {path}", file=sys.stderr);
        sys.exit(1)
    return list(csv.DictReader(open(path, "r", encoding="utf-8")))


def select_targets(rows: List[Dict], accept_status: List[str], accept_any_pmcid: bool) -> List[Dict]:
    out = []
    for r in rows:
        pmcid = (r.get("pmcid") or "").strip()
        status = (r.get("status") or "").strip()
        if not pmcid:
            continue
        if accept_any_pmcid or (status in accept_status):
            out.append(r)
    return out


def try_oai_getrecord(pmcid: str, timeout=90) -> Tuple[bytes, str]:
    """Return (xml_bytes, reason). reason='' if ok."""
    pmc_num = to_pmc_number(pmcid)
    params = {
        "verb": "GetRecord",
        "identifier": f"oai:pubmedcentral.nih.gov:{pmc_num}",
        "metadataPrefix": "pmc",
    }
    try:
        r = safe_get(OAI_URL, params=params, timeout=timeout)
        xml_bytes = r.content
        text = r.text
        if quick_jats_sanity(text):
            return xml_bytes, ""
        else:
            # sometimes OAI returns wrapper; keep but mark suspect
            return xml_bytes, "oai_suspect_jats"
    except requests.HTTPError as e:
        code = e.response.status_code if e.response is not None else "HTTPError"
        return b"", f"oai_http_{code}"
    except Exception as e:
        return b"", f"oai_exc_{e.__class__.__name__}"


def try_efetch_pmc(pmcid: str, timeout=90) -> Tuple[bytes, str]:
    """Return (xml_bytes, reason). reason='' if ok."""
    params = {"db": "pmc", "id": pmcid}
    try:
        r = safe_get(EFETCH_URL, params=params, timeout=timeout)
        xml_bytes = r.content
        text = r.text
        if quick_jats_sanity(text):
            return xml_bytes, ""
        else:
            return xml_bytes, "efetch_suspect_jats"
    except requests.HTTPError as e:
        code = e.response.status_code if e.response is not None else "HTTPError"
        return b"", f"efetch_http_{code}"
    except Exception as e:
        return b"", f"efetch_exc_{e.__class__.__name__}"


def download_one(rec: Dict, outd: pathlib.Path, overwrite: bool) -> Dict:
    pid = (rec.get("protocol_id") or "").strip()
    pmcid = (rec.get("pmcid") or "").strip().upper()
    if not pmcid.startswith("PMC"):
        pmcid = "PMC" + to_pmc_number(pmcid)

    outd.mkdir(parents=True, exist_ok=True)
    out_path = outd / f"{pmcid}.xml"

    if out_path.exists() and not overwrite:
        return {"protocol_id": pid, "pmcid": pmcid, "out_path": str(out_path), "ok": None, "reason": "exists_skipped"}

    # 1) OAI first
    xml, r1 = try_oai_getrecord(pmcid)
    if xml and not r1.startswith("oai_http_"):
        # save even if suspect
        out_path.write_bytes(xml)
        return {"protocol_id": pid, "pmcid": pmcid, "out_path": str(out_path), "ok": True, "reason": r1 or ""}

    # 2) Fall back to EFetch
    xml2, r2 = try_efetch_pmc(pmcid)
    if xml2:
        out_path.write_bytes(xml2)
        return {"protocol_id": pid, "pmcid": pmcid, "out_path": str(out_path), "ok": True, "reason": r2 or "efetch_ok"}

    # both failed
    reason = r1 if r1 else r2
    return {"protocol_id": pid, "pmcid": pmcid, "out_path": "", "ok": False, "reason": reason or "unknown"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", type=str, default=str(DEF_MAP), help="CSV from pmc_01 step")
    ap.add_argument("--outd", type=str, default=str(DEF_OUTD), help="Output dir for JATS XML")
    ap.add_argument("--accept-status", type=str, default="OK,low_conf",
                    help="Comma-separated statuses to accept (e.g., OK,low_conf)")
    ap.add_argument("--accept-any-pmcid", action="store_true", default=False,
                    help="Ignore status; download for any row with a PMCID")
    ap.add_argument("--overwrite", action="store_true", default=False,
                    help="Re-download and overwrite existing XML files")
    args = ap.parse_args()

    map_path = pathlib.Path(args.map)
    outd = pathlib.Path(args.outd)
    accept_status = [s.strip() for s in (args.accept_status or "").split(",") if s.strip()]
    rundir = today_run_dir()

    rows = load_map_csv(map_path)
    targets = select_targets(rows, accept_status, args.accept_any_pmcid)

    successes, failures, skipped = [], [], []
    print(f"[INFO] total rows in map: {len(rows)}")
    print(
        f"[INFO] targets selected:  {len(targets)} (accept_any={args.accept_any_pmcid}, accept_status={accept_status})")

    for rec in targets:
        res = download_one(rec, outd, overwrite=args.overwrite)
        if res["ok"] is True:
            successes.append({**rec, **res})
        elif res["ok"] is False:
            failures.append({**rec, **res})
        else:
            skipped.append({**rec, **res})

    success_log = rundir / "pmc_02_success.jsonl"
    failure_log = rundir / "pmc_02_failures.jsonl"
    skipped_log = rundir / "pmc_02_skipped.jsonl"
    write_jsonl(success_log, successes)
    write_jsonl(failure_log, failures)
    write_jsonl(skipped_log, skipped)

    print(f"[DONE] out_dir={outd}")
    print(f"  downloaded: {len(successes)}")
    print(f"  failed    : {len(failures)}  (see {failure_log})")
    print(f"  skipped   : {len(skipped)}   (existing or filtered)")
    print(f"[LOGS] success -> {success_log}")
    print(f"[LOGS] failures-> {failure_log}")
    print(f"[LOGS] skipped -> {skipped_log}")


if __name__ == "__main__":
    main()
