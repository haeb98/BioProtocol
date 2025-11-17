#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import re
import sys
import time
from pathlib import Path

import requests

UA = "BioProtocol-IDResolver/1.1 (+contact: your_email)"
IDCONV = "https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/?format=json"

PUBMED_RX = re.compile(r"(?:/pubmed/|pubmed\.ncbi\.nlm\.nih\.gov/)(\d+)", re.I)
PMCID_RX = re.compile(r"(?i)pmc(\d+)")
PMID_ONLY = re.compile(r"\d+")


def norm_pmid(x): return "".join(PMID_ONLY.findall(str(x or "")))


def norm_pmcid(x):
    if not x: return None
    m = PMCID_RX.search(str(x))
    return f"PMC{m.group(1)}" if m else None


def autodelim(p: Path):
    head = p.read_text(encoding="utf-8", errors="ignore").splitlines()[0]
    return "\t" if ("\t" in head and "," not in head) else ","


def extract_ids_from_url(u):
    pmid = None;
    pmcid = None
    m = PUBMED_RX.search(u or "")
    if m: pmid = norm_pmid(m.group(1))
    pmcid = norm_pmcid(u)
    return pmid, pmcid


def idconv(pmids):
    if not pmids: return {"records": []}
    params = {"ids": ",".join(pmids)}
    r = requests.get(IDCONV, params=params, headers={"User-Agent": UA}, timeout=20)
    r.raise_for_status()
    return r.json()


def oai_probe(pmcid):
    if not pmcid: return ("no_pmcid", None)
    base = "https://www.ncbi.nlm.nih.gov/pmc/oai/oai.cgi"
    p = {"verb": "GetRecord", "metadataPrefix": "oai", "identifier": f"oai:pubmedcentral.nih.gov:{pmcid[3:]}"}
    try:
        r = requests.get(base, params=p, headers={"User-Agent": UA}, timeout=10)
        if r.status_code == 200 and "<GetRecord>" in r.text:
            return ("ok", 200)
        return ("low_conf", r.status_code)
    except Exception:
        return ("error", None)


def read_urls_csv(p: Path):
    rows = list(csv.DictReader(open(p, encoding="utf-8")))
    if not {"id", "original_article_url"}.issubset(rows[0].keys()):
        raise SystemExit("[urls] need columns id,original_article_url")
    items = []
    for r in rows:
        pid = str(r["id"]).strip()
        url = r["original_article_url"]
        pmid, pmcid_hint = extract_ids_from_url(url)
        pubmed_url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else None
        items.append({"protocol_id": pid, "url": url, "pmid": pmid, "pmcid_hint": pmcid_hint, "pubmed_url": pubmed_url})
    return items


def read_cands_tsv(p: Path):
    delim = autodelim(p)
    items = []
    with open(p, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f, delimiter=delim)
        # 기대 컬럼: protocol_id, status, pick_url, pubmed_url(optional)
        for row in r:
            if (row.get("status") != "ok") or not row.get("pick_url") or not row.get("protocol_id"):
                continue
            pid = row["protocol_id"].strip()
            url = row["pick_url"]
            pmid, pmcid_hint = extract_ids_from_url(url)
            pubmed_url = row.get("pubmed_url") or (f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else None)
            items.append(
                {"protocol_id": pid, "url": url, "pmid": pmid, "pmcid_hint": pmcid_hint, "pubmed_url": pubmed_url})
    if not items:
        print("[warn] no eligible rows from candidates (status!=ok or missing pick_url)", file=sys.stderr)
    return items


def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--urls", help="data/gold/bio_protocol_original_articles.csv")
    src.add_argument("--cands", help="data/gold/biop_original_candidates.csv (or .tsv)")
    ap.add_argument("--out", default="data/gold/pmc_map_from_urls.csv")
    ap.add_argument("--probe-oai", action="store_true")
    args = ap.parse_args()

    if args.urls:
        items = read_urls_csv(Path(args.urls))
    else:
        items = read_cands_tsv(Path(args.cands))

    # idconv batched
    pmid_to_res = {}
    pmids = [x["pmid"] for x in items if x["pmid"]]
    for i in range(0, len(pmids), 200):
        chunk = pmids[i:i + 200]
        try:
            data = idconv(chunk)
            for rec in data.get("records", []):
                pmid = norm_pmid(rec.get("pmid"))
                pmid_to_res[pmid] = {
                    "pmcid": norm_pmcid(rec.get("pmcid")),
                    "doi": rec.get("doi"),
                    "status": rec.get("status"),
                    "reason": rec.get("errmsg")
                }
        except Exception as e:
            for pmid in chunk:
                pmid_to_res[pmid] = {"pmcid": None, "doi": None, "status": "error", "reason": str(e)}
        time.sleep(0.3)

    # write unified output (same schema as legacy pmc_01)
    outp = Path(args.out);
    outp.parent.mkdir(parents=True, exist_ok=True)
    with outp.open("w", newline="", encoding="utf-8") as w:
        fns = ["protocol_id", "original_article_url", "pubmed_url", "pmid", "pmcid", "doi", "status", "reason"]
        cw = csv.DictWriter(w, fieldnames=fns);
        cw.writeheader()
        for it in items:
            pmid = it["pmid"]
            res = pmid_to_res.get(pmid, {}) if pmid else {}
            pmcid = res.get("pmcid") or it["pmcid_hint"]
            doi = res.get("doi")
            status = res.get("status") or ("ok" if pmcid else "no_pmc")
            reason = res.get("reason")
            if args.probe_oai and pmcid and status == "ok":
                oai_status, _ = oai_probe(pmcid)
                if oai_status != "ok":
                    status = "low_conf"
            cw.writerow({
                "protocol_id": it["protocol_id"],
                "original_article_url": it["url"],
                "pubmed_url": it["pubmed_url"],
                "pmid": pmid,
                "pmcid": pmcid,
                "doi": doi,
                "status": status,
                "reason": reason
            })


if __name__ == "__main__":
    main()
