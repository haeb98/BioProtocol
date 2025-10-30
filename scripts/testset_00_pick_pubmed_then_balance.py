# scripts/testset_00_pick_pubmed_then_balance.py
import argparse
import csv
import orjson
import pathlib
import random
import sys
from collections import Counter, defaultdict
from urllib.parse import urlparse

ROOT = pathlib.Path(".")
BIO = ROOT / "data/raw/bio_protocol.json"
URLS = ROOT / "data/gold/bio_protocol_original_articles.csv"
OUT_IDS = ROOT / "data/splits/test_biop_ids.csv"
OUT_DCNT = ROOT / "data/splits/domain_counts_pubmed.csv"

PUBMED_HOSTS = {"ncbi.nlm.nih.gov", "pubmed.ncbi.nlm.nih.gov"}


def is_pubmedish(u: str) -> bool:
    if not u: return False
    try:
        h = urlparse(u).netloc.lower()
    except Exception:
        return False
    return any(host in h for host in PUBMED_HOSTS)


def load_bio_index():
    if not BIO.exists():
        print(f"[ERR] not found: {BIO}", file=sys.stderr);
        sys.exit(1)
    js = orjson.loads(BIO.read_bytes())
    idx = {}
    for r in js:
        pid = str(r.get("id"))
        dom = (r.get("classification") or {}).get("primary_domain", "Unknown")
        ttl = r.get("title", "").strip()
        idx[pid] = {"domain": dom, "title": ttl}
    return idx


def load_pubmed_candidates(idx):
    if not URLS.exists():
        print(f"[ERR] not found: {URLS}", file=sys.stderr);
        sys.exit(1)
    rows = list(csv.DictReader(open(URLS, "r", encoding="utf-8")))
    cand = []
    for r in rows:
        pid = (r.get("id") or r.get("protocol_id") or "").strip()
        url = (r.get("original_article_url") or r.get("original_url") or "").strip()
        if not pid or not url:
            continue
        if not is_pubmedish(url):
            continue
        meta = idx.get(pid)
        if not meta:
            # id가 bio_protocol.json에 없으면 스킵
            continue
        cand.append({
            "protocol_id": pid,
            "domain": meta["domain"],
            "title": meta["title"],
            "original_article_url": url
        })
    return cand


def main(args):
    random.seed(args.seed)
    OUT_IDS.parent.mkdir(parents=True, exist_ok=True)

    idx = load_bio_index()
    pubmed_cand = load_pubmed_candidates(idx)

    # 도메인 카운트(오직 pubmed 후보 내부에서)
    dcnt = Counter([c["domain"] for c in pubmed_cand])
    # 저장
    with open(OUT_DCNT, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f);
        w.writerow(["domain", "count_pubmed"])
        for d, c in sorted(dcnt.items(), key=lambda x: (-x[1], x[0])):
            w.writerow([d, c])

    # eligible 도메인(≥min_per_domain)
    eligible = {d for d, c in dcnt.items() if c >= args.min_per_domain}
    if not eligible:
        print(f"[WARN] No domain has >= {args.min_per_domain} pubmed entries. "
              f"Lower --min-per-domain or check inputs.", file=sys.stderr)

    # 도메인 버킷 → 균등 샘플링
    buckets = defaultdict(list)
    for c in pubmed_cand:
        if c["domain"] in eligible:
            buckets[c["domain"]].append(c)

    picked = []
    for d, arr in buckets.items():
        random.shuffle(arr)
        take = arr[:args.take_per_domain]
        picked.extend(take)

    # 전체 상한
    if len(picked) > args.max_total:
        picked = picked[:args.max_total]

    # 중복 제거(혹시 모를)
    seen = set();
    dedup = []
    for r in picked:
        if r["protocol_id"] in seen: continue
        seen.add(r["protocol_id"]);
        dedup.append(r)
    picked = dedup

    # 출력
    with open(OUT_IDS, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f);
        w.writerow(["protocol_id", "domain", "title", "original_article_url"])
        for r in picked:
            w.writerow([r["protocol_id"], r["domain"], r["title"], r["original_article_url"]])

    print(f"[OK] wrote test IDs: {OUT_IDS} (rows={len(picked)})")
    print(f"[OK] wrote domain counts: {OUT_DCNT}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--min-per-domain", type=int, default=30, help="pubmed 후보 내부에서 도메인 최소 개수")
    p.add_argument("--take-per-domain", type=int, default=5, help="도메인별 최대 샘플 수")
    p.add_argument("--max-total", type=int, default=100, help="전체 상한")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    main(args)
