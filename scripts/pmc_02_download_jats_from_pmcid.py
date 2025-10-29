# scripts/pmc_02_download_jats_from_pmcid.py
import csv
import os
import pathlib
import requests
import shutil
import tempfile
import time

from requests.adapters import HTTPAdapter
from tqdm import tqdm
from urllib3.util.retry import Retry

ROOT = pathlib.Path(".")
MAP = ROOT / "data/gold/pmc_map_from_urls.csv"  # 이전 단계 출력
OUTD = ROOT / "data/gold/pmc_jats"
OUTD.mkdir(parents=True, exist_ok=True)

# E-utilities EFetch: PMCID → JATS XML
EFETCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

# 설정(환경변수로 조절 가능)
HTTP_TIMEOUT = float(os.environ.get("HTTP_TIMEOUT", "60"))
SLEEP = float(os.environ.get("EFETCH_SLEEP", "0.34"))  # E-utilities rate
RESUME = os.environ.get("RESUME", "1") == "1"  # 기본 재개
API_KEY = os.environ.get("NCBI_API_KEY", "")  # 있으면 쿼터↑


def make_session():
    sess = requests.Session()
    retry = Retry(
        total=6, read=6, connect=6, backoff_factor=1.0,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"]
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=10, pool_maxsize=10)
    sess.mount("https://", adapter);
    sess.mount("http://", adapter)
    return sess


def main():
    rows = list(csv.DictReader(open(MAP, "r", encoding="utf-8")))
    targets = [r for r in rows if r.get("pmcid")]
    if not targets:
        raise SystemExit("No PMCID rows in pmc_map_from_urls.csv")

    sess = make_session()
    ok = 0;
    sk = 0;
    er = 0
    for r in tqdm(targets, desc="efetch-jats"):
        pmcid = r["pmcid"]  # e.g., PMC1234567
        outp = OUTD / f"{pmcid}.xml"
        if RESUME and outp.exists():
            sk += 1
            continue
        # EFetch: db=pmc, id=PMCxxxxxx, rettype=xml (default)
        params = {"db": "pmc", "id": pmcid}
        if API_KEY: params["api_key"] = API_KEY
        try:
            res = sess.get(EFETCH, params=params, timeout=HTTP_TIMEOUT)
            res.raise_for_status()
            # 간혹 HTML 에러 페이지가 올 수 있으니 XML 태그 존재성만 얕게 체크
            content = res.content
            if b"<pmc-articleset" not in content and b"<article" not in content:
                # 드물게 gzip 전송, 혹은 임시 오류가 있을 수 있음 → 한 번 더 딜레이 후 재요청
                time.sleep(2.0)
                res2 = sess.get(EFETCH, params=params, timeout=HTTP_TIMEOUT)
                res2.raise_for_status()
                content = res2.content
            # 원자적으로 저장
            tmpfd, tmppath = tempfile.mkstemp(prefix="pmc_", suffix=".xml", dir=str(OUTD))
            os.close(tmpfd)
            with open(tmppath, "wb") as f:
                f.write(content)
            shutil.move(tmppath, outp)
            ok += 1
        except requests.RequestException as e:
            er += 1
            # 에러 케이스는 로그만 남기고 계속 진행
            print(f"[WARN] efetch fail {pmcid}: {e}")
        time.sleep(SLEEP)
    print(f"[OK] saved XMLs -> {OUTD} | ok={ok}, skipped={sk}, errors={er}")


if __name__ == "__main__":
    main()
