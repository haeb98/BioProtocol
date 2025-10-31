#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
report_05_sections_stats.py

Input
  --arts data/gold/gold_articles_sections_pmc.jsonl   # pmc_03 결과
Output
  reports/sections_summary_overall.csv
  reports/sections_summary_by_domain.csv
  reports/sections_titles_top.csv                     # 등장 섹션명 TOP N
  reports/sections_sources_by_article.csv             # 기사별 섹션 소스 요약

Tip: 이 CSV들을 그대로 슬라이드에 넣거나, 엑셀/구글시트로 차트 만들기 좋음.
"""

import argparse
import collections
import json
import pathlib


def load_jsonl(path):
    arr = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                arr.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return arr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arts", required=True)
    ap.add_argument("--outdir", default="reports")
    ap.add_argument("--topn", type=int, default=15, help="상위 섹션명 개수")
    args = ap.parse_args()

    outd = pathlib.Path(args.outdir);
    outd.mkdir(parents=True, exist_ok=True)
    rows = load_jsonl(args.arts)

    # 집계 변수들
    n_articles = len(rows)
    total_sections = 0
    total_tokens = 0
    total_chars = 0

    by_domain = collections.defaultdict(lambda: {
        "articles": 0, "sections": 0, "tokens": 0, "chars": 0,
        "src_regular": 0, "src_merged": 0, "src_heuristic": 0
    })
    section_title_counter = collections.Counter()

    # 기사별 소스 비중 CSV
    art_src_lines = [["protocol_id", "pmcid", "domain", "section_count", "sources_list"]]

    for r in rows:
        pid = r.get("protocol_id", "")
        pmcid = r.get("pmcid", "")
        dom = (r.get("domain") or "Unknown")
        secs = r.get("sections") or {}
        stats = r.get("stats") or {}

        # 기사 단위
        by_domain[dom]["articles"] += 1

        # 섹션 단위
        scount = 0
        src_kinds = []
        for title, text in secs.items():
            st = stats.get(title, {})
            total_sections += 1;
            scount += 1
            section_title_counter[title] += 1

            tok = int(st.get("tokens", len((text or "").split())))
            ch = int(st.get("chars", len(text or "")))
            total_tokens += tok;
            total_chars += ch
            by_domain[dom]["sections"] += 1
            by_domain[dom]["tokens"] += tok
            by_domain[dom]["chars"] += ch

            src = st.get("source", "")
            if src == "regular":
                by_domain[dom]["src_regular"] += 1
            elif src == "merged":
                by_domain[dom]["src_merged"] += 1
            elif src == "heuristic":
                by_domain[dom]["src_heuristic"] += 1
            src_kinds.append(src or "unknown")

        art_src_lines.append([pid, pmcid, dom, scount, "|".join(src_kinds)])

    # 전체 요약 CSV
    overall = [["metric", "value"]]
    overall += [
        ["articles", n_articles],
        ["sections", total_sections],
        ["tokens_total", total_tokens],
        ["chars_total", total_chars],
        ["sections_per_article_avg", round(total_sections / max(1, n_articles), 2)],
        ["tokens_per_article_avg", round(total_tokens / max(1, n_articles), 2)],
        ["chars_per_article_avg", round(total_chars / max(1, n_articles), 2)],
    ]

    # 도메인별 요약 CSV
    by_dom_csv = [["domain", "articles", "sections", "tokens_total", "chars_total",
                   "sections_per_article", "tokens_per_article", "chars_per_article",
                   "src_regular", "src_merged", "src_heuristic"]]
    for d, ag in sorted(by_domain.items(), key=lambda x: (-x[1]["articles"], x[0])):
        a = ag["articles"] or 1
        by_dom_csv.append([
            d, ag["articles"], ag["sections"], ag["tokens"], ag["chars"],
            round(ag["sections"] / a, 2), round(ag["tokens"] / a, 2), round(ag["chars"] / a, 2),
            ag["src_regular"], ag["src_merged"], ag["src_heuristic"]
        ])

    # 섹션명 TOP N
    titles_top = [["section_title", "count"]]
    for t, c in section_title_counter.most_common(args.topn):
        titles_top.append([t, c])

    # 저장
    (outd / "sections_summary_overall.csv").write_text("\n".join([",".join(map(str, row)) for row in overall]),
                                                       encoding="utf-8")
    (outd / "sections_summary_by_domain.csv").write_text("\n".join([",".join(map(str, row)) for row in by_dom_csv]),
                                                         encoding="utf-8")
    (outd / "sections_titles_top.csv").write_text("\n".join([",".join(map(str, row)) for row in titles_top]),
                                                  encoding="utf-8")
    (outd / "sections_sources_by_article.csv").write_text("\n".join([",".join(map(str, row)) for row in art_src_lines]),
                                                          encoding="utf-8")

    print(f"[OK] wrote -> {outd / 'sections_summary_overall.csv'}")
    print(f"[OK] wrote -> {outd / 'sections_summary_by_domain.csv'}")
    print(f"[OK] wrote -> {outd / 'sections_titles_top.csv'}")
    print(f"[OK] wrote -> {outd / 'sections_sources_by_article.csv'}")


if __name__ == "__main__":
    main()
