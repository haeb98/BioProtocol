#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pmc_build_gold_pairs_testset_v2.py

Build gold protocol-article pairs for test set, with a separate metrics report.

This script maps bio_protocol entries to article sections by protocol_id and outputs:
- gold_pairs_testset.jsonl: JSON lines of paired data (protocol + one article section text per pair).
- report/raw_data_comparison.csv: CSV file of selected metric values for each pair.

Key features:
1. Each article's sec_text is taken from one actual section in the article (e.g., "Materials and Methods"), not a concatenation of multiple sections.
2. The JSONL output includes protocol_id, pmcid, domain, and nested `bio` and `article` fields (with section_list), plus the chosen sec_text. It does **not** include metrics or details fields.
3. Metrics (keywords similarity, coverage, etc.) are computed using the bio_protocol's keywords, input/materials, and protocol text. These metrics are **not** in the JSONL but are saved in a separate CSV (report/raw_data_comparison.csv).
"""

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

# -------------------------
# Tokenization & similarity
# -------------------------
TOKEN_RX = re.compile(r"[A-Za-z0-9]+", re.I)


def to_tokens(s: str) -> list[str]:
    """Split text into alphanumeric tokens (lowercase)."""
    return TOKEN_RX.findall((s or "").lower())


def to_tset(s: str) -> set[str]:
    """Get set of unique tokens from text."""
    return set(to_tokens(s))


def jaccard_set(A: set[str], B: set[str]) -> float:
    """Compute Jaccard similarity between two token sets."""
    if not A and not B:
        return 1.0
    if not A or not B:
        return 0.0
    return len(A & B) / max(1, len(A | B))


def char_ngrams(s: str, n: int = 3) -> set[str]:
    """Generate character n-grams (default: trigrams) from text."""
    s = (s or "").lower()
    # Normalize spaces and remove non-alphanumeric characters
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"[^a-z0-9 ]+", "", s)
    if len(s) < n:
        return {s} if s else set()
    return {s[i:i + n] for i in range(len(s) - n + 1)}


def jaccard_text(a: str, b: str) -> float:
    """Jaccard similarity between two texts (max of word-level and char-level)."""
    return max(jaccard_set(to_tset(a), to_tset(b)), jaccard_set(char_ngrams(a, 3), char_ngrams(b, 3)))


# -------------------------
# Bio-Protocol helpers
# -------------------------
def get_keywords_list(brec: dict) -> list[str]:
    """Extract a clean list of keywords from a bio-protocol record."""
    kw = brec.get("keywords")
    out = []
    if not kw:
        return out
    if isinstance(kw, list):
        for k in kw:
            if isinstance(k, str) and k.strip():
                out.append(k.strip())
            elif isinstance(k, dict):
                # Some records store keywords in dicts
                v = (k.get("keyword") or k.get("name") or "").strip()
                if v:
                    out.append(v)
    elif isinstance(kw, str):
        # Split keyword string by common delimiters
        parts = re.split(r"[;,]", kw)
        out = [p.strip() for p in parts if p.strip()]
    # Deduplicate keywords (case-insensitive)
    seen = set()
    res = []
    for k in out:
        lk = k.lower()
        if lk not in seen:
            seen.add(lk)
            res.append(k)
    return res


def get_material_candidates(brec: dict) -> list[str]:
    """Get list of material items from the 'input' or 'materials' field of a bio-protocol."""
    src = brec.get("input") or brec.get("materials") or ""
    items = []
    if isinstance(src, list):
        for x in src:
            if isinstance(x, str) and x.strip():
                # Split on commas, newlines, semicolons
                items.extend(re.split(r"[,\n;]", x))
    elif isinstance(src, str) and src.strip():
        items = re.split(r"[,\n;]", src)
    # Clean whitespace and filter trivial tokens
    items = [re.sub(r"\s+", " ", it).strip() for it in items if it and it.strip()]
    ban = {"-", "—", "and", "or"}
    items = [it for it in items if it.lower() not in ban]
    # Deduplicate items (preserve order)
    seen = set()
    res = []
    for it in items:
        lk = it.lower()
        if lk not in seen:
            seen.add(lk)
            res.append(it)
    return res[:100]


# Regex patterns for parameter extraction
STEP_NUMBER_RX = re.compile(r"^\s*\d+\.\s*")  # matches leading step numbers like "1. "
UNIT_RX = re.compile(
    r"\b(\d+(?:\.\d+)?)\s?(mL|ml|µl|μl|L|g|mg|µg|μg|kg|M|mM|nM|µM|μM|%|°C|degC|min|h|hr|hrs|hours|sec|s|rpm|g|xg)\b",
    re.I
)


def get_param_candidates(brec: dict) -> list[str]:
    """Extract list of parameter tokens (e.g., '10 mL', '37°C', '5 min') from protocol steps text."""
    prot_text = brec.get("protocol") or ""
    if not isinstance(prot_text, str) or not prot_text.strip():
        return []
    out = []
    for line in prot_text.splitlines():
        # Remove leading numbering in steps
        line_clean = STEP_NUMBER_RX.sub("", line)
        for m in UNIT_RX.finditer(line_clean):
            out.append(m.group(0).strip())
    # Deduplicate parameter items (preserve order)
    seen = set()
    res = []
    for item in out:
        li = item.lower()
        if li not in seen:
            seen.add(li)
            res.append(item)
    return res[:150]


# -------------------------
# Coverage metrics (strict & soft)
# -------------------------
def contains_word_boundary(needle: str, hay: str) -> bool:
    """
    Check if `needle` appears in `hay` as a whole word (if needle is a single token)
    or as a substring (if needle has multiple tokens).
    """
    ndl = (needle or "").strip().lower()
    if not ndl:
        return False
    hay_l = (hay or "").lower()
    # If needle is one word, require word boundaries; if multi-word, substring check is sufficient
    if " " not in ndl:
        return re.search(rf"\b{re.escape(ndl)}\b", hay_l) is not None
    return ndl in hay_l


def fraction_contains_strict(cands: list[str], text: str) -> tuple[float, int, int]:
    """
    Calculate strict coverage: fraction of candidate strings that appear in the text (using word-boundary checks).
    Returns (fraction, hits, total).
    """
    if not cands:
        return (0.0, 0, 0)
    hits = 0
    for c in cands:
        if contains_word_boundary(c, text):
            hits += 1
    return (hits / len(cands), hits, len(cands))


def fraction_contains_soft(cands: list[str], text: str, thresh: float = 0.6) -> tuple[float, int, int]:
    """
    Calculate soft coverage: fraction of candidates where at least `thresh` portion of its tokens appear in the text.
    Each candidate string is tokenized and compared to text tokens (order and adjacency not required).
    """
    if not cands:
        return (0.0, 0, 0)
    text_tokens = to_tset(text)
    hits = 0
    for c in cands:
        toks = {t for t in to_tokens(c) if t}
        if not toks:
            continue
        # Calculate coverage of candidate's tokens present in text tokens
        coverage = len(toks & text_tokens) / len(toks)
        if coverage >= thresh:
            hits += 1
    return (hits / len(cands), hits, len(cands))


def normalize_unit_token(s: str) -> str:
    """Normalize unit token for regex matching (unify micro symbols, degree symbols, etc.)."""
    s = (s or "").strip()
    s = s.replace("μl", "µl").replace("μg", "µg")
    s = s.replace("℃", "°c").replace("° C", "°c").replace("°C", "°c").replace("degc", "°c")
    s = re.sub(r"\s+", " ", s)
    return s


def fraction_params_strict(cands: list[str], text: str) -> tuple[float, int, int]:
    """
    Calculate strict coverage for parameter tokens: fraction of parameter strings that can be found in text (allowing minor variations).
    Uses a tolerant regex pattern for matching (ignoring spaces, hyphens, and minor unit differences).
    """
    if not cands:
        return (0.0, 0, 0)
    hay = text or ""
    hits = 0
    for c in cands:
        token = normalize_unit_token(c.lower())
        # Build flexible regex: allow optional spaces/hyphens, and handle °C variants
        token = token.replace(" ", r"\s*").replace("-", r"[-\s]*").replace("°c", r"(?:°\s?c|degc)")
        pattern = fr"(?<!\d){token}(?!\d)"
        if re.search(pattern, hay, flags=re.I):
            hits += 1
    return (hits / len(cands), hits, len(cands))


# -------------------------
# Data loading functions
# -------------------------
def read_ids_csv(path: Path) -> set[str]:
    """Read the set of protocol IDs from a CSV file (expects a column named 'protocol_id')."""
    ids = set()
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            pid = (row.get("protocol_id") or "").strip()
            if pid:
                ids.add(pid)
    return ids


def load_bio(path: Path) -> dict:
    """Load bio_protocol JSON file and index it by protocol_id."""
    data = json.loads(path.read_text(encoding="utf-8"))
    index = {}
    for rec in data:
        pid = str(rec.get("protocol_id") or rec.get("id") or "").strip()
        if pid:
            index[pid] = rec
    return index


def load_articles(path: Path) -> dict:
    """Load the grouped article sections JSONL and group articles by protocol_id."""
    arts_by_pid = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                # Skip any malformed JSON lines
                continue
            pid = rec.get("protocol_id")
            if not pid:
                continue
            arts_by_pid.setdefault(pid, []).append(rec)
    return arts_by_pid


# -------------------------
# Section selection
# -------------------------
def choose_section(article_rec: dict, prefer_regular: bool = True) -> tuple[str, str]:
    """
    Choose one section from the article to use as sec_text.
    Returns (section_text, section_title).
    If prefer_regular is True, sections marked as source "heuristic" will be ignored unless no regular sections are available.
    """
    sections = article_rec.get("sections") or {}
    stats = article_rec.get("stats") or {}
    # Filter out heuristic sections if requested
    available_sections = []
    for title, text in sections.items():
        source = (stats.get(title) or {}).get("source", "")
        if prefer_regular and source == "heuristic":
            continue
        available_sections.append((title, text))
    if not available_sections:
        # If all sections were heuristic (or no stats info), use all sections
        available_sections = list(sections.items())
    if not available_sections:
        return ("", "")  # no section available
    # Prioritize sections likely to contain protocol steps
    priority_sections = []
    other_sections = []
    for title, text in available_sections:
        t_low = title.lower()
        if any(key in t_low for key in ["method", "procedure", "protocol", "experiment"]):
            priority_sections.append((title, text))
        else:
            other_sections.append((title, text))
    # Choose the longest section among priority sections if available, otherwise among other sections
    if priority_sections:
        title, text = max(priority_sections, key=lambda x: len(x[1] or ""))
    else:
        title, text = max(other_sections, key=lambda x: len(x[1] or ""))
    return (text or "", title or "")


# -------------------------
# Main script functionality
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bio", required=True, help="Path to bio_protocol.json file")
    parser.add_argument("--ids", required=True, help="Path to CSV file containing protocol_id column (test set IDs)")
    parser.add_argument("--arts", required=True, help="Path to grouped article sections JSONL file")
    parser.add_argument("--out", required=True, help="Path to output JSONL (gold_pairs_testset.jsonl)")
    parser.add_argument("--report", default="report/raw_data_comparison.csv", help="Path to output CSV for metrics")
    parser.add_argument("--prefer-regular", action="store_true",
                        help="Prefer actual labeled sections over heuristic ones if possible")
    parser.add_argument("--embed-model", default="sentence-transformers/all-mpnet-base-v2",
                        help="SentenceTransformer model name for embedding-based similarity")
    args = parser.parse_args()

    # Load input data
    ids = read_ids_csv(Path(args.ids))
    bio_index = load_bio(Path(args.bio))
    arts_by_pid = load_articles(Path(args.arts))

    # Initialize embedding model for semantic similarity (if available)
    model = None
    if SentenceTransformer:
        try:
            model = SentenceTransformer(args.embed_model)
        except Exception as e:
            print(f"Error loading embedding model '{args.embed_model}': {e}", file=sys.stderr)
            sys.exit(1)
    else:
        print("Warning: sentence_transformers not installed. 'protocol_sim_embedding' will be set to 0.",
              file=sys.stderr)

    # Open output files
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out_file = open(args.out, "w", encoding="utf-8")
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    metrics_file = open(args.report, "w", newline="", encoding="utf-8")
    # Define CSV columns for metrics
    fieldnames = [
        "protocol_id", "pmcid",
        "keywords_sim", "keywords_hits", "keywords_total",
        "material_cov_strict", "material_cov_soft",
        "param_cov_strict", "param_cov_soft",
        "protocol_sim_keyword", "protocol_sim_embedding"
    ]
    writer = csv.DictWriter(metrics_file, fieldnames=fieldnames)
    writer.writeheader()

    # Process each protocol in the test set
    for pid in ids:
        b_rec = bio_index.get(pid)
        if not b_rec:
            continue  # skip if protocol not found in bio_protocol
        articles = arts_by_pid.get(pid, [])
        if not articles:
            continue  # skip if no article sections for this protocol
        # Select the article whose title best matches the Bio-Protocol title
        best_article = None
        best_title_sim = -1.0
        for a in articles:
            sim = jaccard_text(b_rec.get("title") or "", a.get("title") or "")
            if sim > best_title_sim:
                best_title_sim = sim
                best_article = a
        if not best_article:
            continue
        a_rec = best_article

        # Choose one section text from the article
        sec_text, sec_title = choose_section(a_rec, prefer_regular=args.prefer_regular)

        # Compute metrics for this protocol-article pair
        # 1. Keywords similarity
        keywords = get_keywords_list(b_rec)
        if keywords:
            hits = sum(1 for kw in keywords if contains_word_boundary(kw, sec_text))
            keywords_hits = hits
            keywords_total = len(keywords)
            keywords_sim = hits / len(keywords)
        else:
            keywords_sim = None
            keywords_hits = 0
            keywords_total = 0

        # 2. Materials coverage (strict and soft)
        mat_items = get_material_candidates(b_rec)
        strict_mat_frac, strict_mat_hits, strict_mat_total = fraction_contains_strict(mat_items, sec_text)
        soft_mat_frac, soft_mat_hits, _ = fraction_contains_soft(mat_items, sec_text)

        # 3. Parameter coverage (strict and soft)
        param_items = get_param_candidates(b_rec)
        strict_param_frac, strict_param_hits, strict_param_total = fraction_params_strict(param_items, sec_text)
        soft_param_frac, soft_param_hits, _ = fraction_contains_soft(param_items, sec_text)

        # 4. Protocol similarity
        # Keyword-based: Jaccard similarity between full protocol text and section text
        prot_text = b_rec.get("protocol") or ""
        protocol_sim_keyword = jaccard_text(prot_text, sec_text)
        # Embedding-based: cosine similarity between protocol text and section text embeddings
        if model:
            embeddings = model.encode([prot_text, sec_text], show_progress_bar=False)
            vec1, vec2 = embeddings[0], embeddings[1]
            # Compute cosine similarity
            if np.linalg.norm(vec1) == 0 or np.linalg.norm(vec2) == 0:
                protocol_sim_embedding = 0.0
            else:
                protocol_sim_embedding = float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))
        else:
            protocol_sim_embedding = 0.0

        # Write metrics to CSV (round float values to 4 decimal places)
        writer.writerow({
            "protocol_id": pid,
            "pmcid": a_rec.get("pmcid"),
            "keywords_sim": "" if keywords_sim is None else f"{keywords_sim:.4f}",
            "keywords_hits": keywords_hits,
            "keywords_total": keywords_total,
            "material_cov_strict": f"{strict_mat_frac:.4f}",
            "material_cov_soft": f"{soft_mat_frac:.4f}",
            "param_cov_strict": f"{strict_param_frac:.4f}",
            "param_cov_soft": f"{soft_param_frac:.4f}",
            "protocol_sim_keyword": f"{protocol_sim_keyword:.4f}",
            "protocol_sim_embedding": f"{protocol_sim_embedding:.4f}"
        })

        # Prepare JSONL output object for this pair
        out_obj = {
            "protocol_id": pid,
            "pmcid": a_rec.get("pmcid"),
            "domain": a_rec.get("domain") or b_rec.get("classification", {}).get("primary_domain"),
            "bio": {
                "title": b_rec.get("title"),
                "keywords": get_keywords_list(b_rec),
                "hierarchical_protocol": b_rec.get("hierarchical_protocol")
            },
            "article": {
                "title": a_rec.get("title"),
                "meta": a_rec.get("meta", {}),
                "section_list": [sec_title] if sec_title else []
            },
            "sec_text": sec_text
        }
        out_file.write(json.dumps(out_obj, ensure_ascii=False) + "\n")

    # Close output files
    out_file.close()
    metrics_file.close()
    print(f"[DONE] Generated {args.out} and {args.report}")


if __name__ == "__main__":
    main()
