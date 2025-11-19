#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
enhanced_cov_verifier.py
========================

This script extends the chain‑of‑verification (CoV) parameter checker
from ``eval/verify_10_cov_on_ir.py`` by introducing two additional
features to improve recall without overfitting:

1. **Value/Unit Normalisation with Space Removal:** Numeric values
   followed by a micro prefix often appear with inconsistent spacing
   (e.g. ``5 μg`` vs ``5μg``).  The helper functions here remove
   whitespace between digits and micro characters and unify various
   micro symbols to Greek mu (``μ``).  Multiplication signs (``×``)
   are also converted to ASCII ``x``.  These changes are applied both
   to the IR parameters and to the Methods text before searching for
   evidence.

2. **Context Sentence Extraction:** Instead of passing the entire
   Methods section to the LLM, the verifier extracts a small number of
   sentences containing the parameter name (case‑insensitive).  If
   relevant sentences are found they form the context; otherwise the
   full Methods text (with whitespace normalisation) is used.  This
   reduces the likelihood that long Methods sections exceed the model
   context window and increases the chance the model focuses on
   relevant evidence.

The remainder of the logic—constructing the prompt, calling the
OpenAI API, parsing the JSON result, and writing the output—remains
close to the original verifier.  You can control the number of
sentences extracted via the ``--max-hits`` command line option; set it
to 0 to fall back to full Methods every time.

Example usage:

::

    python enhanced_cov_verifier.py \
        --gold data/gold/gold_pairs_testset.jsonl \
        --ir runs/s2_parser.ir.jsonl \
        --out runs/enhanced_cov_results.jsonl \
        --model gpt-3.5-turbo \
        --max-hits 3

The output JSONL will contain one object per parameter with verdict,
evidence span, and the original LLM response.
"""

import argparse
import json
import os
import re
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from openai import OpenAI  # type: ignore
except Exception:
    OpenAI = None  # type: ignore


@dataclass
class ParamRecord:
    """Container for a single parameter extracted from an IR node."""
    protocol_id: str
    node_id: str
    param_index: int
    name: Optional[str]
    value: Any
    unit: Optional[str]


def load_gold_pairs(path: Path) -> Dict[str, str]:
    """Load gold pairs into a mapping of protocol_id -> Methods text."""
    mapping: Dict[str, str] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
            except Exception:
                continue
            pid = obj.get("protocol_id")
            if not pid:
                continue
            sec_text = obj.get("sec_text") or obj.get("text")
            if not sec_text:
                article = obj.get("article") or {}
                sec_text = article.get("sec_text") or article.get("text")
            if not sec_text:
                continue
            mapping[str(pid)] = sec_text
    return mapping


def load_ir(path: Path) -> List[Dict[str, Any]]:
    """Load IR records from JSONL."""
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if isinstance(obj, dict):
                records.append(obj)
    return records


def extract_params_from_ir(ir_record: Dict[str, Any]) -> List[ParamRecord]:
    """Flatten parameters in IR into a list of ParamRecord."""
    protocol_id = str(ir_record.get("protocol_id", ""))
    params: List[ParamRecord] = []
    for node in ir_record.get("nodes", []):
        node_id = node.get("id", "UNKNOWN")
        node_params = node.get("params") or []
        if not isinstance(node_params, list):
            node_params = [node_params]
        for idx, p in enumerate(node_params):
            if not isinstance(p, dict):
                p = {"name": None, "value": None, "unit": None}
            params.append(
                ParamRecord(
                    protocol_id=protocol_id,
                    node_id=node_id,
                    param_index=idx,
                    name=p.get("name"),
                    value=p.get("value"),
                    unit=p.get("unit"),
                )
            )
    return params


def unify_chars(s: str) -> str:
    """Unify micro and multiplication characters within a string."""
    return s.replace("×", "x").replace("µ", "μ")


def remove_space_before_micro(text: str) -> str:
    """
    Remove whitespace between digits and a micro symbol (μ or µ).

    For example, ``5 μg`` becomes ``5μg``.  This helps the LLM match
    units even when there are spacing differences.
    """
    # unify both micro symbols first
    text = unify_chars(text)
    # remove spaces between digit and mu
    return re.sub(r"(\d)\s+(μ)", r"\1\2", text)


def normalize_param(name: Optional[str], value: Any, unit: Optional[str]) -> Tuple[Optional[str], Any, Optional[str]]:
    """
    Normalise value/unit fields for the LLM.  Converts micro and multiplication
    characters to canonical forms and removes spaces between digits and
    micro prefixes.  Ratios (1:500) and multipliers (2x) are preserved
    as strings.
    """
    new_unit: Optional[str] = None
    if unit is not None:
        try:
            u = unify_chars(str(unit))
            new_unit = u.strip() or None
        except Exception:
            new_unit = unit
    new_value: Any = value
    if isinstance(value, str):
        try:
            v = unify_chars(value)
            # remove whitespace between digits and μ
            v = re.sub(r"(\d)\s+(μ)", r"\1\2", v)
            new_value = v
        except Exception:
            new_value = value
    return name, new_value, new_unit


def extract_sentences(text: str) -> List[str]:
    """
    Split a text into sentences using simple punctuation heuristics.
    Returns a list of trimmed sentences.
    """
    raw = (text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if not raw:
        return []
    out: List[str] = []
    cur: List[str] = []
    for tok in re.split(r"([.!?])", raw):
        if tok is None:
            continue
        cur.append(tok)
        if tok in (".", "?", "!"):
            s = "".join(cur).strip()
            if s:
                out.append(s)
            cur = []
    rest = "".join(cur).strip()
    if rest:
        out.append(rest)
    return [s for s in out if s]


def find_relevant_sentences(sentences: List[str], param_name: Optional[str], value: Any, unit: Optional[str], max_hits: int) -> List[str]:
    """
    Find up to ``max_hits`` sentences containing the parameter name (case insensitive).
    If ``param_name`` is None or empty, returns an empty list.  The value
    and unit are not currently used for matching but could be added later.
    """
    if max_hits <= 0:
        return []
    if not param_name:
        return []
    name = param_name.strip()
    if not name:
        return []
    # unify characters in search term to match processed sentences
    name_pattern = re.escape(unify_chars(name))
    hits: List[str] = []
    for s in sentences:
        s_norm = unify_chars(remove_space_before_micro(s))
        if re.search(name_pattern, s_norm, flags=re.IGNORECASE):
            hits.append(s)
            if len(hits) >= max_hits:
                break
    return hits


def build_verification_prompt(methods_text: str, param: ParamRecord, max_hits: int) -> str:
    """
    Build the prompt for the LLM.  Attempts to extract relevant sentences
    containing the parameter name up to ``max_hits``.  If no sentences
    are found or ``max_hits == 0``, uses the full Methods text (after
    removing extra spaces before micro prefixes).
    """
    # normalise the parameter fields for display
    name, value, unit = normalize_param(param.name, param.value, param.unit)
    # prepare Methods text with whitespace removal around μ
    processed_methods = remove_space_before_micro(methods_text)
    sentences = extract_sentences(processed_methods)
    context_sentences = find_relevant_sentences(sentences, name, value, unit, max_hits)
    if context_sentences:
        context_text = "\n".join(context_sentences)
    else:
        # fallback: use entire processed methods (up to some limit)
        context_text = processed_methods
    system_instructions = (
        "You are a strict scientific protocol verifier. "
        "You only trust information that is explicitly present in the given Methods section. "
        "Do NOT guess or infer plausible values. "
        "If the parameter is not clearly supported by the text, mark it as 'unsupported' or 'ambiguous'."
    )
    user_content = f"""
[Methods Section]
-----------------
{context_text}
-----------------

[Parameter to verify]
- name: {name}
- value: {value}
- unit: {unit}

Task:
1. Check if the given parameter (name + value + unit) is explicitly or nearly‑exactly supported by the Methods text.
2. If yes, set "verdict" to "supported" and copy the most relevant sentence (or short span) from the Methods into "evidence_span".
3. If the name is mentioned but the value/unit differ or are unclear, set "verdict" to "ambiguous" and still provide the best evidence_span.
4. If the parameter cannot be found or is clearly not supported, set "verdict" to "unsupported" and set evidence_span to "".

Output a JSON object with the following keys only:
- verdict: one of ["supported", "ambiguous", "unsupported"]
- evidence_span: string
"""
    return f"{system_instructions}\n\nNow answer in JSON only.\n" + user_content.strip()


def call_llm(prompt: str, client: Any, model: str, max_retries: int = 2) -> str:
    """Call the OpenAI API and return the raw JSON string response."""
    last_err: Optional[Exception] = None
    for _ in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            return resp.choices[0].message.content
        except Exception as e:
            last_err = e
            time.sleep(1.2)
    return json.dumps({"verdict": "ambiguous", "evidence_span": ""})


def parse_llm_verdict(raw_output: str) -> Dict[str, Any]:
    """Parse the LLM's JSON output safely."""
    raw_output = raw_output.strip()
    try:
        obj = json.loads(raw_output)
        verdict = obj.get("verdict", "ambiguous")
        evidence_span = obj.get("evidence_span", "")
    except Exception:
        verdict = "ambiguous"
        evidence_span = ""
    if verdict not in ("supported", "ambiguous", "unsupported"):
        verdict = "ambiguous"
    return {
        "verdict": verdict,
        "evidence_span": evidence_span,
        "llm_raw": raw_output,
    }


def run_verification(
        gold_pairs_path: Path,
        ir_path: Path,
        out_path: Path,
        client: Any,
        model: str,
        max_hits: int,
        max_retries: int = 2,
) -> None:
    """
    Run the enhanced CoV verification across all IR records and write the
    results to ``out_path``.
    """
    print(f"[INFO] Loading gold pairs from {gold_pairs_path}")
    protocol_to_methods = load_gold_pairs(gold_pairs_path)
    print(f"[INFO] Loaded Methods for {len(protocol_to_methods)} protocols.")
    print(f"[INFO] Loading IR records from {ir_path}")
    ir_records = load_ir(ir_path)
    print(f"[INFO] Loaded {len(ir_records)} IR records.")
    out_f = out_path.open("w", encoding="utf-8")
    total_params = 0
    verdict_counts = {"supported": 0, "ambiguous": 0, "unsupported": 0}
    for ir in ir_records:
        pid = ir.get("protocol_id")
        methods_text = protocol_to_methods.get(str(pid))
        if not methods_text:
            continue
        param_records = extract_params_from_ir(ir)
        for p in param_records:
            total_params += 1
            prompt = build_verification_prompt(methods_text, p, max_hits)
            raw = call_llm(prompt, client, model, max_retries=max_retries)
            verdict_info = parse_llm_verdict(raw)
            v = verdict_info["verdict"]
            verdict_counts[v] = verdict_counts.get(v, 0) + 1
            out_obj = {
                **asdict(p),
                "verdict": verdict_info["verdict"],
                "evidence_span": verdict_info["evidence_span"],
                "llm_raw": verdict_info["llm_raw"],
            }
            out_f.write(json.dumps(out_obj, ensure_ascii=False) + "\n")
    out_f.close()
    print("[INFO] Done.")
    print(f"  total params: {total_params}")
    for k, v in verdict_counts.items():
        rate = v / total_params if total_params > 0 else 0.0
        print(f"  {k}: {v} ({rate:.3f})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Enhanced CoV verifier with context extraction and normalisation")
    parser.add_argument("--gold", type=str, required=True, help="Path to gold_pairs_testset.jsonl")
    parser.add_argument("--ir", type=str, required=True, help="Path to s2_parser.ir.jsonl")
    parser.add_argument("--out", type=str, required=True, help="Output path for verification results (JSONL)")
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo", help="OpenAI chat model to use")
    parser.add_argument("--max-hits", type=int, default=3, help="Maximum number of sentences to extract as context (0 disables extraction)")
    parser.add_argument("--max-retries", type=int, default=2, help="Number of retries when calling the OpenAI API")
    args = parser.parse_args()
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("ERROR: OPENAI_API_KEY environment variable is not set. Please set it before running this script.")
    if OpenAI is None:
        raise SystemExit("ERROR: The openai package is not installed. Please install it with 'pip install openai'.")
    client = OpenAI()
    run_verification(
        gold_pairs_path=Path(args.gold),
        ir_path=Path(args.ir),
        out_path=Path(args.out),
        client=client,
        model=args.model,
        max_hits=args.max_hits,
        max_retries=args.max_retries,
    )


if __name__ == "__main__":
    main()