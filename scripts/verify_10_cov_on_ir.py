#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
verify_10_cov_on_ir.py
========================

This script performs a chain‑of‑verification (CoV) style check on the
parameters contained in an IR graph (as produced by the S2 parser).  For
each parameter (name/value/unit triple) it queries an LLM to decide
whether the parameter is directly supported by the Methods text
available in the gold testset.  It then writes a JSONL file with the
verdict ("supported", "ambiguous" or "unsupported") and the evidence
span used to justify the decision.

Compared to the initial prototype, this version has been updated to
reflect the actual structure of the input files and to integrate
OpenAI API usage correctly.  Specifically:

* `data/gold/gold_pairs_testset.jsonl` is expected to contain a field
  called ``sec_text`` for each record.  If ``sec_text`` is not
  present, the loader falls back to ``text`` but will skip records
  entirely if no suitable text field is found.
* The script checks for an ``OPENAI_API_KEY`` in the environment.  If
  it is missing, execution aborts with an informative error.  This
  follows the pattern used in the project's other LLM scripts and
  ensures the API key is picked up automatically by the OpenAI client.
* A command line parameter ``--model`` allows you to specify which
  OpenAI chat model to use (defaulting to ``gpt-3.5-turbo``).  The
  OpenAI client is instantiated once and passed into the LLM call
  helper to avoid reconnect overhead.
* The helper ``call_llm`` performs a single OpenAI chat
  completion request with ``temperature=0.0`` and JSON response
  formatting.  If the request fails it returns an ``ambiguous``
  verdict with an empty evidence span rather than raising an
  exception.  You can adjust the ``max_retries`` argument if you
  prefer retry behaviour.

Example usage:

    python verify_10_cov_on_ir.py \
        --gold data/gold/gold_pairs_testset.jsonl \
        --ir runs/s2_parser.ir.jsonl \
        --out runs/verify_cov_params.jsonl \
        --model gpt-3.5-turbo

Ensure that ``OPENAI_API_KEY`` is set in your environment before
running the script.  The output will contain one JSON object per
parameter encountered in the IR, with the fields described below.
"""

import argparse
import json
import os
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

# Third‑party import guarded so the script can be imported without
# immediately requiring openai.  The client is created in the main
# function once the API key has been checked.
try:
    from openai import OpenAI  # type: ignore
except Exception:
    OpenAI = None  # type: ignore


@dataclass
class ParamRecord:
    """
    Container for a single parameter extracted from an IR node.

    Attributes
    ----------
    protocol_id : str
        Identifier of the protocol this parameter belongs to.
    node_id : str
        The ID of the node within the IR graph (e.g. "S1").
    param_index : int
        The index of the parameter within the node's ``params`` list.
    name : Optional[str]
        Name of the parameter (e.g. "Tris-HCl").  May be None.
    value : Any
        Value of the parameter (e.g. 10).  May be None or non‑numeric.
    unit : Optional[str]
        Unit of the parameter (e.g. "mM").  May be None.
    """
    protocol_id: str
    node_id: str
    param_index: int
    name: Optional[str]
    value: Any
    unit: Optional[str]


def load_gold_pairs(path: Path) -> Dict[str, str]:
    """
    Read the gold pairs testset and build a mapping from protocol_id to
    Methods text (``sec_text``).

    Parameters
    ----------
    path : Path
        Path to the ``gold_pairs_testset.jsonl`` file.

    Returns
    -------
    Dict[str, str]
        Mapping from protocol_id to the Methods text.

    Notes
    -----
    The input JSONL file is expected to have at least a ``protocol_id``
    and ``sec_text`` field on each line.  If ``sec_text`` is missing,
    the loader will fall back to a ``text`` field.  Records lacking
    either will be skipped.
    """
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
            # Prefer explicit sec_text.  Fall back to text.
            sec_text = obj.get("sec_text") or obj.get("text")
            if not sec_text:
                # Some older records may nest the text under 'article'.
                article = obj.get("article") or {}
                sec_text = article.get("sec_text") or article.get("text")
            if not sec_text:
                continue
            mapping[str(pid)] = sec_text
    return mapping


def load_ir(path: Path) -> List[Dict[str, Any]]:
    """
    Load an IR jsonl file produced by the S2 parser.

    Parameters
    ----------
    path : Path
        Path to the IR file.

    Returns
    -------
    List[Dict[str, Any]]
        A list of IR records, each as a dict.
    """
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
    """
    Flatten all parameters in the IR record into a list of ParamRecord.

    Parameters
    ----------
    ir_record : Dict[str, Any]
        A single IR record containing nodes and edges.

    Returns
    -------
    List[ParamRecord]
        A list of flattened parameter records.
    """
    protocol_id = str(ir_record.get("protocol_id", ""))
    params: List[ParamRecord] = []
    for node in ir_record.get("nodes", []):
        node_id = node.get("id", "UNKNOWN")
        node_params = node.get("params") or []
        if not isinstance(node_params, list):
            # In case params is a dict or other type, normalize to list
            node_params = [node_params]
        for idx, p in enumerate(node_params):
            if not isinstance(p, dict):
                # If the param is a bare string or other type, coerce into a
                # dict with only raw value.  Name, value and unit will be
                # derived by the LLM during verification.
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


def build_verification_prompt(methods_text: str, param: ParamRecord) -> str:
    """
    Construct a verification prompt for the LLM.

    The prompt asks the LLM to decide whether the given parameter is
    explicitly supported by the provided Methods section.  It follows a
    strict format instructing the model not to guess values.

    Parameters
    ----------
    methods_text : str
        The Methods section of the paper.
    param : ParamRecord
        The parameter to be verified.

    Returns
    -------
    str
        A complete prompt string to be sent to the LLM.
    """
    system_instructions = (
        "You are a strict scientific protocol verifier. "
        "You only trust information that is explicitly present in the given Methods section. "
        "Do NOT guess or infer plausible values. "
        "If the parameter is not clearly supported by the text, mark it as 'unsupported' or 'ambiguous'."
    )
    user_content = f"""
[Methods Section]
-----------------
{methods_text}
-----------------

[Parameter to verify]
- name: {param.name}
- value: {param.value}
- unit: {param.unit}

Task:
1. Check if the given parameter (name + value + unit) is explicitly or nearly‑exactly supported by the Methods text.
2. If yes, set "verdict" to "supported" and copy the most relevant sentence (or short span) from the Methods into "evidence_span".
3. If the name is mentioned but the value/unit differ or are unclear, set "verdict" to "ambiguous" and still provide the best evidence_span.
4. If the parameter cannot be found or is clearly not supported, set "verdict" to "unsupported" and set evidence_span to "".

Output a JSON object with the following keys only:
- verdict: one of ["supported", "ambiguous", "unsupported"]
- evidence_span: string
"""
    return (
            f"{system_instructions}\n\nNow answer in JSON only.\n" + user_content.strip()
    )


def call_llm(prompt: str, client: Any, model: str, max_retries: int = 2) -> str:
    """
    Call the OpenAI chat completions API to obtain a JSON answer.

    Parameters
    ----------
    prompt : str
        The prompt to send to the LLM.
    client : Any
        An instantiated OpenAI client.
    model : str
        The model name to use (e.g. ``gpt-3.5-turbo``).
    max_retries : int, optional
        Number of retries if the API call fails.  Defaults to 2.

    Returns
    -------
    str
        The raw JSON string returned by the LLM.  If the call fails
        repeatedly, it returns a minimal JSON with an ambiguous verdict.
    """
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
            time.sleep(1.2)  # exponential backoff could be added
    # If we get here, return an ambiguous verdict to avoid crashing the loop
    return json.dumps({"verdict": "ambiguous", "evidence_span": ""})


def parse_llm_verdict(raw_output: str) -> Dict[str, Any]:
    """
    Parse the LLM's JSON output into a dictionary with fixed keys.

    Parameters
    ----------
    raw_output : str
        The raw JSON string returned by the LLM.

    Returns
    -------
    Dict[str, Any]
        Parsed verdict containing ``verdict`` and ``evidence_span``.
    """
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
        max_retries: int = 2,
) -> None:
    """
    Execute the full verification pipeline over the IR and write results.

    Parameters
    ----------
    gold_pairs_path : Path
        Path to the gold pairs JSONL file.
    ir_path : Path
        Path to the IR JSONL file.
    out_path : Path
        Where to write the verification results (JSONL).
    client : Any
        Instantiated OpenAI client.
    model : str
        Name of the model to use.
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
            prompt = build_verification_prompt(methods_text, p)
            # Propagate the max_retries argument to the LLM call
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
    parser = argparse.ArgumentParser(
        description="CoV‑style verifier for IR parameters (Experiment #1)"
    )
    parser.add_argument(
        "--gold",
        type=str,
        required=True,
        help="Path to gold_pairs_testset.jsonl",
    )
    parser.add_argument(
        "--ir",
        type=str,
        required=True,
        help="Path to s2_parser.ir.jsonl",
    )
    parser.add_argument(
        "--out",
        type=str,
        required=True,
        help="Output path for verification results (JSONL)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-3.5-turbo",
        help="OpenAI chat model to use (e.g. gpt-3.5-turbo, gpt-4o-mini)",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=2,
        help="Number of retries when calling the OpenAI API",
    )
    args = parser.parse_args()

    # Check for API key
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit(
            "ERROR: OPENAI_API_KEY environment variable is not set. "
            "Please set it before running this script."
        )
    if OpenAI is None:
        raise SystemExit(
            "ERROR: The openai package is not installed. Please install it with 'pip install openai'."
        )

    # Instantiate the client
    client = OpenAI()

    run_verification(
        gold_pairs_path=Path(args.gold),
        ir_path=Path(args.ir),
        out_path=Path(args.out),
        client=client,
        model=args.model,
        max_retries=args.max_retries,
    )


if __name__ == "__main__":
    main()
