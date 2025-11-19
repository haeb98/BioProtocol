"""
extract_gold_tasks_testset.py
---------------------------------

This script extracts high-level tasks from the ``gold_pairs_testset.jsonl`` file
used in the BioProtocol experiments.  Each line in the testset file contains a
``protocol_id`` and a nested ``bio`` field with a ``hierarchical_protocol``
structure.  The top-level numeric keys (e.g. ``"1"``, ``"2"``) of
``hierarchical_protocol`` correspond to the major experimental phases or tasks,
and each such entry typically has a ``title`` field.

The goal of this script is to produce a simplified ground-truth task list for
each protocol in the test set by extracting those top-level titles.  The
resulting JSONL file can be used to evaluate task planners on the same
protocols.

Example usage::

    python extract_gold_tasks_testset.py \
        --pairs data/gold/gold_pairs_testset.jsonl \
        --out data/gold/gold_tasks_testset.jsonl

The output will contain records like::

    {"protocol_id": "Bio-protocol-2096", "tasks": [{"task_id": "T1", "title": "Cell Preparation"}, ...]}

"""

import argparse
import json
from typing import Any, Dict, List


def extract_top_level_tasks(hierarchical_protocol: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract top-level tasks from a hierarchical_protocol structure.

    Only keys without a dot (e.g. ``"1"``, ``"2"``) are treated as tasks.
    Each task must have a ``title`` field; if missing, a placeholder title
    is generated.

    Args:
        hierarchical_protocol: A dictionary representing the hierarchical
            protocol.  Keys like "1", "2" map to dicts with "title".

    Returns:
        A list of task dictionaries with ``task_id`` and ``title``.
    """
    tasks: List[Dict[str, Any]] = []
    if not isinstance(hierarchical_protocol, dict):
        return tasks
    # Filter keys that do not contain a dot (top-level)
    top_keys = [k for k in hierarchical_protocol.keys() if "." not in k]
    # Sort keys numerically if possible to maintain order
    try:
        top_keys = sorted(top_keys, key=lambda x: float(x))
    except ValueError:
        top_keys = sorted(top_keys)
    for idx, key in enumerate(top_keys, start=1):
        entry = hierarchical_protocol.get(key)
        title = None
        if isinstance(entry, dict):
            title = entry.get("title") or entry.get("section_title") or entry.get("heading")
        # If title is missing or not a string, generate a placeholder
        if not title or not isinstance(title, str):
            title = f"Task {idx}"
        tasks.append({"task_id": f"T{idx}", "title": title.strip()})
    return tasks


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract gold tasks from gold_pairs_testset.jsonl")
    parser.add_argument(
        "--pairs",
        required=True,
        help="Path to gold_pairs_testset.jsonl (JSONL file with protocol_id and bio.hierarchical_protocol)",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Output JSONL file for extracted tasks",
    )
    args = parser.parse_args()

    with open(args.pairs, encoding="utf-8") as f_in, open(args.out, "w", encoding="utf-8") as f_out:
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            pid = rec.get("protocol_id") or rec.get("id")
            bio = rec.get("bio")
            if not pid or not bio:
                continue
            hierarchy = bio.get("hierarchical_protocol")
            tasks = extract_top_level_tasks(hierarchy)
            f_out.write(json.dumps({"protocol_id": pid, "tasks": tasks}, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
