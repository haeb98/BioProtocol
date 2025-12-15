# src/data_loader.py
import json
import os
from typing import Dict, Any

from src.types import GraphState

# this file's directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# project root = src/.. (= BioProtocolAgent/)
ROOT_DIR = os.path.dirname(BASE_DIR)

DATA_DIR = os.path.join(ROOT_DIR, "data")


def load_bio_index(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        data = json.load(f)
    return {item["id"]: item for item in data}


def load_pairs_index(path: str) -> Dict[str, Any]:
    idx = {}
    with open(path, "r") as f:
        for line in f:
            rec = json.loads(line)
            pid = rec["protocol_id"]
            idx[pid] = rec
    return idx


# BIO_INDEX = load_bio_index("../data/bio_protocol.json")
PAIR_PATH = os.path.join(DATA_DIR, "gold_pairs_testset_v2.jsonl")
PAIRS_INDEX = load_pairs_index(PAIR_PATH)


def make_initial_state(protocol_id: str) -> GraphState:
    # bio = PAIRS_INDEX[protocol_id]
    pair = PAIRS_INDEX[protocol_id]

    bio = pair["bio"]
    article = pair["article"]
    methods_text = pair["sec_text"]
    pmcid = pair["pmcid"]

    return GraphState(
        protocol_id=protocol_id,
        bio=bio,
        article=article,
        methods_text=methods_text,
        pmcid=pmcid,
        tasks=[],
        steps_raw=[],
        steps_ir=[],
        protocol_draft="",
        tool_logs=[],
        verification_logs=[],
    )
