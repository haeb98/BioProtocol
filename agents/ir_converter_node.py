# agents/ir_converter_node.py

import re

from agents.ir_state import IRState, IRNode, Param

UNIT_HINT = r"(μL|uL|mL|µL|mg/mL|μg/mL|ug/mL|mM|μg|µg|mg|g|hr|h|min|s|°C|rpm|cells|%)"


def extract_params(text: str, step_id: str, start_param_idx: int = 0):
    params = []
    # 숫자 + 단위 패턴
    pattern = r"(\d+[\d\.]*\s*)(" + UNIT_HINT + r")"
    matches = list(re.finditer(pattern, text))
    for i, match in enumerate(matches):
        value, unit = match.group(1).strip(), match.group(2)
        param_id = f"p{start_param_idx + i + 1}"
        name = {
            "°C": "temperature", "min": "time", "h": "time", "hr": "time",
            "μL": "volume", "µL": "volume", "uL": "volume", "mL": "volume",
            "mM": "concentration", "%": "percentage", "rpm": "speed",
            "μg": "mass", "µg": "mass", "mg": "mass", "cells": "cell_count",
        }.get(unit, "parameter")

        params.append(
            Param(
                param_id=param_id,
                step_id=step_id,
                name=name,
                value=value,
                unit=unit,
                span=match.group(0),
            )
        )
    return params


def run_ir_converter(state: IRState) -> IRState:
    ir_nodes, param_table, ir_edges = [], [], []
    steps_by_task = {}
    param_count = 0

    for step in state.steps_structured:
        # dict 로 들어온 step 기준
        step_id = step.get("step_id") or step.get("id")
        if not step_id:
            continue
        task_id = step.get("task_id", "T?")
        text = step.get("text") or step.get("description") or ""
        if not text.strip():
            continue

        action = text.strip().split()[0].lower()
        params = extract_params(text, step_id, param_count)
        param_count += len(params)

        ir_nodes.append(IRNode(id=step_id, action=action, params=params))
        param_table.extend(params)
        steps_by_task.setdefault(task_id, []).append(step_id)

    # task 내부 순서대로 edge 연결
    for step_ids in steps_by_task.values():
        for i in range(len(step_ids) - 1):
            ir_edges.append([step_ids[i], step_ids[i + 1]])

    state.ir_nodes = ir_nodes
    state.ir_edges = ir_edges
    state.param_table = param_table
    return state
