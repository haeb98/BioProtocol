# prototype/tools/ir_parser.py

import json

from openai import OpenAI


def parse_step_ir(step_text, methods_text, step_id=None, metadata=None):
    """
    Parse a step into an IR node dict with:
    id, action, parameters, materials, produces + metadata
    """
    client = OpenAI()
    messages = [
        {"role": "system", "content": (
            "You are an AI assistant that extracts structured information from experimental procedures.\n"
            "Given a step instruction, return a JSON with keys:\n"
            "- 'id': the step ID\n"
            "- 'action': the core verb phrase\n"
            "- 'parameters': list of {name, value, unit, raw} dictionaries if possible\n"
            "- 'materials': list of relevant substances (e.g. media, supplements)\n"
            "- 'produces': what this step yields\n"
            "Ensure valid JSON output."
        )},
        {"role": "user", "content": f"Step: {step_text}"}
    ]

    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            temperature=0.2
        )
        content = response.choices[0].message.content.strip()
    except Exception:
        return _fallback_ir(step_text, step_id, metadata)

    try:
        json_str = content[content.find("{"): content.rfind("}") + 1]
        ir_core = json.loads(json_str)
    except Exception:
        return _fallback_ir(step_text, step_id, metadata)

    # 통합 IR 구성
    ir_node = {
        "protocol_id": metadata.get("protocol_id"),
        "task_id": metadata.get("task_id"),
        "task_name": metadata.get("task_name"),
        "step_id": step_id,
        "step_text": step_text,
        "span_chunk": metadata.get("span_chunk", ""),
        "id": ir_core.get("id", step_id),
        "action": ir_core.get("action", step_text),
        "parameters": ir_core.get("parameters", []),
        "materials": ir_core.get("materials", []),
        "produces": ir_core.get("produces", None)
    }
    return ir_node


def _fallback_ir(step_text, step_id, metadata):
    return {
        "protocol_id": metadata.get("protocol_id"),
        "task_id": metadata.get("task_id"),
        "task_name": metadata.get("task_name"),
        "step_id": step_id,
        "step_text": step_text,
        "span_chunk": metadata.get("span_chunk", ""),
        "id": step_id,
        "action": step_text,
        "parameters": [],
        "materials": [],
        "produces": None
    }
