# agents/verifier_node.py

import json

from openai import OpenAI

from agents.verifier_state import VerifierState

client = OpenAI()


def call_llm(prompt: str, model: str = "gpt-3.5-turbo"):
    try:
        resp = client.chat.completions.create(
            model=model,
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
            response_format="json",
        )
        return json.loads(resp.choices[0].message.content)
    except Exception as e:
        print(f"Error in verifier LLM call: {e}")
        return {"verdict": "ambiguous", "evidence_span": ""}


def build_prompt(methods_text: str, param: dict) -> str:
    return f"""
You are a scientific verifier.

METHODS:
-----
{methods_text}
-----

PARAMETER TO VERIFY:
- name: {param['name']}
- value: {param['value']}
- unit: {param['unit']}

TASK:
1. If this value (with this unit) is clearly mentioned → "supported".
2. If related info exists but value/unit is unclear or slightly different → "ambiguous".
3. If you cannot find it → "unsupported".

OUTPUT: JSON with fields:
- verdict: one of ["supported", "ambiguous", "unsupported"]
- evidence_span: short quote from METHODS or "" if none
"""


def run_verifier(state: VerifierState) -> VerifierState:
    results = []
    for param in state.param_table:
        prompt = build_prompt(state.methods_text, param)
        res = call_llm(prompt)
        results.append(
            {
                "protocol_id": state.protocol_id,
                "step_id": param["step_id"],
                "param_id": param["param_id"],
                "name": param["name"],
                "value": param["value"],
                "unit": param["unit"],
                "verdict": res.get("verdict", "ambiguous"),
                "evidence_span": res.get("evidence_span", ""),
            }
        )

    state.param_verdicts = results
    return state
