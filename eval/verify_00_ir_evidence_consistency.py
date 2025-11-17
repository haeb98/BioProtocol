# eval/verify_00_ir_evidence_consistency.py
import argparse
import json
import random

from openai import OpenAI

SYS_PROMPT = (
    "You are a strict scientific fact checker.\n"
    "Given a protocol Step (action + params) and:\n"
    "- the original METHODS text\n"
    "- optional external EVIDENCE snippets\n"
    "you must decide, for each parameter, whether it is DIRECTLY SUPPORTED "
    "by at least one of these texts.\n\n"
    "Rules:\n"
    "- 'Supported' only if the value (and unit) is explicitly stated or can be "
    "unambiguously inferred (e.g., 'six fields' -> 6).\n"
    "- If the text doesn't mention it, or it's ambiguous, mark 'Unsupported'.\n"
    "- External evidence from other protocols is valid ONLY if it clearly matches "
    "the same assay, organism, and context.\n"
    "- Do NOT guess reasonable values. Be conservative.\n"
    "Output strict JSON with keys: supported_params (list of names), "
    "unsupported_params (list of names), notes (string).\n"
)


def call_llm(client, model, methods_text, step, evidences):
    evidence_texts = []
    for ev in evidences[:3]:  # 상위 3개 snippet만 사용
        evidence_texts.append(f"- {ev.get('title', '')} :: {ev.get('snippet', '')}")
    user = {
        "methods_text": methods_text,
        "step": {
            "title": step.get("title"),
            "action": step.get("action"),
            "params": step.get("params") or [],
        },
        "evidence_snippets": evidence_texts,
    }
    resp = client.chat.completions.create(
        model=model,
        temperature=0.0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYS_PROMPT},
            {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
        ],
    )
    return json.loads(resp.choices[0].message.content)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ir", required=True, help="runs/s2b_grounded.evidence_only.ir.jsonl")
    ap.add_argument("--pairs", required=True, help="data/gold/gold_pairs_testset.jsonl")
    ap.add_argument("--out", required=True, help="eval/ir_evidence_consistency.jsonl")
    ap.add_argument("--model", default="gpt-4.1-mini")
    ap.add_argument("--max-steps", type=int, default=1000)
    args = ap.parse_args()

    client = OpenAI()

    # protocol_id -> methods text
    methods_map = {}
    with open(args.pairs, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            pid = r.get("protocol_id")
            if pid:
                methods_map[pid] = r.get("sec_text") or ""

    samples = []
    with open(args.ir, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            pid = r.get("protocol_id")
            methods = methods_map.get(pid, "")
            for n in r["nodes"]:
                if n.get("type") != "Step":
                    continue
                if not n.get("params"):
                    continue
                samples.append((pid, methods, n))

    random.shuffle(samples)
    samples = samples[:args.max_steps]

    out = open(args.out, "w", encoding="utf-8")
    for pid, methods, step in samples:
        evidences = step.get("evidence") or []
        res = call_llm(client, args.model, methods, step, evidences)
        out.write(json.dumps({
            "protocol_id": pid,
            "step_id": step.get("id"),
            "step_title": step.get("title"),
            "params": step.get("params"),
            "check": res,
        }, ensure_ascii=False) + "\n")
    out.close()
    print(f"[OK] wrote {len(samples)} checks to {args.out}")


if __name__ == "__main__":
    main()
