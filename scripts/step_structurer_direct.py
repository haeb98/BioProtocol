#!/usr/bin/env python
"""
Direct Step Structurer (Methods-only)

- Input : gold_pairs_testset.jsonl (sec_text / text 에 methods 포함)
- Output: runs/steps_direct.jsonl

각 protocol 에 대해:
  - Methods 텍스트만 보고 상위 6~15개 정도의 실행 가능한 step 으로 쪼갠다.
  - Task 정보는 사용하지 않음 (Task Planner 없는 설정, Ablation A)
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict, Any

from openai import OpenAI


# ------- LLM 호출 헬퍼 -------

def call_llm_for_steps_direct(
        client: OpenAI,
        model: str,
        protocol_id: str,
        methods_text: str,
        max_steps: int = 15,
) -> Dict[str, Any]:
    """
    Methods 전체 텍스트만을 보고 executable steps 를 추출.

    반환 형태 (이상적):
    {
      "steps": [
        {
          "id": "S1",
          "title": "...",
          "instruction": "..."
        },
        ...
      ]
    }
    """

    # 컨텍스트 과부하 방지용 잘라내기 (너무 긴 Methods 보호)
    MAX_CHARS = 12000
    if len(methods_text) > MAX_CHARS:
        methods_text = methods_text[:MAX_CHARS]

    system_msg = (
        "You are an expert experimental biologist who writes clear, "
        "reproducible experimental protocols."
    )

    user_msg = f"""
You are given the MATERIALS & METHODS (or METHODS) section of a biology protocol.

Your goal is to extract a **linear sequence of executable steps** that a wet-lab scientist could follow.

Requirements:
- Focus on the **main experimental workflow**, not background or data interpretation.
- Each step must be:
  - Executable (e.g., 'Seed cells', 'Fix cells', 'Perform immunostaining').
  - Ordered logically in time.
- Write between 6 and {max_steps} steps per protocol.
- Do NOT include sub-sub-steps like 'repeat 3 times' as separate steps unless crucial.
- Avoid splitting steps too finely (no 'pick up pipette', 'open tube' etc.).

For each step, provide:
- id: S1, S2, S3, ...
- title: short phrase describing the step (e.g., 'Cell seeding', 'Immunostaining').
- instruction: 1-3 sentences that summarise what to do at this step.
  - You may paraphrase the text but keep the original meaning.
  - Do NOT fabricate parameters that are not in the text.

Return a JSON object with the following schema, and nothing else:
{{
  "steps": [
    {{
      "id": "S1",
      "title": "...",
      "instruction": "..."
    }},
    ...
  ]
}}

TEXT:
\"\"\"{methods_text}\"\"\"
"""

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.2,
        )
        content = resp.choices[0].message.content
        # content 가 JSON 문자열이라고 가정하고 파싱
        parsed = json.loads(content)
        return {"parsed": parsed, "raw": content, "error": None}
    except Exception as e:
        # 실패 시 raw 에 에러 메시지만 담기
        return {"parsed": None, "raw": str(e), "error": str(e)}


# ------- 입출력 유틸 -------

def extract_methods_text(pair_obj: Dict[str, Any]) -> str:
    """
    gold_pairs_testset.jsonl 한 줄에서 methods 텍스트를 가져오는 유틸.
    필드 이름이 환경마다 조금 다를 수 있어 안전하게 처리.
    """
    # 우선 sec_text 를 최우선으로 사용
    if "sec_text" in pair_obj and pair_obj["sec_text"]:
        return pair_obj["sec_text"]

    # 다음으로 text
    if "text" in pair_obj and pair_obj["text"]:
        return pair_obj["text"]

    # chars 는 너무 긴 raw 일 수 있지만 최후의 수단
    if "chars" in pair_obj and pair_obj["chars"]:
        return pair_obj["chars"]

    return ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pairs",
        type=str,
        required=True,
        help="Path to gold_pairs_testset.jsonl (or similar).",
    )
    parser.add_argument(
        "--out",
        type=str,
        required=True,
        help="Output JSONL path for direct steps (e.g., runs/steps_direct.jsonl).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="OpenAI chat model name (default: gpt-4o-mini).",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=15,
        help="Maximum number of steps to ask the model for (default: 15).",
    )
    args = parser.parse_args()

    client = OpenAI()  # OPENAI_API_KEY 환경변수 필요

    in_path = Path(args.pairs)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    with in_path.open("r", encoding="utf-8") as fin, out_path.open(
            "w", encoding="utf-8"
    ) as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            total += 1
            obj = json.loads(line)
            protocol_id = obj.get("protocol_id") or obj.get("id") or f"protocol_{total}"

            methods_text = extract_methods_text(obj)
            if not methods_text:
                record = {
                    "protocol_id": protocol_id,
                    "steps": [],
                    "llm_raw": json.dumps(
                        {"error": "empty methods text"},
                        ensure_ascii=False,
                    ),
                    "source": "direct_methods_only",
                }
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                continue

            llm_result = call_llm_for_steps_direct(
                client=client,
                model=args.model,
                protocol_id=protocol_id,
                methods_text=methods_text,
                max_steps=args.max_steps,
            )

            steps_out: List[Dict[str, Any]] = []
            if llm_result["parsed"] and isinstance(
                    llm_result["parsed"].get("steps", []), list
            ):
                for s in llm_result["parsed"]["steps"]:
                    sid = s.get("id") or f"S{len(steps_out) + 1}"
                    title = s.get("title") or ""
                    instr = s.get("instruction") or ""
                    steps_out.append(
                        {
                            "id": sid,
                            "title": title,
                            "instruction": instr,
                            # task_id 는 direct 에서는 없음
                            "task_id": None,
                        }
                    )

            record = {
                "protocol_id": protocol_id,
                "steps": steps_out,
                "llm_raw": llm_result["raw"],
                "source": "direct_methods_only",
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Done. Processed {total} protocols.")
    print(f"Saved to: {out_path}")


if __name__ == "__main__":
    main()
