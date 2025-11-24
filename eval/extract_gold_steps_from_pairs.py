# eval/extract_gold_steps_from_pairs.py
"""
gold_pairs_testset.jsonl 의 hierarchical_protocol 에서
하위 1.1, 1.2, 2.1 ... 등의 step 텍스트를 뽑아
gold_steps_testset.jsonl 로 저장하는 스크립트.

- 최상위 노드(1,2,3,...)는 task 로 간주하여 T1, T2, ... 로 매핑
- 하위 노드 중 value 가 문자열인 것들을 step 으로 사용
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Any


def numeric_key_sort(k: str):
    """
    '1', '2', '3.1', '10.2.3' 같은 hierarchical key 를
    숫자 리스트로 변환해서 정렬용 key 로 사용.
    """
    parts = k.split(".")
    nums = []
    for p in parts:
        try:
            nums.append(int(p))
        except ValueError:
            nums.append(0)
    return nums


def extract_steps_from_hierarchical_protocol(hp: Dict[str, Any]):
    """
    hierarchical_protocol dict 로부터
    - top-level key ('1','2',...) → T1,T2,... 매핑
    - 문자열 value 를 가진 하위 노드들을 step 으로 추출
    """
    # 1) top-level task key 찾기 (예: '1', '2', '3')
    top_keys = [
        k
        for k, v in hp.items()
        if "." not in k and isinstance(v, dict) and "title" in v
    ]
    top_keys_sorted = sorted(top_keys, key=numeric_key_sort)

    # '1' -> 'T1', '2' -> 'T2' ...
    top_to_task_id = {
        k: f"T{i + 1}" for i, k in enumerate(top_keys_sorted)
    }

    steps = []

    # 2) 전체 key 를 순회하면서 문자열 value 만 step 으로 사용
    for key in sorted(hp.keys(), key=numeric_key_sort):
        value = hp[key]

        # 문자열이 아닌 경우 (dict 등) 은 title 이거나 group 이라 보고 스킵
        if not isinstance(value, str):
            continue

        # 최상위 key 가 아닌 경우만 step 으로 본다 ( '1.1', '2.3', '3.1.2' 등 )
        if "." not in key:
            # 예외적으로 텍스트가 바로 오는 경우가 있을 수 있지만
            # 대부분 top-level 은 dict 이므로 여기선 스킵
            continue

        # top-level 번호 추출 ( '3.1.2' -> '3' )
        top = key.split(".")[0]
        task_id = top_to_task_id.get(top, None)

        step_text = value.strip()
        if not step_text:
            continue

        steps.append(
            {
                "step_id": key,  # hierarchical key 그대로
                "task_id": task_id,  # 매칭된 T1/T2... (없으면 None)
                "text": step_text,
            }
        )

    return steps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pairs",
        type=str,
        required=True,
        help="gold_pairs_testset.jsonl 경로",
    )
    parser.add_argument(
        "--out",
        type=str,
        required=True,
        help="gold_steps_testset.jsonl 출력 경로",
    )
    args = parser.parse_args()

    in_path = Path(args.pairs)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_protocols = 0
    n_total_steps = 0

    with in_path.open() as fin, out_path.open("w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)

            protocol_id = obj.get("protocol_id")
            title = obj.get("details", {}).get("bio_title") or obj.get(
                "details", {}
            ).get("title", "")

            hp = obj.get("bio", {}).get("hierarchical_protocol")
            if not hp:
                continue

            steps = extract_steps_from_hierarchical_protocol(hp)
            if not steps:
                continue

            out_obj = {
                "protocol_id": protocol_id,
                "title": title,
                "steps": steps,
            }
            fout.write(json.dumps(out_obj, ensure_ascii=False) + "\n")

            n_protocols += 1
            n_total_steps += len(steps)

    print(f"Saved {n_protocols} protocols, total {n_total_steps} steps to {out_path}")


if __name__ == "__main__":
    main()
