#!/usr/bin/env python
import argparse
import json


def iter_concatenated_json_objects(path):
    """
    tasks_rag.jsonl처럼 각 JSON object가 여러 줄로 쓰여 있고
    사이에 콤마/배열 없이 그냥 이어져 있는 파일을
    brace count를 이용해서 하나씩 파싱하는 generator.
    """
    buf = ""
    brace = 0

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            brace += line.count("{") - line.count("}")
            buf += line
            if brace == 0 and buf.strip():
                # 하나의 JSON object 완성
                try:
                    obj = json.loads(buf)
                    yield obj
                except Exception as e:
                    print("❌ JSON parse error:", e)
                    print("---- offending chunk ----")
                    print(buf[:1000])
                    print("---- end ----")
                    raise
                buf = ""


def normalize_rag_to_flat(in_path, out_path):
    """
    RAG 결과 파일(in_path)을 baseline 형식(out_path)으로 변환:
    - input  : {protocol_id, tasks: [{task_id, title, description, ...}], llm_raw}
    - output : 한 줄에 한 개 태스크
              {"protocol_id": ..., "task_id": ..., "title": ..., "description": ...}
    """
    n_proto = 0
    n_tasks = 0

    with open(out_path, "w", encoding="utf-8") as out_f:
        for obj in iter_concatenated_json_objects(in_path):
            n_proto += 1
            protocol_id = obj.get("protocol_id")
            tasks = obj.get("tasks", [])

            for t in tasks:
                rec = {
                    "protocol_id": protocol_id,
                    "task_id": t.get("task_id"),
                    "title": t.get("title", ""),
                    "description": t.get("description", ""),
                }
                out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_tasks += 1

    print(f"✅ Converted {n_proto} protocols, {n_tasks} tasks → {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", required=True, help="RAG 결과 파일 (예: runs/tasks_rag.jsonl)")
    parser.add_argument("--output", "-o", required=True, help="변환된 flat JSONL 파일 경로")
    args = parser.parse_args()

    normalize_rag_to_flat(args.input, args.output)


if __name__ == "__main__":
    main()
