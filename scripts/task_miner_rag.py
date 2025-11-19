#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
RAG-assisted task miner for Bio-protocol gold testset.

- 입력: gold_pairs_testset.jsonl (protocol_id, sec_text 포함)
- RAG 코퍼스: protocols_wo_test50.annot.jsonl (data/rag/corpus/ 아래)
- 출력: 각 protocol_id에 대해 high-level task 리스트 (tasks_rag.jsonl)

baseline(task_miner_baseline.py) 와 동일한 출력 포맷:
{
  "protocol_id": "...",
  "tasks": [
    {
      "task_id": "Bio-protocol-2096::T1",
      "title": "...",
      "description": "...",
      "key_materials": [...],
      "goal": "to ..."
    },
    ...
  ],
  "llm_raw": "<원본 LLM 응답(JSON 문자열 또는 에러 메시지)>"
}
"""

import argparse
import json
import os
import re
import time
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Tuple

from openai import OpenAI

client = OpenAI()


@dataclass
class Task:
    """Representation of a single high–level experimental task."""
    task_id: str
    title: str
    description: str
    key_materials: List[str]
    goal: str


def tokenize(text: str) -> List[str]:
    """Simple whitespace and punctuation tokeniser for similarity scoring."""
    return re.findall(r"\b\w+\b", text.lower())


def load_corpus(corpus_path: str) -> List[Dict[str, Any]]:
    """
    Load a corpus of protocol chunks from a JSONL file.

    protocols_wo_test50.annot.jsonl 의 각 라인은 대략 다음과 같은 형태라고 가정:
    {
      "id": "Bio-protocol-2096",
      "protocol": "<methods or full protocol text ...>",
      ... (other fields)
    }

    만약 실제 키 이름이 다르면 아래에서 get(...) 부분만 바꿔주면 됨.
    """
    corpus: List[Dict[str, Any]] = []
    with open(corpus_path, encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            text = (
                    obj.get("protocol")
                    or obj.get("protocol_text")
                    or obj.get("sec_text")
                    or obj.get("text")
                    or ""
            )
            corpus.append(
                {
                    "id": obj.get("id") or obj.get("protocol_id"),
                    "text": text,
                }
            )
    return corpus


def compute_similarity(query_tokens: List[str], doc_tokens: List[str]) -> float:
    """Compute a simple Jaccard similarity between two token lists."""
    set_q = set(query_tokens)
    set_d = set(doc_tokens)
    if not set_q or not set_d:
        return 0.0
    return len(set_q & set_d) / len(set_q | set_d)


def retrieve_top_k(methods_text: str, corpus: List[Dict[str, Any]], k: int) -> List[Dict[str, Any]]:
    """Retrieve the top-k similar entries from the corpus based on Jaccard similarity."""
    query_tokens = tokenize(methods_text)
    scored: List[Tuple[float, Dict[str, Any]]] = []
    for entry in corpus:
        doc_tokens = tokenize(entry["text"])
        sim = compute_similarity(query_tokens, doc_tokens)
        if sim > 0:
            scored.append((sim, entry))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [entry for _, entry in scored[:k]]


def build_messages(methods_text: str, retrieved_structures: List[str], max_tasks: int) -> List[Dict[str, str]]:
    """
    OpenAI chat messages 형식(system/user)을 구성.

    response_format={"type": "json_object"} 와 함께 사용되며
    모델은 반드시 JSON object 하나만 반환해야 한다.
    """
    examples = "\n\n".join(
        f"Example protocol structure:\n{ex}" for ex in retrieved_structures
    )

    system_msg = (
        "You are an expert experimental scientist. "
        "You will be given the Methods section of a biomedical research paper "
        "and optional example protocol structures from similar experiments. "
        "Your job is to extract a list of high-level experimental tasks "
        "that a person must perform in order to replicate the experiment.\n\n"
        "Use the example structures only as inspiration for how to break down the experiment, "
        "but DO NOT copy tasks that are not supported by the Methods text. "
        "Each task must represent a coherent phase of the experiment (e.g., cell seeding, drug treatment, imaging, data analysis), "
        "not a single micro-action.\n\n"
        f"Aim to produce roughly 4–{max_tasks} tasks. "
        "If the experiment logically has more phases than this, combine adjacent small steps into a single task.\n\n"
        "For EACH task, provide:\n"
        "- title: short title (3–12 words)\n"
        "- description: 1–2 sentence summary of what is done in that phase\n"
        "- key_materials: list of key materials/reagents/equipment used in that phase\n"
        "- goal: short phrase starting with 'to ...' describing the purpose of the phase\n\n"
        "You MUST respond with a SINGLE JSON object of the form:\n"
        "{ \"tasks\": [ {\"title\": ..., \"description\": ..., \"key_materials\": [...], \"goal\": ...}, ... ] }\n"
        "Do NOT include any explanation, comments, or markdown. Return only the JSON object."
    )

    user_msg = (
        f"Methods section of the target protocol:\n\n{methods_text}\n\n"
        "Optional example task structures from similar protocols:\n"
        f"{examples}\n\n"
        "Now extract the high-level experimental tasks as described. "
        "Remember to output ONLY the JSON object with key 'tasks'."
    )

    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]


def call_openai_chat(messages: List[Dict[str, str]], model: str, max_retries: int = 3) -> str:
    """
    Wrapper around OpenAI chat.completions API (openai>=1.0.0).

    - response_format={"type": "json_object"} 를 사용하여
      항상 JSON string 이 반환되도록 강제.
    """
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY environment variable is not set")

    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.0,
                max_tokens=1024,
                response_format={"type": "json_object"},
            )
            content = resp.choices[0].message.content
            if not content:
                raise ValueError("Empty response content from model")
            return content
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            # 지수적 backoff
            time.sleep(2 ** attempt)


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG-assisted task miner")
    parser.add_argument(
        "--pairs",
        required=True,
        help="Path to gold_pairs_testset.jsonl containing protocol_id and sec_text",
    )
    parser.add_argument(
        "--corpus",
        required=True,
        help="Path to annotated protocol corpus for retrieval (e.g., protocols_wo_test50.annot.jsonl)",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Output path for predicted tasks JSONL (e.g., runs/tasks_rag.jsonl)",
    )
    parser.add_argument(
        "--model",
        default="gpt-4o-mini",
        help="OpenAI chat model name (default: gpt-4o-mini)",
    )
    parser.add_argument(
        "--max-tasks",
        type=int,
        default=10,
        help="Suggested maximum number of tasks per protocol",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of similar protocols to retrieve for context",
    )
    args = parser.parse_args()

    corpus = load_corpus(args.corpus)

    with open(args.pairs, encoding="utf-8") as f_in, open(args.out, "w", encoding="utf-8") as f_out:
        for line in f_in:
            rec = json.loads(line)
            pid = rec.get("protocol_id")
            methods = rec.get("sec_text") or rec.get("text") or ""

            retrieved_entries = retrieve_top_k(methods, corpus, args.top_k)
            retrieved_structures: List[str] = []
            for entry in retrieved_entries:
                # retrieval된 프로토콜의 앞부분만 힌트로 사용
                snippet = entry["text"][:400].replace("\n", " ")
                retrieved_structures.append(snippet)

            messages = build_messages(methods, retrieved_structures, args.max_tasks)

            try:
                raw = call_openai_chat(messages, args.model)
                # response_format=json_object 덕분에 여기서 바로 파싱 가능
                tasks_data = json.loads(raw)

                tasks_list: List[Dict[str, Any]] = []
                for idx, task in enumerate(tasks_data.get("tasks", []), start=1):
                    task_obj = Task(
                        task_id=f"{pid}::T{idx}",
                        title=task.get("title", "").strip(),
                        description=task.get("description", "").strip(),
                        key_materials=[m.strip() for m in task.get("key_materials", [])],
                        goal=task.get("goal", "").strip(),
                    )
                    tasks_list.append(asdict(task_obj))

            except Exception as e:
                # 디버깅을 위해 에러 메시지 기록
                tasks_list = []
                raw = json.dumps({"error": str(e)}, ensure_ascii=False)

            output_rec = {
                "protocol_id": pid,
                "tasks": tasks_list,
                "llm_raw": raw,
            }
            f_out.write(json.dumps(output_rec, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
