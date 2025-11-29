# prototype/run_verifier_bio2096.py

import argparse
import json

from openai import OpenAI

from tools.doc_search import doc_search
from tools.ir_parser import parse_step_ir


def load_input(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_output(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def call_llm(client, messages):
    response = client.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=messages,
        temperature=0.3
    )
    return response.choices[0].message.content


def run_verification(protocol):
    client = OpenAI()
    methods_text = protocol["sec_text"]
    steps = protocol["steps"]
    verified_steps = []

    for idx, step in enumerate(steps):
        step_text = step.get("step_text", "")
        history = []
        logs = []
        final_ir = None

        messages = [
            {"role": "system", "content": (
                "You are a verification agent for experimental protocols. "
                "You will analyze a step and verify its validity by optionally using external tools like doc_search or ir_parser. "
                "Use Thought -> Action -> Observation style and always end with Final Answer."
            )},
            {"role": "user", "content": f"Step: {step_text}"}
        ]

        for _ in range(5):  # maximum 5 LLM interactions
            response = call_llm(client, messages)
            logs.append(response)

            if "Final Answer:" in response:
                final_ir = response.split("Final Answer:")[-1].strip()
                break

            if "Action:" in response:
                action_line = next((l for l in response.split("\n") if l.startswith("Action:")), "")
                if "doc_search:" in action_line:
                    query = action_line.split("doc_search:")[1].strip().strip('"')
                    result = doc_search(query, methods_text)
                    messages.append({"role": "assistant", "content": response})
                    messages.append({"role": "user", "content": f"Observation: {result}"})
                elif "ir_parser:" in action_line:
                    result = parse_step_ir(step_text, methods_text)
                    messages.append({"role": "assistant", "content": response})
                    messages.append({"role": "user", "content": f"Observation: {result}"})
                else:
                    break
            else:
                break

        verified_steps.append({
            "original_step": step_text,
            "reasoning_log": logs,
            "final_ir": final_ir
        })
    return verified_steps


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        protocol = json.load(f)

    result = run_verification(protocol)
    save_output(result, args.output)
