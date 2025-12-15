# prototype/run_verifier_bio2096.py

import argparse
import json

from openai import OpenAI

from tools.calculator import calculate
from tools.doc_search_sematic import doc_search
from tools.ir_parser import parse_step_ir
from tools.rag_tool import rag_tool


def load_input(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_output(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


import re


def parse_tool_action(action_line):
    match = re.match(r'\[?(\w+):\s*["“](.*?)["”]\]?', action_line.strip())
    if match:
        tool = match.group(1)
        arg = match.group(2)
        return tool, arg
    return None, None


def call_llm(client, messages):
    response = client.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=messages,
        temperature=0.3
    )
    return response.choices[0].message.content


def run_verification(protocol):
    client = OpenAI()
    methods_text = protocol.get("sec_text", "")
    steps = protocol.get("steps", [])
    # steps = steps[:5]  # Limit to first 10 steps for efficiency
    ir_nodes = []
    trace_list = []

    for i, step in enumerate(steps):
        print(f"\n🧪 Verifying Step {i + 1}/{len(steps)} - {step.get('step_id')}")
        step_id = step.get("step_id")
        step_text = step.get("step_text", "")
        span_chunk = step.get("span_chunks", "")
        step_trace = []
        final_ir = None
        messages = [
            {"role": "system", "content": (
                """
                You are a verification agent for experimental protocols. Your goal is to produce a fully grounded IR (Information Representation) node for the given step using tools such as:

                - doc_search[query]: Search within the Methods section of the protocol to find evidence.
                - rag_tool[query]: Retrieve similar steps from a database of known protocols.
                - ir_parser[text]: Extract structured action, materials, parameters, and products from the step.
                - calculator[expression]: Compute quantities, concentrations, or convert units.
                
                Use a ReAct style format:
                Thought: ...
                Action: [tool_name: "your query"]
                Observation: ...
                (Repeat until you are confident you can produce a full IR)
                
                Only when you are confident that your IR includes all necessary fields (action, parameters, materials, produces) and is either fully supported or its origin is explained, you must write:
                Final Answer: {...json...}
                
                Do NOT output Final Answer if you haven't used any tools or your answer lacks grounding.
                """

            )},
            {"role": "user", "content": f"""Verify this step and produce a structured IR:
                    Step: {step_text}, Span chuncks from Methods: {span_chunk}
                    Use tools as needed. DO NOT return Final Answer until you have enough evidence."""}
        ]

        for j in range(5):  # maximum 5 interactions
            print(f"  ↪️ Iteration {j + 1}: Asking LLM...")
            response = call_llm(client, messages)
            print(f"    📥 LLM response:\n{response}\n")

            # Parse Thought and Action from the LLM's response
            thought, action = "", ""
            for line in response.splitlines():
                if line.startswith("Thought:"):
                    thought = line[len("Thought:"):].strip()
                elif line.startswith("Action:"):
                    action = line[len("Action:"):].strip()

            step_entry = {"thought": thought, "action": action}

            # Check if LLM provided the final answer
            if "Final Answer:" in response:
                try:
                    ir_text = response.split("Final Answer:")[1].strip()
                    ir_data = json.loads(ir_text)
                    final_ir = ir_data
                    step_entry["final_answer"] = ir_data
                except Exception:
                    step_entry["final_answer"] = ir_text
                    final_ir = None
                step_trace.append(step_entry)
                print("✅ Final Answer received.\n")
                break  # stop loop

            # If an action was provided, use the corresponding tool
            result = None
            tool, arg = parse_tool_action(action)
            if tool == "doc_search":
                result = doc_search(arg, methods_text)
            elif tool == "rag_tool":
                result = rag_tool(arg)
            elif tool == "calculator":
                result = calculate(arg)
            elif tool == "ir_parser":
                result = parse_step_ir(
                    step_text,
                    methods_text,
                    step_id=step["step_id"],
                    metadata={
                        "protocol_id": step["protocol_id"],
                        "task_id": step["task_id"],
                        "task_name": step["task_name"],
                        "span_chunk": step["span_chunk"]
                    }
                )

            # Update LLM context and trace
            step_entry["observation"] = result
            step_trace.append(step_entry)
            messages.append({"role": "assistant", "content": response})
            messages.append(
                {"role": "user", "content": f"Observation: {json.dumps(result, ensure_ascii=False)}"})

            print(f"↪️ No Final Answer yet. Continuing...\n")

        # If the LLM did not return an IR, use the parser fallback
        if not final_ir:
            print("⚠️ Final Answer not returned. Using ir_parser fallback.")
            final_ir = parse_step_ir(
                step_text,
                methods_text,
                step_id=step["step_id"],
                metadata={
                    "protocol_id": step["protocol_id"],
                    "task_id": step["task_id"],
                    "task_name": step["task_name"],
                    "span_chunk": step["span_chunk"]
                }
            )

        # **confidence/halucination 플래그 추가**
        used_tool = any("action" in entry for entry in step_trace)
        if isinstance(final_ir, dict):
            # 도구 사용 시 높은 확신, 사용 없으면 낮은 확신 (예시 값)
            final_ir["confidence"] = 0.9 if used_tool else 0.2
            final_ir["hallucination"] = False if used_tool else True
        else:
            # parse_step_ir이 문자열 등을 반환했을 경우 대비
            final_ir = {
                "action": step_text,
                "parameters": [],
                "materials": [],
                "produces": [],
                "confidence": 0.0,
                "hallucination": True
            }
        # Ensure full metadata is included in final_ir
        if isinstance(final_ir, dict):
            final_ir.update({
                "protocol_id": step.get("protocol_id"),
                "task_id": step.get("task_id"),
                "task_name": step.get("task_name"),
                "step_id": step.get("step_id"),
                "step_text": step.get("step_text"),
                "span_chunk": step.get("span_chunk", "")
            })

        ir_nodes.append(final_ir)
        trace_list.append({
            "step_id": step_id,
            "step_text": step_text,
            "trace": step_trace
        })

    print(f"\n✅ Completed {len(steps)} steps. Saving outputs...")

    return ir_nodes, trace_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input JSON file for Bio-protocol-2096")
    parser.add_argument("--output_ir", required=True, help="Output file for verifier IR JSON")
    parser.add_argument("--output_trace", required=True, help="Output file for verifier trace JSON")
    args = parser.parse_args()

    protocol = load_input(args.input)
    ir_nodes, trace = run_verification(protocol)
    save_output(ir_nodes, args.output_ir)
    save_output(trace, args.output_trace)
    print(f"✅ IR output saved to {args.output_ir}")
    print(f"✅ Trace output saved to {args.output_trace}")
