import argparse
import json

from tools.writer_agent import generate_protocol


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_prompt(input_data, mode: str) -> str:
    if mode == "sec_text":
        return input_data.get("sec_text", "")

    elif mode == "base":
        return ""

    elif mode == "tasks":
        tasks = [s["task_name"] for s in input_data.get("steps", [])]
        # sec_text = input_data.get("sec_text", "")
        return "\nTask List:\n" + "\n".join(f"- {t}" for t in sorted(set(tasks)))

    elif mode == "task_steps":
        task_map = {}
        for s in input_data.get("steps", []):
            task_map.setdefault(s["task_name"], []).append(s["step_text"])
        # sec_text = input_data.get("sec_text", "")
        return "\n".join(
            f"Task: {task}\n" + "\n".join(f"  - {step}" for step in steps)
            for task, steps in task_map.items()
        )

    elif mode == "task_step_ir":
        ir_path = "prototype/output/bio2096_verified_ir_60.json"
        with open(ir_path, "r", encoding="utf-8") as f:
            irs = json.load(f)
        task_map = {}
        for ir in irs:
            task = ir.get("task_name", "Unknown Task")
            text = f"[{ir.get('action', '')}] → materials: {ir.get('materials', [])}, params: {ir.get('parameters', [])}"
            task_map.setdefault(task, []).append(text)
        sec_text = input_data.get("sec_text", "")
        return sec_text + "\n".join(
            f"Task: {task}\n" + "\n".join(f"  - {line}" for line in lines)
            for task, lines in task_map.items()
        )

    else:
        raise ValueError(f"Unknown mode: {mode}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input JSON (e.g., prototype/output/bio2096_input.json)")
    parser.add_argument("--mode", required=True, choices=["sec_text", "tasks", "task_steps", "task_step_ir", "base"])
    parser.add_argument("--title", required=True, help="Protocol title to use")
    parser.add_argument("--output", required=True, help="Output .txt file path")
    args = parser.parse_args()

    input_data = load_json(args.input)
    prompt = extract_prompt(input_data, args.mode)
    result = generate_protocol(prompt, title=args.title)

    with open(args.output, "w", encoding="utf-8") as f:
        f.write(result)
    print(f"✅ Protocol written to {args.output}")
