# writer_agent.py

import json
import os

import openai


def load_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data


def get_task_map(tasks):
    return {t["task_id"]: t for t in tasks}


def get_task_children(tasks):
    children = {}
    for t in tasks:
        parent = t.get("parent_id")
        if parent:
            children.setdefault(parent, []).append(t["task_id"])
        else:
            children.setdefault(None, []).append(t["task_id"])
    return children


def get_steps_by_task(steps):
    step_map = {}
    for s in steps:
        task_id = s.get("task_id") or s.get("task_ref")
        if task_id:
            step_map.setdefault(task_id, []).append(s["step_id"])
    return step_map


def load_ir_steps(ir_json):
    return {s["step_id"]: s for s in ir_json.get("steps", [])}


def format_instruction(ir_step):
    """Generate instruction text via OpenAI"""
    openai.api_key = os.getenv("OPENAI_API_KEY")
    step_data = {
        "action": ir_step.get("action", ""),
        "materials": ir_step.get("materials", []),
        "parameters": ir_step.get("parameters", [])
    }
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system",
                 "content": "Convert the action, materials, and parameters into a clear experimental step."},
                {"role": "user", "content": json.dumps(step_data)}
            ],
            temperature=0.2
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"⚠️ Failed to generate step: {e}")
        return ir_step.get("action", "Do the experimental step.")


def build_hierarchical_protocol(
        task_id, prefix, task_map, task_children, steps_by_task, ir_steps
):
    out = {}
    children = task_children.get(task_id, [])

    for idx, t_id in enumerate(children, start=1):
        number = f"{prefix}{idx}" if prefix else str(idx)
        task = task_map.get(t_id, {})
        title = task.get("title") or task.get("task_name") or f"Task {t_id}"
        node = {"title": title}

        # If it has sub-tasks, recurse
        if t_id in task_children:
            subsections = build_hierarchical_protocol(
                t_id, number + ".", task_map, task_children, steps_by_task, ir_steps
            )
            node.update(subsections)
        else:
            # Otherwise, populate step instructions
            step_ids = steps_by_task.get(t_id, [])
            for i, s_id in enumerate(step_ids, start=1):
                step_num = f"{number}.{i}"
                step_obj = ir_steps.get(s_id)
                if step_obj:
                    step_txt = format_instruction(step_obj)
                    out[step_num] = step_txt
            # Also insert the section header
            out[number] = {"title": title}
            continue

        out[number] = node

    return out


def main():
    # File paths (relative to project root)
    ir_path = "prototype/output/bio2096_verifier.json"
    tasks_path = "runs/b1_tasks_new.jsonl"
    steps_path = "runs/b2_steps_new.jsonl"
    output_path = "prototype/output/bio2096_writer.json"

    # Load input data
    with open(ir_path, "r", encoding="utf-8") as f:
        ir_data = json.load(f)

    tasks = load_jsonl(tasks_path)
    steps = load_jsonl(steps_path)

    # Prepare maps
    task_map = get_task_map(tasks)
    task_children = get_task_children(tasks)
    steps_by_task = get_steps_by_task(steps)
    ir_steps = load_ir_steps(ir_data)

    # Build protocol hierarchy
    hierarchical_protocol = build_hierarchical_protocol(
        None, "", task_map, task_children, steps_by_task, ir_steps
    )

    final_output = {
        "protocol_id": "Bio-protocol-2096",
        "hierarchical_protocol": hierarchical_protocol
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final_output, f, ensure_ascii=False, indent=2)

    print(f"✅ Output saved to {output_path}")


if __name__ == "__main__":
    main()
