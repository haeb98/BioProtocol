import argparse
import json

from openai import OpenAI


def load_input(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_txt(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def call_llm(messages):
    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=messages,
        temperature=0.3
    )
    return response.choices[0].message.content.strip()


def build_prompt(input_type, data, title):
    if input_type == "sec_text":
        content = data
        description = "This is the Methods section text from a biological paper."
    elif input_type == "tasks":
        content = "\n".join([f"- {task}" for task in data])
        description = "This is a list of experimental tasks."
    elif input_type == "task_steps":
        content = "\n".join([f"## {t['task_name']}\n" + "\n".join([f"- {s}" for s in t["steps"]]) for t in data])
        description = "This is a list of tasks and their step descriptions."
    elif input_type == "ir":
        content = json.dumps(data, indent=2, ensure_ascii=False)
        description = "This is a structured IR (Information Representation) from a protocol, including actions, parameters, materials, and outcomes."
    else:
        raise ValueError("Invalid input_type")

    return [
        {"role": "system", "content": (
            "You are a lab assistant that writes experimental protocols in natural language based on given scientific content."
        )},
        {"role": "user", "content": (
            f"""Protocol title: {title}

{description}
-------------------------
{content}

Please write the full experimental protocol in numbered list format (1., 2., 3., ...)."""
        )}
    ]


def main(args):
    title = args.title
    if args.input_type == "sec_text":
        input_data = load_txt(args.input)
    else:
        input_data = load_input(args.input)

    messages = build_prompt(args.input_type, input_data, title)
    output = call_llm(messages)

    with open(args.output, "w", encoding="utf-8") as f:
        f.write(output)
    print(f"✅ Protocol saved to {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_type", choices=["sec_text", "tasks", "task_steps", "ir"], required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--title", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    main(args)
