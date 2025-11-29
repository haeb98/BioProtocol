import argparse
import json
import os

from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract protocol steps from sec_text using GPT.")
    parser.add_argument("--input_file", required=True, help="Path to gold_pairs_testset.jsonl")
    parser.add_argument("--output_file", required=True, help="Path to save results in JSONL")
    parser.add_argument("--model", default="gpt-4-1106-preview", help="OpenAI model to use")
    parser.add_argument("--resume", action="store_true", help="Resume if file already exists")
    return parser.parse_args()


def load_processed_ids(path):
    if not os.path.exists(path):
        return set()
    with open(path, "r", encoding="utf-8") as f:
        return {json.loads(line)["protocol_id"] for line in f}


def generate_prompt(sec_text):
    sys = (
        "You are a biomedical lab assistant. Your job is to read a biological protocol's "
        "methods section and generate a list of detailed experimental steps."
        "Do not hallucinate. List all steps sequentially."
    )
    user = (
        f"METHODS SECTION:\n{sec_text}\n\n"
        f"Extract a numbered list of experimental steps clearly, "
        f"in JSON format with key 'steps'.\n"
        f"Output only valid JSON."
    )

    return sys, user


def call_llm(client, system_msg, user_msg, model="gpt-4-1106-preview", max_retries=3):
    for i in range(max_retries):
        try:
            res = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg}
                ],
                response_format={"type": "json_object"},
                temperature=0.3,
            )
            return json.loads(res.choices[0].message.content)
        except Exception as e:
            print(f"⚠️ Retry {i + 1} failed: {e}")
    raise RuntimeError(f"Failed after {max_retries} retries.")


def main():
    args = parse_args()
    load_dotenv()

    client = OpenAI()

    processed_ids = load_processed_ids(args.output_file) if args.resume else set()
    mode = "a" if args.resume else "w"

    with open(args.input_file, "r", encoding="utf-8") as fin, \
            open(args.output_file, mode, encoding="utf-8") as fout:

        for line in tqdm(fin, desc="Processing protocols"):
            ex = json.loads(line.strip())
            protocol_id = ex["protocol_id"]
            sec_text = ex.get("sec_text", "")

            if not sec_text or protocol_id in processed_ids:
                continue

            try:
                sys_prompt, user_prompt = generate_prompt(sec_text)
                result = call_llm(client, sys_prompt, user_prompt, model=args.model)

                if isinstance(result, dict) and "steps" in result:
                    steps = result["steps"]
                elif isinstance(result, list):
                    steps = result
                else:
                    steps = result.get("steps", [])

                output = {"protocol_id": protocol_id, "steps": steps}
                fout.write(json.dumps(output, ensure_ascii=False) + "\n")
                fout.flush()
            except Exception as e:
                print(f"❌ Error on {protocol_id}: {e}")
                continue


if __name__ == "__main__":
    main()
