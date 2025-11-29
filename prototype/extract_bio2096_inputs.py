import json
import os
from pathlib import Path

# Set base directory relative to this script
BASE_DIR = Path(__file__).resolve().parent.parent
B2_PATH = BASE_DIR / "runs/b2_steps_new.jsonl"
ARTICLE_PATH = BASE_DIR / "data/gold/gold_pairs_testset.jsonl"
OUTPUT_PATH = BASE_DIR / "output/bio2096_input.json"


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    steps_data = load_jsonl(B2_PATH)
    articles = load_jsonl(ARTICLE_PATH)

    # Find matching entry
    steps = [s for s in steps_data if s["protocol_id"] == "Bio-protocol-2096"]
    article = next(a for a in articles if a["protocol_id"] == "Bio-protocol-2096")

    bundle = {
        "protocol_id": "Bio-protocol-2096",
        "sec_text": article["sec_text"],
        "steps": steps,
    }
    os.makedirs(OUTPUT_PATH.parent, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(bundle, f, indent=2, ensure_ascii=False)

    print(f"✅ Saved input for Bio-protocol-2096 to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
