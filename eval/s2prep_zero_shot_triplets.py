# eval/s2prep_zero_shot_triplets.py
import json
import orjson
import os
import pathlib

from tqdm import tqdm

ROOT = pathlib.Path(".")
GOLD = ROOT / "data/gold/gold_pairs_pmc_bioprotocol_balanced.jsonl"
OUT = ROOT / "eval/s2prep_triplets.jsonl"
OUT.parent.mkdir(parents=True, exist_ok=True)

try:
    from openai import OpenAI

    client = OpenAI()
except Exception:
    import openai

    client = openai.OpenAI()

PROMPT = (
    "You are an information extraction model. From the METHODS text, extract a list of triples capturing materials with numeric values and units.\n"
    "Return strictly as JSON: {\"materials_zu\": [{\"material\":\"\",\"value\":0.0,\"unit\":\"\"}, ...]}\n"
    "Rules:\n- Parse only when a numeric value and a unit co-occur near a material name.\n- Use SI-like units when possible (µL, mL, °C, min, rpm, g, mM, µM, %)."
)


def call_llm(txt):
    r = client.chat.completions.create(
        model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
        messages=[{"role": "system", "content": PROMPT}, {"role": "user", "content": "METHODS:\n" + txt[:8000]}],
        temperature=0.0, max_tokens=1200
    )
    return r.choices[0].message.content


def parse_json(s):
    try:
        return json.loads(s)
    except Exception:
        import re
        m = re.search(r"\{[\s\S]*\}", s or "")
        if m:
            try:
                return json.loads(m.group(0))
            except:
                return {}
        return {}


def main():
    with open(GOLD, "r", encoding="utf-8") as fin, open(OUT, "w", encoding="utf-8") as fou:
        for line in tqdm(fin, desc="s2prep-extract"):
            pair = orjson.loads(line)
            pid = pair["protocol_id"]
            methods = pair["article"].get("methods_text", "")
            if not methods: continue
            raw = call_llm(methods)
            obj = parse_json(raw)
            rec = {"protocol_id": pid, "triplets": obj.get("materials_zu", [])}
            fou.write(orjson.dumps(rec).decode() + "\n")
    print(f"[OK] saved {OUT}")


if __name__ == "__main__":
    main()
