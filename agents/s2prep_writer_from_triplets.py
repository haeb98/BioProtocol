# agents/s2prep_writer_from_triplets.py
import orjson
import pathlib

ROOT = pathlib.Path(".")
TRI = ROOT / "eval/s2prep_triplets.jsonl"
GOLD = ROOT / "data/gold/gold_pairs_pmc_bioprotocol_balanced.jsonl"
OUT = ROOT / "eval/s2prep_generations.jsonl"
OUT.parent.mkdir(parents=True, exist_ok=True)


def load_gold_index(path):
    idx = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = orjson.loads(line)
            pid = str(obj["protocol_id"])
            prot = obj.get("protocol") or {}
            idx[pid] = {
                "domain": obj.get("domain", "Unknown"),
                "gold_hier": prot.get("hierarchical_protocol"),
                "gold_text": prot.get("text", "")
            }
    return idx


def make_hier(triplets):
    hp = {};
    sec = 1
    hp[str(sec)] = {"title": "Preparation & Materials"}
    i = 1
    for t in triplets or []:
        mat = (t.get("material") or "").strip()
        val = t.get("value")
        uni = (t.get("unit") or "").strip()
        line = f"Prepare {mat}" if mat else "Prepare"
        if val is not None or uni:
            line += f": {val if val is not None else ''}{uni}"
        hp[f"{sec}.{i}"] = line.strip()
        i += 1
    sec += 1
    hp[str(sec)] = {"title": "Procedure (triplets-only; actions unavailable)"}
    return {"hierarchical_protocol": hp}


def main():
    gold = load_gold_index(GOLD)
    with open(TRI, "r", encoding="utf-8") as fin, open(OUT, "w", encoding="utf-8") as fou:
        for line in fin:
            obj = orjson.loads(line)
            pid = str(obj["protocol_id"])
            hp = make_hier(obj.get("triplets", []))
            gi = gold.get(pid, {})
            rec = {
                "protocol_id": pid,
                "domain": gi.get("domain", "Unknown"),
                "gen_hier": hp["hierarchical_protocol"],
                "gold_hier": gi.get("gold_hier"),
                "gold_text": gi.get("gold_text")
            }
            fou.write(orjson.dumps(rec).decode() + "\n")
    print(f"[OK] wrote {OUT}")


if __name__ == "__main__":
    main()
