# agents/s2_writer_from_ir.py
import pathlib

import orjson

ROOT = pathlib.Path(".")
IN_IR = ROOT / "eval/s2_ir.jsonl"
GOLD = ROOT / "data/gold/gold_pairs_pmc_bioprotocol_balanced.jsonl"
OUT = ROOT / "eval/s2_generations.jsonl"
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
                "gold_text": prot.get("text", ""),
                "title": prot.get("title", "")
            }
    return idx


def hier_from_ir(ir):
    hp = {}
    sec = 1;
    hp[str(sec)] = {"title": "Preparation & Materials"};
    i = 1
    for m in (ir.get("materials") or []):
        line = f"Prepare {m.get('name', '')}".strip()
        conc = (m.get("concentration") or {})
        amt = (m.get("amount") or {})
        if conc.get("value") is not None:
            line += f" ({conc.get('value')}{conc.get('unit', '')})"
        if amt.get("value") is not None:
            line += f", amount {amt.get('value')}{amt.get('unit', '')}"
        if m.get("catalog"):  line += f", Catalog: {m['catalog']}"
        if m.get("supplier"): line += f", Supplier: {m['supplier']}"
        hp[f"{sec}.{i}"] = line;
        i += 1

    for s in (ir.get("solutions") or []):
        line = f"Prepare solution: {s.get('name', '')}"
        hp[f"{sec}.{i}"] = line;
        i += 1
        for comp in (s.get("recipe") or []):
            a = comp.get("amount") or {}
            hp[f"{sec}.{i}"] = f" - Add {comp.get('name', '')}: {a.get('value')}{a.get('unit', '')}"
            i += 1

    sec += 1;
    hp[str(sec)] = {"title": "Procedure"};
    i = 1
    for st in (ir.get("steps") or []):
        params = st.get("params") or {}
        bits = []
        for k in ["time", "temperature", "volume", "speed", "g", "concentration"]:
            if k in params and isinstance(params[k], dict):
                v = params[k].get("value");
                u = params[k].get("unit", "")
                if v is not None:
                    bits.append(f"{k}={v}{u}")
        base = st.get("action", "Do")
        if st.get("reagent"): base += f" {st['reagent']}"
        hp[f"{sec}.{i}"] = (base + (f" ({', '.join(bits)})" if bits else "")).strip()
        i += 1

    return {"hierarchical_protocol": hp}


def main():
    gold = load_gold_index(GOLD)
    with open(IN_IR, "r", encoding="utf-8") as fin, open(OUT, "w", encoding="utf-8") as fou:
        for line in fin:
            obj = orjson.loads(line)
            pid = str(obj["protocol_id"])
            ir = obj.get("ir") or {}
            gen = hier_from_ir(ir) if ir else {
                "hierarchical_protocol": {"1": {"title": "Preparation & Materials"}, "2": {"title": "Procedure"}}}
            gi = gold.get(pid, {})
            rec = {
                "protocol_id": pid,
                "domain": gi.get("domain", "Unknown"),
                "gen_hier": gen.get("hierarchical_protocol"),
                "gold_hier": gi.get("gold_hier"),
                "gold_text": gi.get("gold_text")
            }
            fou.write(orjson.dumps(rec).decode() + "\n")
    print(f"[OK] wrote {OUT}")


if __name__ == "__main__":
    main()
