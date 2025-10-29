import json
import orjson
import pathlib
import re

from tqdm import tqdm

ROOT = pathlib.Path(".")
IR = ROOT / "eval/s2_ir.jsonl"
GOLD = ROOT / "data/gold/gold_pairs_pmc_bioprotocol_balanced.jsonl"
OUTJ = ROOT / "eval/s2_ir_report.json"

TOK = re.compile(r"[a-z0-9\+\-\./%°µ]+", re.I)
NUM = re.compile(r"\d+(?:\.\d+)?")
UNIT = re.compile(r"(°c|degc|°f|min|h|hr|s|sec|ms|µl|ul|ml|l|mg|g|kg|ng|pg|mm|cm|µm|um|nm|mM|µM|uM|nM|%|rpm|g/l|mg/ml)",
                  re.I)


def to_tokens(s): return set(TOK.findall((s or "").lower()))


def parse_params(s):
    out = []
    for m in re.finditer(rf"{NUM.pattern}\s*{UNIT.pattern}", s):
        out.append(m.group(0).lower())
    return set(out)


def flatten_hier(h):
    if not isinstance(h, dict): return ""

    def keyer(k):
        return [int(p) if p.isdigit() else p for p in str(k).split(".")]

    lines = []
    for k in sorted(h.keys(), key=keyer):
        v = h[k]
        if isinstance(v, dict) and "title" in v:
            lines.append(v["title"])
        elif isinstance(v, str):
            lines.append(v)
    return "\n".join(lines)


def main():
    gold_idx = {}
    with open(GOLD, "r", encoding="utf-8") as f:
        for line in f:
            obj = orjson.loads(line)
            pid = str(obj["protocol_id"])
            gold_idx[pid] = flatten_hier(obj["protocol"].get("hierarchical_protocol") or {}) or obj["protocol"].get(
                "text", "")

    n = 0;
    mats_p = r_p = f1_p = 0.0;
    par_p = r_r = p_f1 = 0.0
    with open(IR, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="eval-s2-ir"):
            r = orjson.loads(line)
            pid = str(r["protocol_id"])
            gold_txt = gold_idx.get(pid, "")
            if not gold_txt: continue

            gold_mat = to_tokens(gold_txt)
            gold_par = parse_params(gold_txt)

            ir = r.get("ir") or {}
            pred_mats = set()
            for m in (ir.get("materials") or []):
                if m.get("name"): pred_mats |= to_tokens(m["name"])
            pred_pars = set()
            for st in (ir.get("steps") or []):
                txt = json.dumps(st.get("params") or {})
                pred_pars |= parse_params(txt)

            # materials F1
            tp = len(pred_mats & gold_mat);
            fp = len(pred_mats - gold_mat);
            fn = len(gold_mat - pred_mats)
            P = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            R = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            F = (2 * P * R) / (P + R) if (P + R) > 0 else 0.0
            mats_p += P;
            r_p += R;
            f1_p += F

            # params F1
            tp = len(pred_pars & gold_par);
            fp = len(pred_pars - gold_par);
            fn = len(gold_par - pred_pars)
            P = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            R = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            F = (2 * P * R) / (P + R) if (P + R) > 0 else 0.0
            par_p += P;
            r_r += R;
            p_f1 += F

            n += 1

    rep = {
        "n": n,
        "materials_avg": {"P": round(mats_p / n, 4), "R": round(r_p / n, 4), "F1": round(f1_p / n, 4)} if n else {},
        "params_avg": {"P": round(par_p / n, 4), "R": round(r_r / n, 4), "F1": round(p_f1 / n, 4)} if n else {}
    }
    OUTJ.write_text(json.dumps(rep, indent=2))
    print(f"[OK] report: {OUTJ}")


if __name__ == "__main__":
    main()
