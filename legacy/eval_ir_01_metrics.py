# eval/eval_ir_01_metrics.py
import argparse
import json
import re
import statistics as st
from pathlib import Path


def flatten_gold_steps(hier):
    # "1","1.1","1.2"... 의 leaf 문장만
    steps = []
    for k, v in hier.items():
        if isinstance(v, str):
            steps.append({"sid": k, "text": v})
    return steps


def normalize(t):
    return re.sub(r"[^a-z0-9]+", " ", t.lower()).strip()


def jaccard(a, b):
    A = set(a.split());
    B = set(b.split())
    return len(A & B) / (len(A | B) + 1e-9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="data/gold/gold_pairs_testset_top15.jsonl")
    ap.add_argument("--pred", default="runs/s2_llm_top15.ir.jsonl")
    ap.add_argument("--out", default="reports/ir_eval_a_step.csv")
    args = ap.parse_args()
    Path("reports").mkdir(exist_ok=True, parents=True)

    gold = []
    for line in Path(args.pairs).read_text().splitlines():
        r = json.loads(line);
        gold.append(r)

    pred_by_doc = {}
    for line in Path(args.pred).read_text().splitlines():
        p = json.loads(line);
        pred_by_doc[p["doc_id"]] = p

    rows = []
    for r in gold:
        doc = r["article"]["id"]
        gsteps = flatten_gold_steps(r["protocol"]["hierarchical_protocol"])
        gtexts = [normalize(s["text"]) for s in gsteps]

        p = pred_by_doc.get(doc, {"steps": []})
        psteps = p.get("steps", [])
        # 간단 overlap / param coverage
        acts = [normalize(s.get("action", "")) for s in psteps if s.get("action")]
        mats = [normalize(m) for s in psteps for m in s.get("materials", [])]
        params = [pp for s in psteps for pp in s.get("parameters", [])]
        param_cov = sum(
            1 for pp in params if pp.get("value") not in (None, "") and str(pp.get("unit", "")).strip() != "")
        # step 매칭: 문장 유사도 최대값 평균
        step_sim = []
        for gt in gtexts:
            best = 0.0
            for s in psteps:
                txt = normalize(" ".join([s.get("action", "")] + s.get("materials", []) + [
                    ", ".join([f"{x.get('value', '')} {x.get('unit', '')}" for x in s.get('parameters', [])])]))
                best = max(best, jaccard(gt, txt))
            step_sim.append(best)

        rows.append({
            "protocol_id": r["protocol_id"],
            "pmcid": r["article"]["id"],
            "action_overlap": len([a for a in acts if a]),
            "material_overlap": len([m for m in mats if m]),
            "param_coverage": param_cov,
            "avg_step_match": st.mean(step_sim) if step_sim else 0.0,
            "n_pred_steps": len(psteps),
            "n_gold_steps": len(gsteps),
        })

    # 저장
    import csv
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader();
        w.writerows(rows)
    print(f"[OK] wrote -> {args.out}")


if __name__ == "__main__":
    main()
