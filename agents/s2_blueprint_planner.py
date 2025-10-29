# agents/s2_blueprint_planner.py
# ... (파일 헤더/주석 동일) ...
import argparse
import json
import re
from collections import defaultdict, deque


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            yield json.loads(line)


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def norm(s):
    return (s or "").strip().lower()


# ---------- 라이트한 문자열→IR 변환기 ----------
VALUNIT_PAT = re.compile(r"(?P<val>\d+(\.\d+)?)\s*(?P<unit>(mM|M|μ?g/?mL|mg/?mL|μ?L|mL|%|rpm|×\s?g|g))", re.I)
TIME_PAT = re.compile(r"(?P<time>\d+(\.\d+)?)\s*(s|sec|secs|second|seconds|m|min|mins|minute|minutes|h|hr|hour|hours)",
                      re.I)
TEMP_PAT = re.compile(r"(?P<temp>-?\d+(\.\d+)?)\s*°?\s*C", re.I)
RPM_PAT = re.compile(r"(?P<rpm>\d{2,5})\s*rpm", re.I)
G_PAT = re.compile(r"(?P<g>\d+(\.\d+)?)\s*×\s*g", re.I)

COMMON_VERBS = {
    "add", "mix", "vortex", "incubate", "centrifuge", "wash", "rinse", "filter",
    "prepare", "resuspend", "dissolve", "measure", "read", "quantify",
    "inject", "mount", "stain", "dry", "equilibrate", "prewarm", "thaw", "shake", "stir"
}

HEADER_TOKENS = {"meta", "materials", "solutions", "equipment", "steps", "dependencies", "outputs"}


def coerce_str_to_ir(s):
    """아주 가벼운 휴리스틱: 맨 앞 단어가 동사면 action, 나머지는 material로 간주.
       숫자+단위/시간/온도/rpm/g 는 condition/value/unit으로 끌어온다."""
    txt = s.strip()
    if not txt: return None
    tokens = txt.split()
    if not tokens: return None

    # action 후보
    cand = tokens[0].lower().strip(",.")
    action = cand if cand in COMMON_VERBS else None
    material = " ".join(tokens[1:]).strip() if action else txt

    # 값/조건 추출
    value, unit = None, None
    m = VALUNIT_PAT.search(txt)
    if m:
        value = float(m.group("val"))
        unit = m.group("unit")

    cond = {}
    t = TIME_PAT.search(txt)
    if t: cond["time"] = t.group(0)
    tp = TEMP_PAT.search(txt)
    if tp: cond["temp"] = tp.group(0).replace(" ", "")
    rp = RPM_PAT.search(txt)
    if rp: cond["rpm"] = rp.group("rpm")
    gp = G_PAT.search(txt)
    if gp: cond["g"] = gp.group("g")

    return {
        "action": action or "step",
        "material": material if material else None,
        "value": value, "unit": unit,
        "tool": None,
        "condition": cond if cond else None,
        "depends_on": [],
        "evidence_id": None,
        "step_hint": None
    }


def coerce_triplet_like(obj):
    """(s,p,o), {"subject":..,"predicate":..,"object":..} 류를 IR로 변환"""
    if isinstance(obj, (list, tuple)) and len(obj) == 3:
        s, p, o = obj
        return {
            "action": str(p).lower(),
            "material": str(o),
            "value": None, "unit": None,
            "tool": None, "condition": None,
            "depends_on": [], "evidence_id": None, "step_hint": None
        }
    if isinstance(obj, dict) and {"predicate", "object"} <= set(obj.keys()):
        return {
            "action": str(obj.get("predicate") or "step").lower(),
            "material": str(obj.get("object") or "") or None,
            "value": obj.get("value"), "unit": obj.get("unit"),
            "tool": obj.get("tool"),
            "condition": obj.get("condition"),
            "depends_on": obj.get("depends_on") or [],
            "evidence_id": obj.get("evidence_id"), "step_hint": obj.get("step_hint")
        }
    return None


def sanitize_ir_list(irs):
    cleaned = []
    n_skipped = 0
    n_coerced = 0
    for it in irs:
        if isinstance(it, str):
            # dict인데 action 키가 없고 triplet 스타일이면 변환
            if "action" not in it:
                coerced = coerce_triplet_like(it)
                if coerced:
                    cleaned.append(coerced);
                    n_coerced += 1
                else:
                    # 최소 필드만 구성해서 살리기
                    cleaned.append({
                        "action": str(it.get("predicate") or it.get("verb") or "step").lower(),
                        "material": str(it.get("material") or it.get("object") or it.get("target") or "") or None,
                        "value": it.get("value"), "unit": it.get("unit"),
                        "tool": it.get("tool"), "condition": it.get("condition"),
                        "depends_on": it.get("depends_on") or [],
                        "evidence_id": it.get("evidence_id"), "step_hint": it.get("step_hint")
                    })
                    n_coerced += 1
            else:
                cleaned.append(it)
        elif isinstance(it, (list, tuple)):
            coerced = coerce_triplet_like(it)
            if coerced:
                cleaned.append(coerced);
                n_coerced += 1
            else:
                n_skipped += 1
        elif isinstance(it, str):
            coerced = coerce_str_to_ir(it)
            if coerced:
                cleaned.append(coerced);
                n_coerced += 1
            else:
                n_skipped += 1
        else:
            n_skipped += 1
    return cleaned, n_skipped, n_coerced


# ---------- 라벨/렌더링/카테고리/의존성/토폴로지 (원래 로직) ----------
def has_value(ir):
    return (ir.get("value") is not None) or any(v for v in (ir.get("condition") or {}).values())


def label_of(ir):
    act = (ir.get("action") or "step").strip().capitalize()
    mat = ir.get("material")
    val = ir.get("value")
    unit = ir.get("unit")
    cond = ir.get("condition") or {}
    bits = []
    if mat: bits.append(mat)
    if val is not None and unit:
        bits.append(f"({val} {unit})")
    elif val is not None:
        bits.append(f"({val})")
    cparts = []
    if cond.get("speed"): cparts.append(f"{cond['speed']}")
    if cond.get("g"): cparts.append(f"{cond['g']} × g")
    if cond.get("rpm"): cparts.append(f"{cond['rpm']} rpm")
    if cond.get("time"): cparts.append(f"{cond['time']}")
    if cond.get("temp"): cparts.append(f"{cond['temp']}")
    if cparts:
        bits.append(", ".join(cparts))
    body = " ".join(bits) if bits else ""
    return (f"{act} {body}".strip() if body else act)


def render_step(ir):
    act = norm(ir.get("action") or "step")
    mat = ir.get("material") or ""
    val = ir.get("value")
    unit = ir.get("unit")
    cond = ir.get("condition") or {}
    verb_map = {
        "sterilize": "Sterilize", "wash": "Wash", "rinse": "Rinse", "incubate": "Incubate",
        "centrifuge": "Centrifuge", "mix": "Mix", "add": "Add", "prepare": "Prepare",
        "dissolve": "Dissolve", "measure": "Measure", "read": "Read", "inject": "Inject",
        "resuspend": "Resuspend", "vortex": "Vortex", "filter": "Filter", "equilibrate": "Equilibrate",
        "prewarm": "Pre-warm", "dry": "Dry", "analyze": "Analyze", "quantify": "Quantify",
        "pipette": "Pipette", "mount": "Mount", "stain": "Stain", "sonicate": "Sonicate", "step": "Do"
    }
    head = verb_map.get(act, act.capitalize())
    parts = [head]
    if mat: parts.append(mat)
    if val is not None and unit: parts.append(f"{val} {unit}")
    tail = []
    if cond.get("g"):   tail.append(f"{cond['g']} × g")
    if cond.get("rpm"): tail.append(f"{cond['rpm']} rpm")
    if cond.get("temp"): tail.append(f"{cond['temp']}")
    if cond.get("time"): tail.append(f"{cond['time']}")
    if tail:
        if act in ("incubate", "centrifuge", "mix", "resuspend", "stir", "shake"):
            parts.append("for " + ", ".join(tail) if cond.get("time") else ", ".join(tail))
        else:
            parts.append("(" + ", ".join(tail) + ")")
    return " ".join([p for p in parts if p]).strip().rstrip(".") + "."


PREP_ACTIONS = {"sterilize", "prepare", "dissolve", "mix", "resuspend", "equilibrate", "prewarm", "thaw", "buffer",
                "solution"}
ANALYSIS_ACTIONS = {"measure", "read", "quantify", "analyze", "inject", "calculate"}


def guess_dep_edges(irs):
    n = len(irs)
    producers = defaultdict(list)
    for i, ir in enumerate(irs):
        act = norm(ir.get("action"))
        mat = norm(ir.get("material"))
        if mat and (act in PREP_ACTIONS or act.startswith("prepare")):
            producers[mat].append(i)
    edges = set()
    for i, ir in enumerate(irs):
        for d in ir.get("depends_on") or []:
            if isinstance(d, int) and 0 <= d < n:
                edges.add((d, i))
    for i, ir in enumerate(irs):
        mat = norm(ir.get("material"))
        if mat and mat in producers:
            for pidx in producers[mat]:
                if pidx != i:
                    edges.add((pidx, i))

    def cat(ir):
        a = norm(ir.get("action"))
        if a in PREP_ACTIONS: return 0
        if a in ANALYSIS_ACTIONS: return 2
        return 1

    for i in range(n):
        for j in range(i + 1, n):
            if cat(irs[i]) < cat(irs[j]):
                edges.add((i, j))
    return sorted({(u, v) for (u, v) in edges if u != v})


def topo_sort(n, edges):
    g = defaultdict(list)
    indeg = [0] * n
    for u, v in edges: g[u].append(v); indeg[v] += 1
    q = deque([i for i in range(n) if indeg[i] == 0])
    order = []
    while q:
        u = q.popleft();
        order.append(u)
        for v in g[u]:
            indeg[v] -= 1
            if indeg[v] == 0: q.append(v)
    if len(order) < n:
        remain = [i for i in range(n) if i not in order]
        order.extend(remain)
    return order


def section_of(ir):
    a = norm(ir.get("action"))
    if a in PREP_ACTIONS: return "prep"
    if a in ANALYSIS_ACTIONS: return "analysis"
    return "proc"


def reindex_plan(steps):
    out = [];
    sec_idx = 0;
    sub_idx = 0
    for s in steps:
        if "title" in s:
            sec_idx += 1;
            sub_idx = 0
            out.append({"sid": f"{sec_idx}", "title": s["title"]})
        else:
            sub_idx += 1
            out.append({"sid": f"{sec_idx}.{sub_idx}", **s})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="eval/s2_ir.jsonl")
    ap.add_argument("--out", dest="out", default="eval/s2_blueprints.jsonl")
    ap.add_argument("--prep_keywords", default="sterilize,prepare,buffer,solution,prewarm,thaw")
    ap.add_argument("--analysis_keywords", default="measure,read,quantify,analyze,calculate")
    args = ap.parse_args()

    global PREP_ACTIONS, ANALYSIS_ACTIONS
    PREP_ACTIONS |= {k.strip() for k in args.prep_keywords.split(",") if k.strip()}
    ANALYSIS_ACTIONS |= {k.strip() for k in args.analysis_keywords.split(",") if k.strip()}

    rows_out = []
    for rec in read_jsonl(args.inp):
        pid = rec.get("protocol_id") or rec.get("id")
        domain = rec.get("domain")
        irs_raw = rec.get("ir") or []
        irs, n_skipped, n_coerced = sanitize_ir_list(irs_raw)

        if not irs:
            rows_out.append({
                "protocol_id": pid, "domain": domain,
                "stats": {"num_ir": 0, "num_nodes": 0, "num_edges": 0,
                          "unresolved_params": 0, "pct_with_values": 0.0,
                          "coerced": n_coerced, "skipped": n_skipped},
                "dag": {"nodes": [], "edges": []},
                "plan": [{"sid": "1", "title": "Preparation & Materials"},
                         {"sid": "2", "title": "Procedure"},
                         {"sid": "3", "title": "Analysis / Readout"}]
            })
            continue

        edges = guess_dep_edges(irs)
        n = len(irs)
        order = topo_sort(n, edges)

        nodes = [];
        idmap = {}
        for rank, idx in enumerate(order, start=1):
            nid = f"n{rank}";
            idmap[idx] = nid
            nodes.append({"id": nid, "ir_idx": idx, "label": label_of(irs[idx])})
        dag_edges = [[idmap[u], idmap[v]] for (u, v) in edges if u in idmap and v in idmap]

        prep_steps = [];
        proc_steps = [];
        anal_steps = []
        for idx in order:
            ir = irs[idx]
            target = section_of(ir)
            step = {
                "text": render_step(ir),
                "evidence_id": ir.get("evidence_id"),
                "ir_idx": idx
            }
            if target == "prep":
                prep_steps.append(step)
            elif target == "analysis":
                anal_steps.append(step)
            else:
                proc_steps.append(step)

        plan = []
        if prep_steps:
            plan.append({"title": "Preparation & Materials"})
            plan += prep_steps
        plan.append({"title": "Procedure"})
        plan += proc_steps
        if anal_steps:
            plan.append({"title": "Analysis / Readout"})
            plan += anal_steps

        plan = reindex_plan(plan)

        unresolved = 0
        valued = 0
        for ir in irs:
            valued += 1 if has_value(ir) else 0
            if (ir.get("value") is None) and not any(v for v in (ir.get("condition") or {}).values()):
                unresolved += 1

        stats = {
            "num_ir": len(irs),
            "num_nodes": len(nodes),
            "num_edges": len(dag_edges),
            "unresolved_params": unresolved,
            "pct_with_values": round(valued / max(1, len(irs)), 4),
            "coerced": n_coerced,
            "skipped": n_skipped
        }

        rows_out.append({
            "protocol_id": pid,
            "domain": domain,
            "stats": stats,
            "dag": {"nodes": nodes, "edges": dag_edges},
            "plan": plan
        })

    write_jsonl(args.out, rows_out)
    print(f"[OK] saved blueprints -> {args.out}")


if __name__ == "__main__":
    main()
