# agents/s00_writer_methods_only.py
import json
import orjson
import os
import pathlib
import re
import time

from tqdm import tqdm

# OpenAI client (new or legacy)
try:
    from openai import OpenAI

    _HAS_NEW = True
except Exception:
    import openai

    _HAS_NEW = False

ROOT = pathlib.Path(".")
PAIRS = ROOT / "data/gold/gold_pairs_pmc_bioprotocol_balanced.jsonl"
OUT = ROOT / "eval/s00_generations.jsonl"
DBG = ROOT / "eval/_debug_s00"
OUT.parent.mkdir(parents=True, exist_ok=True)
DBG.mkdir(parents=True, exist_ok=True)

MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
SELF_WIN = int(os.environ.get("S00_WIN", "1400"))
SELF_STEP = int(os.environ.get("S00_STEP", "1100"))
MAX_EVID = int(os.environ.get("S00_MAX_EVID", "8"))
MAX_TOK = int(os.environ.get("S00_MAX_TOK", "1600"))
SLEEP = float(os.environ.get("S00_SLEEP", "0.05"))

# ---------- Prompt ----------
SYS = (
    "You are a biomedical protocol writer.\n"
    "Using ONLY the provided METHODS evidence, produce a hierarchical protocol JSON with this shape:\n"
    "{\"hierarchical_protocol\": {\"1\": {\"title\": \"Preparation\"}, \"1.1\": \"...step...\", \"2\": {\"title\":\"Procedure\"}, \"2.1\": \"...\"}}\n"
    "Rules:\n"
    "- Do NOT invent organisms/strains/reagents/parameters; extract only if explicitly present in EVIDENCE.\n"
    "- Normalize units: µL/mL/°C/min/rpm/g/mM/µM/%.\n"
    "- Steps should be atomic actions with parameters (time, temp, volume, g, rpm, conc) when present.\n"
    "- If inputs/recipes are stated, list them as early Preparation steps.\n"
    "- Return ONLY JSON, no markdown or commentary."
)


def openai_client():
    return OpenAI() if _HAS_NEW else openai.OpenAI()


def chat(client, system, user, max_tokens=MAX_TOK, temperature=0.0):
    r = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=temperature,
        max_tokens=max_tokens
    )
    return r.choices[0].message.content


def parse_json(s):
    try:
        return json.loads(s)
    except Exception:
        m = re.search(r"\{[\s\S]*\}", s or "")
        if not m: return None
        try:
            return json.loads(m.group(0))
        except Exception:
            return None


def windows(s, w=SELF_WIN, step=SELF_STEP):
    s = s or ""
    if len(s) <= w: return [s]
    out = [];
    i = 0
    while i < len(s):
        out.append(s[i:i + w])
        i += step
    return out


def pick_methods(article: dict):
    # Methods가 우선, 없으면 Abstract/Intro 순으로 폴백 최소화
    for key in ["methods_text", "abstract_text", "intro_text"]:
        t = (article or {}).get(key, "")
        if t and len(t.strip()) > 0:
            return t.strip()
    # 최후 수단: article의 여러 텍스트를 합쳐 짧게
    parts = []
    for key in ["results_text", "discussion_text", "conclusion_text"]:
        t = (article or {}).get(key, "")
        if t: parts.append(t)
    return ("\n\n".join(parts))[:5000]


def build_evidence_text(methods, max_evid=MAX_EVID):
    chunks = windows(methods)
    chunks = chunks[:max_evid]
    ev = "\n\n".join([f"[EVIDENCE {i + 1}] {c}" for i, c in enumerate(chunks)])
    return ev


def hier_or_minimal(obj):
    if isinstance(obj, dict) and "hierarchical_protocol" in obj and isinstance(obj["hierarchical_protocol"], dict):
        return obj
    # minimal skeleton
    return {"hierarchical_protocol": {"1": {"title": "Preparation"}, "2": {"title": "Procedure"}}}


def main():
    cli = openai_client()
    with open(PAIRS, "r", encoding="utf-8") as fin, open(OUT, "w", encoding="utf-8") as fou:
        for line in tqdm(fin, desc="s00-write(methods-only)"):
            pair = orjson.loads(line)
            pid = str(pair.get("protocol_id"))
            prot = pair.get("protocol") or {}
            art = pair.get("article") or {}

            methods = pick_methods(art)
            if not methods:
                # save minimal
                rec = {
                    "protocol_id": pid,
                    "domain": pair.get("domain", "Unknown"),
                    "gen_hier": {"1": {"title": "Preparation"}, "2": {"title": "Procedure"}},
                    "gold_hier": prot.get("hierarchical_protocol"),
                    "gold_text": prot.get("text", "")
                }
                fou.write(orjson.dumps(rec).decode() + "\n")
                continue

            ev = build_evidence_text(methods)
            title = prot.get("title") or art.get("title") or ""
            user = f"STUDY TITLE: {title}\n\nEVIDENCE (METHODS):\n{ev}"

            raw = chat(cli, SYS, user)
            obj = parse_json(raw)
            dbg = {"pid": pid, "raw_len": len(raw or ""), "had_methods": bool(methods)}
            (DBG / f"{pid}.json").write_text(json.dumps(dbg, ensure_ascii=False, indent=2), encoding="utf-8")

            hp = hier_or_minimal(obj)
            rec = {
                "protocol_id": pid,
                "domain": pair.get("domain", "Unknown"),
                "gen_hier": hp.get("hierarchical_protocol"),
                "gold_hier": prot.get("hierarchical_protocol"),
                "gold_text": prot.get("text", "")
            }
            fou.write(orjson.dumps(rec).decode() + "\n")
            time.sleep(float(SLEEP))
    print(f"[OK] wrote {OUT}")


if __name__ == "__main__":
    main()
