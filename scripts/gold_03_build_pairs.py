# 입력: data/gold/gold_articles.jsonl, data/raw/bio_protocol.json
# 출력: data/gold/gold_pairs.jsonl
import orjson
import pathlib

ROOT = pathlib.Path(".")
ART = ROOT / "data/gold/gold_articles.jsonl"
BIO = ROOT / "data/raw/bio_protocol.json"
OUT = ROOT / "data/gold/gold_pairs.jsonl"
OUT.parent.mkdir(parents=True, exist_ok=True)


def flatten_hier(h):
    if not isinstance(h, dict): return ""

    def keyer(k):
        return [int(p) if p.isdigit() else p for p in k.split(".")]

    lines = []
    for k in sorted(h.keys(), key=keyer):
        v = h[k]
        if isinstance(v, dict) and "title" in v:
            lines.append(f"## {v['title']}")
        elif isinstance(v, str):
            lines.append(v)
    return "\n".join(lines)


def build_protocol_text(r):
    title = r.get("title") or ""
    inputs = r.get("input") or ""
    proto = r.get("protocol") or ""
    hier = flatten_hier(r.get("hierarchical_protocol") or {})
    parts = [p for p in [title, "# Materials & Inputs", inputs, "# Protocol", proto, "# Outline", hier] if
             p and str(p).strip()]
    return "\n\n".join(parts)


def main():
    bio = orjson.loads(BIO.read_bytes())
    bio_idx = {str(x.get("id")): x for x in bio}
    out = open(OUT, "w", encoding="utf-8")
    n = 0
    with open(ART, "r", encoding="utf-8") as f:
        for line in f:
            a = orjson.loads(line)
            pid = a["protocol_id"]  # e.g., Bio-protocol-5128
            br = bio_idx.get(pid)
            if not br: continue
            pair = {
                "protocol_id": pid,
                "domain": (br.get("classification") or {}).get("primary_domain") or "Unknown",
                "article": {
                    "id": a["article_id"],
                    "text": a["text"],
                    "meta": a.get("meta", {})
                },
                "protocol": {
                    "title": br.get("title") or "",
                    "text": build_protocol_text(br),
                    "keywords": br.get("keywords") or "",
                    "url": br.get("url") or ""
                }
            }
            out.write(orjson.dumps(pair).decode() + "\n");
            n += 1
    out.close()
    print("[OK]", OUT, "(pairs=", n, ")")


if __name__ == "__main__": main()
