# scripts/check_goldpair_sanity.py
import orjson
import pathlib

from sentence_transformers import SentenceTransformer, util

ROOT = pathlib.Path(".")
PAIRS = ROOT / "data/gold/gold_pairs_pmc_bioprotocol_balanced.jsonl"
m = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

bad = []
for line in open(PAIRS, "r", encoding="utf-8"):
    o = orjson.loads(line)
    pid = str(o["protocol_id"])
    pt = (o.get("protocol") or {}).get("title", "")
    at = (o.get("article") or {}).get("title", "")
    if not pt or not at:
        bad.append((pid, "missing title"));
        continue
    s = util.cos_sim(m.encode(pt, normalize_embeddings=True),
                     m.encode(at, normalize_embeddings=True))[0][0].item()
    if s < 0.25: bad.append((pid, f"low_title_sim={s:.2f}"))
print("[suspects]", len(bad))
for x in bad[:30]: print(x)
