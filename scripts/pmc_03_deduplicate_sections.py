import argparse
import json
import re

from tqdm import tqdm


def truncate_at_duplicate(text: str, section_title: str, lookahead_words=40):
    """
    섹션 제목 다음 첫 문단(40단어 정도) 기준으로 중복되는 지점 이후를 잘라냄
    """
    pattern = rf"{re.escape(section_title)}\n(.+?)"
    match = re.search(pattern, text, re.DOTALL)
    if not match:
        return text

    after_title = match.group(1).strip()
    seed_words = " ".join(after_title.split()[:lookahead_words])
    if not seed_words:
        return text

    first_index = text.find(seed_words)
    second_index = text.find(seed_words, first_index + len(seed_words))

    if second_index != -1:
        return text[:second_index].rstrip()
    return text


def process_record(record):
    new_sections = {}
    new_stats = {}
    for sec_title, sec_text in record.get("sections", {}).items():
        deduped = truncate_at_duplicate(sec_text, sec_title)
        new_sections[sec_title] = deduped
        new_stats[sec_title] = {
            "chars": len(deduped),
            "tokens": len(deduped.split()),
            "source": record.get("stats", {}).get(sec_title, {}).get("source", "deduped")
        }
    record["sections"] = new_sections
    record["stats"] = new_stats
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="input_path", required=True, help="Input JSONL file")
    parser.add_argument("--out", dest="output_path", required=True, help="Output JSONL file")
    args = parser.parse_args()

    with open(args.input_path, "r", encoding="utf-8") as f:
        lines = [json.loads(line) for line in f if line.strip()]

    with open(args.output_path, "w", encoding="utf-8") as out_f:
        for record in tqdm(lines, desc="Deduplicating records"):
            cleaned = process_record(record)
            out_f.write(json.dumps(cleaned, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
