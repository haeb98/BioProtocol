# prototype/tools/doc_search.py

import re


def doc_search(query, methods_text):
    sentences = re.split(r"\.\s+", methods_text)
    return "\n".join([s for s in sentences if query.lower() in s.lower()])
