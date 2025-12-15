# prototype/tools/doc_search.py

import re


def doc_search(query, methods_text):
    """
    Retrieve relevant sentences from methods text that contain the query term (case-insensitive).
    Splits methods_text into sentences and returns those containing the query.
    """
    sentences = re.split(r'(?<=[.?!])\s+', methods_text.strip())
    results = [s for s in sentences if query.lower() in s.lower()]
    return "\n".join(results)
