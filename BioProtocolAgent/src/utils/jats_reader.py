# src/utils/jats_reader.py
import os
import xml.etree.ElementTree as ET
from typing import List, Tuple

# 현재 파일: BioProtocol/BioProtocolAgent/src/utils/jats_reader.py
# BASE_DIR = BioProtocol/BioProtocolAgent/src
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 최상위 BioProtocol 폴더로 이동
# BASE_DIR (= .../BioProtocol/BioProtocolAgent/src)
# → dirname → .../BioProtocol/BioProtocolAgent
# → dirname → .../BioProtocol
PROJECT_ROOT = os.path.dirname(os.path.dirname(BASE_DIR))

# 최종적인 JATS XML 경로
PMC_JATS_DIR = os.path.join(PROJECT_ROOT, "data", "gold", "pmc_jats")

print("[DEBUG] PMC_JATS_DIR =", PMC_JATS_DIR)


def _strip_tag(tag: str) -> str:
    """{namespace}tag 형태에서 tag 부분만 떼기"""
    if "}" in tag:
        return tag.split("}", 1)[1]
    return tag


def load_jats_xml(pmcid: str) -> ET.ElementTree:
    """
    pmcid (예: 'PMC1234567')를 받아서 JATS XML 파싱 결과 리턴.
    """
    xml_path = os.path.join(PMC_JATS_DIR, f"{pmcid}.xml")
    if not os.path.exists(xml_path):
        raise FileNotFoundError(f"JATS XML not found: {xml_path}")
    tree = ET.parse(xml_path)
    return tree


def extract_article_spans(tree: ET.ElementTree, max_len: int = 2000) -> List[Tuple[str, str]]:
    """
    논문 전체에서 검색 대상으로 쓸 (section_label, text) 리스트를 만든다.

    - section_label: 대략적인 섹션 이름 (abstract, methods, results, fig, table 등)
    - text: 해당 부분의 텍스트 (길면 max_len만큼 잘라서 사용)
    """
    root = tree.getroot()
    spans: List[Tuple[str, str]] = []

    # 모든 자식 노드를 순회하면서 텍스트 추출
    # 너무 복잡한 XPath 대신, 태그 이름으로 대충 섹션 역할을 구분
    for elem in root.iter():
        tag = _strip_tag(elem.tag)

        # 너무 짧거나 의미 없는 태그는 건너뛰기
        if tag in {"article", "front", "back"}:
            continue

        # 텍스트 추출
        text = "".join(elem.itertext()).strip()
        if not text:
            continue

        # 공백 정리
        text = " ".join(text.split())
        if not text:
            continue

        # 섹션 라벨 대략 지정
        if tag in {"title"}:
            label = "title"
        elif tag in {"abstract"}:
            label = "abstract"
        elif tag in {"sec"}:
            label = "sec"
        elif tag in {"p"}:
            label = "paragraph"
        elif tag in {"fig", "fig-group"}:
            label = "figure"
        elif tag in {"table-wrap"}:
            label = "table"
        else:
            label = tag

        spans.append((label, text[:max_len]))

    return spans
