# src/types.py
from typing import TypedDict, List, Dict, Any, Optional


class ParameterIR(TypedDict, total=False):
    name: str
    value: Optional[float]
    unit: Optional[str]
    raw_text: Optional[str]
    supported_by: List[str]


# src/types.py

class MaterialIR(TypedDict, total=False):
    name: str  # "RNA sample", "8 M LiCl", "RNase-free water"
    role: str  # "substrate", "reagent", "buffer", ...
    volume: str  # "0.1 volume", "1 volume", "200 μL"
    concentration: str  # "8 M", "3%", ...
    state: str  # "on ice", "at RT" 같은 상태
    source: str  # "methods_text" or "llm_inferred"


class ConditionIR(TypedDict, total=False):
    type: str  # "temperature", "duration", "speed", "pH", ...
    value: str  # "2 hours", "on ice", "37 °C"
    unit: str  # "h", "°C", "rpm" 등 (없으면 빈 문자열)
    source: str  # "methods_text" or "llm_inferred"


class ActionIR(TypedDict, total=False):
    action_id: str  # "Bio-protocol-2302::T3::S1::A1" 같은 식
    step_id: str
    protocol_id: str
    task_id: str

    action: str  # "Add", "Incubate", "Mix", "Centrifuge" 등
    description: str  # 자연어 설명 (step_text의 부분문장 등)

    materials: List[MaterialIR]
    conditions: List[ConditionIR]
    produces: List[str]  # "RNA in diluted LiCl" 같은 결과물

    evidence_span: str  # methods_text에서 근거가 된 span
    source: str  # "llm_extracted"


class StepIR(TypedDict, total=False):
    step_id: str
    task_id: str
    title: Optional[str]
    action: Optional[str]
    step_text: str
    step_rationale: Optional[str]
    span_chunk: Optional[str]
    materials: List[str]
    parameters: List[ParameterIR]
    evidence_spans: List[str]
    verified: bool
    verification_notes: Optional[str]


class GraphState(TypedDict, total=False):
    protocol_id: str

    bio: Dict[str, Any]
    article: Dict[str, Any]
    methods_text: str

    tasks: List[Dict[str, Any]]
    steps_raw: List[StepIR]
    steps: List[StepIR]
    actions: List[ActionIR]

    protocol_draft: str

    tool_logs: List[Dict[str, Any]]
    verification_logs: List[Dict[str, Any]]
