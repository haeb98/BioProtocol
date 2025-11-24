# agents/step_state.py

from typing import List, Optional, Dict

from pydantic import BaseModel, Field

from .planner_state import Task


class Step(BaseModel):
    """
    하나의 실험 Step.
    - gold hierarchical_protocol에서의 step 구조와 맞춰갈 수 있도록 설계
    """
    id: str = Field(..., description="예: 'S1', 'S2' 등")
    task_id: Optional[str] = Field(
        default=None,
        description="이 step이 속한 상위 Task ID (예: 'T1'). 없으면 None."
    )
    title: str = Field(..., description="간단한 step 이름")
    description: str = Field(..., description="실험자가 바로 따라 할 수 있는 설명(짧은 문장)")
    step_type: Optional[str] = Field(
        default=None,
        description="optional: 'procedure', 'qc', 'analysis', 'setup' 등"
    )


class StepState(BaseModel):
    """
    Step Structurer 노드의 입출력 State.
    나중에 LangGraph에서 이걸 state로 써도 됨.
    """
    protocol_id: str
    methods_text: str

    # Task Planner 출력 (입력으로 받음)
    tasks_planned: List[Task] = Field(default_factory=list)

    # 설정값
    max_steps_per_task: int = 8

    # Step Structurer 출력
    steps_structured: List[Step] = Field(
        default_factory=list,
        description="구조화된 step 리스트"
    )

    # 디버깅 / 연구용 raw 정보
    step_raw: Optional[Dict] = Field(
        default=None,
        description="LLM 프롬프트, 응답 원본 등 디버깅용"
    )
