# agents/planner_state.py

from typing import List, Optional, Dict

from pydantic import BaseModel, Field


class Task(BaseModel):
    """
    Task Planner가 뽑는 하나의 태스크 단위.
    gold_tasks_testset.jsonl의 형식과 최대한 맞춤.
    """
    id: str = Field(..., description="예: 'T1', 'T2' 등")
    title: str = Field(..., description="간단한 태스크 이름")
    description: str = Field(..., description="실험자가 이해할 수 있는 태스크 설명")
    type: Optional[str] = Field(
        default=None,
        description="optional: 'experiment', 'preparation', 'qc', 'analysis' 등 태스크 타입"
    )


class PlannerState(BaseModel):
    """
    Task Planner 노드가 읽고/쓰는 상태.
    나중에 LangGraph의 State로 그대로 사용할 수 있게 설계.
    """
    protocol_id: str
    methods_text: str

    # 설정값들
    rag_enabled: bool = True
    max_tasks: int = 12

    # Task Planner의 출력
    tasks_planned: List[Task] = Field(
        default_factory=list,
        description="Planner가 최종적으로 생성한 태스크 리스트"
    )

    # 디버깅 / 연구용 로그: 프롬프트, LLM raw 응답 등
    planner_raw: Optional[Dict] = Field(
        default=None,
        description="프롬프트, LLM 응답 전체 등 디버깅용 raw 정보"
    )
