from typing import List, Dict

from pydantic import BaseModel, Field


class Param(BaseModel):
    param_id: str
    step_id: str
    name: str
    value: str
    unit: str
    span: str


class IRNode(BaseModel):
    id: str
    action: str
    params: List[Param] = []


class IRState(BaseModel):
    protocol_id: str
    methods_text: str
    steps_structured: List[Dict] = Field(default_factory=list)
    ir_nodes: List[IRNode] = Field(default_factory=list)
    ir_edges: List[List[str]] = Field(default_factory=list)
    param_table: List[Param] = Field(default_factory=list)

    @classmethod
    def from_step_state(cls, step_state):
        return cls(
            protocol_id=step_state.protocol_id,
            methods_text=step_state.methods_text,
            steps_structured=step_state.steps_structured
        )
