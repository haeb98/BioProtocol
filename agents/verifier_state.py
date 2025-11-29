from typing import List, Dict

from pydantic import BaseModel, Field


class VerifierState(BaseModel):
    protocol_id: str
    methods_text: str
    param_table: List[Dict]
    param_verdicts: List[Dict] = Field(default_factory=list)

    @classmethod
    def from_ir_state(cls, ir_state):
        return cls(
            protocol_id=ir_state.protocol_id,
            methods_text=ir_state.methods_text,
            param_table=ir_state.param_table
        )
