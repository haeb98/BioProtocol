# src/graph_builder.py
from langgraph.graph import StateGraph, END
from src.nodes.action_extractor import action_extractor_node
from src.nodes.condition_extractor import condition_extractor_node
from src.nodes.step_planner import step_planner_node
from src.nodes.task_planner import task_planner_node
from src.types import GraphState


def identity_node(state: GraphState) -> GraphState:
    # 그냥 state를 그대로 넘기는 테스트용 노드
    return state


def build_graph():
    g = StateGraph(GraphState)

    g.add_node("task_planner", task_planner_node)
    g.add_node("step_planner", step_planner_node)
    g.add_node("condition_extractor", condition_extractor_node)
    g.add_node("action_extractor", action_extractor_node)

    g.set_entry_point("task_planner")
    g.add_edge("task_planner", "step_planner")
    g.add_edge("step_planner", "action_extractor")
    g.add_edge("action_extractor", END)
    return g.compile()
