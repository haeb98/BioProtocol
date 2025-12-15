# src/graph_builder_A.py
from langgraph.graph import StateGraph, END
from src.nodes.step_planner import step_planner_node
from src.types import GraphState


def dummy_task_planner_node(state: GraphState) -> GraphState:
    """
    Task Planner 없이 전체 methods를 하나의 큰 task로 보는 A조건용 노드.
    """
    protocol_id = state["protocol_id"]
    bio_title = state["bio"]["title"]

    tasks = [{
        "task_id": f"{protocol_id}::T1",
        "task_name": f"Full protocol for {bio_title}",
        "description": "Overall experimental procedure as described in the methods.",
        "span_chunk": state["methods_text"][:2000],
        "protocol_id": protocol_id,
    }]

    new_state = dict(state)
    new_state["tasks"] = tasks
    return new_state


def build_graph_A():
    g = StateGraph(GraphState)

    g.add_node("dummy_task_planner", dummy_task_planner_node)
    g.add_node("step_planner", step_planner_node)

    g.set_entry_point("dummy_task_planner")
    g.add_edge("dummy_task_planner", "step_planner")
    g.add_edge("step_planner", END)

    return g.compile()
