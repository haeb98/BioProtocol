# src/graph_builder_ablation.py
from langgraph.graph import StateGraph
from src.nodes.action_extractor import action_extractor_node
from src.nodes.step_planner import step_planner_node
from src.nodes.task_planner import task_planner_node
from src.nodes.verifier_react import verifier_node  # 앞에서 만든 ReAct verifier
from src.nodes.writer import writer_node_simple, writer_node_prompted, writer_node_skeleton
from src.types import GraphState


def build_graph(mode: str):
    """
    mode:
      P1 = methods -> writer_simple
      P2 = methods -> writer_prompted
      P3 = methods -> task_miner -> writer_skeleton
      P4 = methods -> task_miner -> step_planner -> writer_skeleton
      P5 = methods -> task_miner -> step_planner -> action_extractor -> writer_skeleton
      P6 = methods -> task_miner -> step_planner -> action_extractor -> verifier -> writer_skeleton
    """
    g = StateGraph(GraphState)

    # 항상 있는 입력 노드는 생략 (data_loader에서 initial state 세팅)

    if mode == "P1":
        g.add_node("writer", writer_node_simple)
        g.set_entry_point("writer")
        g.set_finish_point("writer")

    elif mode == "P2":
        g.add_node("writer", writer_node_prompted)
        g.set_entry_point("writer")
        g.set_finish_point("writer")

    elif mode == "P3":
        g.add_node("task_miner", task_planner_node)
        g.add_node("writer", writer_node_skeleton)
        g.set_entry_point("task_miner")
        g.add_edge("task_miner", "writer")
        g.set_finish_point("writer")

    elif mode == "P4":
        g.add_node("task_miner", task_planner_node)
        g.add_node("step_planner", step_planner_node)
        g.add_node("writer", writer_node_skeleton)
        g.set_entry_point("task_miner")
        g.add_edge("task_miner", "step_planner")
        g.add_edge("step_planner", "writer")
        g.set_finish_point("writer")

    elif mode == "P5":
        g.add_node("task_miner", task_planner_node)
        g.add_node("step_planner", step_planner_node)
        g.add_node("action_extractor", action_extractor_node)
        g.add_node("writer", writer_node_skeleton)
        g.set_entry_point("task_miner")
        g.add_edge("task_miner", "step_planner")
        g.add_edge("step_planner", "action_extractor")
        g.add_edge("action_extractor", "writer")
        g.set_finish_point("writer")

    elif mode == "P6":
        g.add_node("task_miner", task_planner_node)
        g.add_node("step_planner", step_planner_node)
        g.add_node("action_extractor", action_extractor_node)
        g.add_node("verifier", verifier_node)
        g.add_node("writer", writer_node_skeleton)

        g.set_entry_point("task_miner")
        g.add_edge("task_miner", "step_planner")
        g.add_edge("step_planner", "action_extractor")
        g.add_edge("action_extractor", "verifier")
        g.add_edge("verifier", "writer")
        g.set_finish_point("writer")

    else:
        raise ValueError(f"Unknown mode: {mode}")

    return g.compile()
