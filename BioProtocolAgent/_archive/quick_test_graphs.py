# quick_test_graphs.py (원하면 src/ 밑에 두고 python -m src.quick_test_graphs 로 실행)
from src.data_loader import make_initial_state
from src.graph_builder import build_graph
from src.graph_builder_A import build_graph_A

if __name__ == "__main__":
    pid = "Bio-protocol-2219"  # gold_pairs에 있는 프로토콜ID 중 하나

    state = make_initial_state(pid)

    graph_A = build_graph_A()
    out_A = graph_A.invoke(state)

    graph_B = build_graph()
    out_B = graph_B.invoke(state)

    print("=== A: Methods → Step Structurer ===")
    print("Tasks:", [t["task_name"] for t in out_A["tasks"]])
    print("Steps:", [s["step_id"] for s in out_A["steps_raw"]][:5])

    print("\n=== B: Task → Step Structurer ===")
    print("Tasks:", [t["task_name"] for t in out_B["tasks"]])
    print("Steps:", [s["step_id"] for s in out_B["steps_raw"]][:5])
