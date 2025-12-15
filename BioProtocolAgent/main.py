# main.py
from src.data_loader import make_initial_state
from src.graph_builder import build_graph

if __name__ == "__main__":
    graph = build_graph()

    pid = "Bio-protocol-2302"
    init_state = make_initial_state(pid)
    final_state = graph.invoke(init_state)

    print("=== Protocol ID ===")
    print(final_state["protocol_id"])
    print(final_state["bio"]["title"])
    print()

    for act in final_state["actions"]:
        print("==", act["action_id"], "==")
        print("ACTION :", act["action"])
        print("DESC   :", act["description"])
        print("MATS   :", act["materials"])
        print("CONDS  :", act["conditions"])
        print("PROD   :", act["produces"])
        print("EVID   :", act["evidence_span"])
        print()
