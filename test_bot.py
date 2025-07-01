from agents.graph_builder import build_graph

graph = build_graph()

# Round 1
state = {
    "input": "I'm feeling shortness of breath when walking",
    "memory": {}
}
result = graph.invoke(state)

print("🤖 Bot:", result["response"])

# Round 2: user gives more detail
state = {
    "input": "I've had it for 3 days.",
    "memory": result["memory"]
}
result2 = graph.invoke(state)

print("🤖 Bot:", result2["response"])
