from fastapi import Request
from langgraph.graph.state import CompiledStateGraph


def get_graph(request: Request) -> CompiledStateGraph:
    return request.app.state.graph
