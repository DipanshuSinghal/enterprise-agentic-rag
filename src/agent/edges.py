from src.agent.state import AgentState

def route_after_grading(state: AgentState) -> str:
    """Decides whether to generate an answer or loop back to optimize the search query."""
    last_step = state["steps"][-1]
    
    if last_step == "generate":
        return "generate"
    else:
        return "transform_query"