from langgraph.graph import StateGraph, END
from src.agent.state import AgentState
from src.agent.nodes import GraphNodes
from src.agent.edges import route_after_grading

def build_agent_graph():
    """Compiles individual node functions and routing criteria into a LangGraph runner."""
    # 1. Initialize the state tracker scheme
    workflow = StateGraph(AgentState)
    
    # 2. Instantiate node class containing business logic
    nodes = GraphNodes()
    
    # 3. Define the structural nodes in our graph network
    workflow.add_node("retrieve", nodes.retrieve)
    workflow.add_node("grade_documents", nodes.grade_documents)
    workflow.add_node("generate", nodes.generate)
    workflow.add_node("transform_query", nodes.transform_query)
    
    # 4. Define data execution paths (edges)
    workflow.set_entry_point("retrieve")
    
    # Connect retrieve node output directly to grading verification node
    workflow.add_edge("retrieve", "grade_documents")
    
    # Add a conditional routing junction after grading
    workflow.add_conditional_edges(
        "grade_documents",
        route_after_grading,
        {
            "generate": "generate",
            "transform_query": "transform_query"
        }
    )
    
    # If query transformation happens, loop back and re-query database
    workflow.add_edge("transform_query", "retrieve")
    
    # Connect successful answer generations to exit terminal
    workflow.add_edge("generate", END)
    
    # 5. Compile state machine system
    return workflow.compile()