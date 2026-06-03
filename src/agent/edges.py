from loguru import logger
from src.agent.state import AgentState

def route_after_grading(state: AgentState) -> str:
    """Evaluates the grading verdict strings to branch execution paths dynamically."""
    current_steps = state.get("steps", [])
    chunks = state.get("retrieved_docs", [])
    
    # Safety Check: If retrieval completely failed to find text, route to live web search
    if not chunks:
        logger.warning("--- EDGE ASSESSMENT: NO DOCUMENTS FOUND -> ROUTING TO WEB SEARCH ---")
        return "web_search"
        
    # Standard Adaptive Route: Check the last validation step metric token
    if current_steps and current_steps[-1] == "grade_documents:yes":
        logger.success("--- EDGE ASSESSMENT: LOCAL KNOWLEDGE VERIFIED -> ROUTING TO GENERATE ---")
        return "generate"
        
    # 🟢 CHANGED: Instead of looping to 'transform_query', route directly to 'web_search'
    logger.warning("--- EDGE ASSESSMENT: INSUFFICIENT LOCAL CONTEXT -> ROUTING TO LIVE WEB SEARCH ---")
    return "web_search"