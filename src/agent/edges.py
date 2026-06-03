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

def route_input_intent(state: AgentState) -> str:
    """Evaluates whether the user query needs a database lookup or is just casual chat."""
    query = state["query"].lower()
    
    # Casual/Conversational/Game keywords that don't need a vector database lookup
    chat_keywords = ["hello", "hi", "choose a number", "pick a number", "multiply", "add", "subtract", "game"]
    
    # Check if it's a follow-up or basic math instruction
    if any(keyword in query for keyword in chat_keywords) or len(query.split()) < 4:
        return "skip_to_generate"
        
    return "execute_rag"