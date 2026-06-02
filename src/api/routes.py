import os
from fastapi import APIRouter, HTTPException
from src.api.schemas import QueryRequest, QueryResponse
from src.agent.graph import build_agent_graph
from loguru import logger

# 1. Explicitly initialize and name the APIRouter instance 'router'
router = APIRouter(prefix="/api/v1", tags=["Agentic RAG Engine"])

# 2. Compile our LangGraph runner once on API startup to save memory overhead
agent_pipeline = build_agent_graph()

@router.post("/query", response_model=QueryResponse)
async def execute_agent_query(payload: QueryRequest):
    """
    Inference Gateway Endpoint: Accepts user queries, processes them 
    through the self-correcting agent graph, and returns structured results.
    """
    logger.info(f"API Inbound Request Received: '{payload.query}'")
    
    # Establish fresh baseline state dictionary for LangGraph
    initial_state = {
        "query": payload.query,
        "retrieved_docs": [],
        "generation": "",
        "steps": []
    }
    
    try:
        # Execute the self-correcting state graph asynchronously
        final_state = await agent_pipeline.ainvoke(initial_state)
        
        # Return validated JSON response matching Pydantic expectations
        return QueryResponse(
            query=final_state.get("query", payload.query),
            generation=final_state.get("generation", "Agent failed to produce a valid response."),
            steps_traced=final_state.get("steps", []),
            success=True
        )
        
    except Exception as e:
        logger.error(f"System failure during agent execution loop: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal agent system error: {str(e)}")