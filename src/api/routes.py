import os
import json
from fastapi import APIRouter, HTTPException, Request
from src.api.schemas import QueryRequest
from src.agent.graph import build_agent_graph
from loguru import logger
from ollama import AsyncClient
from fastapi.responses import StreamingResponse

# 1. Explicitly initialize and name the APIRouter instance 'router'
router = APIRouter(prefix="/api/v1", tags=["Agentic RAG Engine"])

# 2. Compile our LangGraph runner once on API startup to save memory overhead
agent_pipeline = build_agent_graph()

@router.post("/query")
async def execute_agent_query(payload: QueryRequest, request: Request):
    """
    Inference Gateway Endpoint: Accepts user queries along with a unique thread_id,
    orchestrates retrieval/grading via LangGraph, and streams text responses live.
    """
    # Safely extract variables from the validated Pydantic payload schema
    user_query = payload.query
    
    # 🟢 DYNAMIC MEMORY TRACKING: Extract thread_id from payload, fallback safely if missing
    # This ensures your memory loops (e.g., math games, follow-up questions) work cleanly.
    payload_dict = payload.model_dump() if hasattr(payload, "model_dump") else payload.dict()
    session_thread_id = payload_dict.get("thread_id", "default_fallback_session")
    
    logger.info(f"API Inbound Request [Thread: {session_thread_id}]: '{user_query}'")
    
    # Establish fresh baseline state dictionary for LangGraph execution boundary
    initial_state = {
        "query": user_query,
        "retrieved_docs": [],
        "generation": "",
        "steps": []
    }
    
    async def event_stream():
        try:
            # Configure multi-turn persistent thread context parameters using our unique session id
            config = {"configurable": {"thread_id": session_thread_id}}
            
            # 1. Execute the full graph state loop boundaries synchronously to find context
            final_state = await agent_pipeline.ainvoke(initial_state, config=config)
            
            # Stream executed graph steps to frontend typewriter to update processing status bars
            for step in final_state.get("steps", []):
                yield f"data: {json.dumps({'node': step})}\n\n"
            
            # 2. Extract documentation blocks (handles both local vector chunks and live web snippets)
            captured_docs = final_state.get("retrieved_docs", [])
            context_str = "\n\n".join(captured_docs) if captured_docs else "No specific context available."
            
            # Structure our local LLM prompt wrapper safely
            prompt = (
                f"[SYSTEM: You are an expert academic assistant. Use the following context blocks to answer the question accurately. "
                f"Maintain deep context awareness across your historical chat interactions.]\n\n"
                f"CONTEXT:\n{context_str}\n\n"
                f"USER QUERY: {user_query}\n\n"
                f"Answer:"
            )
            
            logger.info(f"Initializing AsyncClient token generator stream for thread: {session_thread_id}")
            
            # 🟢 DYNAMIC LOCAL MODEL TARGET
            # Points straight to your local model tags without throwing 404 string errors
            target_model = "llama3.1:latest"  
            
            # 3. Stream raw string tokens directly down the HTTP network pipeline
            async for chunk in await AsyncClient().chat(
                model=target_model, 
                messages=[{'role': 'user', 'content': prompt}], 
                stream=True
            ):
                token = chunk['message']['content']
                yield f"data: {json.dumps({'text': token})}\n\n"
            
            # 4. Stream out our citations block package at the very end
            yield f"data: {json.dumps({'sources': captured_docs})}\n\n"
                        
        except Exception as e:
            logger.error(f"Streaming gateway asynchronous failure: {str(e)}")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")