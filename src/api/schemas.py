from pydantic import BaseModel, Field
from typing import List, Optional

class QueryRequest(BaseModel):
    """
    Inbound Request Validation Schema.
    Validates payload elements arriving from the user frontend client.
    """
    query: str = Field(
        ...,
        description="The natural language question to ask the self-correcting agent database.",
        examples=["What is the content discussed in the documentation?"]
    )
    
    # 🟢 NEW: Thread Identifier tracking variable for persistent session memory states
    thread_id: Optional[str] = Field(
        default="default_session",
        description="Unique UUID string tracking independent conversational multi-turn history states."
    )

class QueryResponse(BaseModel):
    """
    Outbound Response Validation Schema.
    Structures the JSON response returned to the user client interface.
    """
    query: str = Field(..., description="The query string used for final generation.")
    generation: str = Field(..., description="The grounded text answer synthesized by Gemini.")
    steps_traced: List[str] = Field(..., description="The collection of nodes executed inside LangGraph.")
    success: bool = Field(default=True, description="Uptime execution confirmation flag.")