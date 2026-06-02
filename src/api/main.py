import os
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# FIX: Force load_dotenv to look at the exact root directory path relative to this file
env_path = Path(__file__).parent.parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

# Now that environment variables are strictly loaded, we import the routes
from src.api.routes import router as api_router

# Instantiate the global app instance
app = FastAPI(
    title="Enterprise Agentic RAG Pipeline",
    description="FAANG-grade production self-correcting vector knowledge database.",
    version="1.0.0"
)

# Configure Cross-Origin Resource Sharing (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount the self-correcting agent graph API endpoints
app.include_router(api_router)

@app.get("/health", tags=["Infrastructure Checking"])
async def health_check():
    """Basic uptime monitoring verification gateway."""
    return {"status": "healthy", "service": "enterprise-rag-api"}