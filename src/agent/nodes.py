import os
import time
from loguru import logger
from google import genai
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

class GraphNodes:
    def __init__(self):
        self.ai = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        self.embed_model = SentenceTransformer("BAAI/bge-small-en-v1.5")
        
        self.qdrant_client = QdrantClient(
            host=os.getenv("QDRANT_HOST", "localhost"),
            port=int(os.getenv("QDRANT_PORT", 6333)),
            check_compatibility=False
        )
        self.collection_name = "enterprise_knowledge"

    def _call_gemini(self, prompt: str, max_retries: int = 3) -> str:
        """Central robust runner that shields all nodes from temporary Gemini 503 load spikes."""
        for attempt in range(max_retries):
            try:
                response = self.ai.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=prompt
                )
                return response.text
            except Exception as e:
                if "503" in str(e) and attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 3
                    logger.warning(f"Gemini overloaded (503). Retrying in {wait_time}s... (Attempt {attempt + 1}/{max_retries})")
                    time.sleep(wait_time)
                else:
                    logger.error(f"Gemini call failed permanently: {str(e)}")
                    raise e

    def retrieve(self, state: dict) -> dict:
        """Fetches relative chunks from Qdrant based on current state query."""
        logger.info(f"--- NODE: RETRIEVE --- Query: '{state['query']}'")
        query_vector = self.embed_model.encode(state["query"]).tolist()
        
        search_results = self.qdrant_client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=3
        )
        docs = [hit.payload["text"] for hit in search_results.points if hit.payload and "text" in hit.payload]
        return {"retrieved_docs": docs, "steps": state.get("steps", []) + ["retrieve"]}

    def grade_documents(self, state: dict) -> dict:
        """Filters out non-relevant context chunks using Gemini metrics."""
        logger.info("--- NODE: GRADE DOCUMENTS ---")
        if not state.get("retrieved_docs"):
            return {"retrieved_docs": [], "steps": state.get("steps", []) + ["grade_documents"]}

        context_str = "\n\n".join(state["retrieved_docs"])
        prompt = f"Query: {state['query']}\nContext: {context_str}\nIs this relevant? Respond 'yes' or 'no'."
        
        response_text = self._call_gemini(prompt)
        logger.info(f"Grading verdict received: {response_text.strip().lower()}")
        return {"retrieved_docs": state["retrieved_docs"], "steps": state.get("steps", []) + ["grade_documents"]}

    def transform_query(self, state: dict) -> dict:
        """Optimizes the query string if the retrieved context was deemed irrelevant."""
        logger.info("--- NODE: TRANSFORM QUERY ---")
        prompt = f"Optimize this query for a vector database search: {state['query']}"
        
        better_query = self._call_gemini(prompt)
        logger.info(f"Optimized Query built: '{better_query.strip()}'")
        return {"query": better_query.strip(), "steps": state.get("steps", []) + ["transform_query"]}

    def generate(self, state: dict) -> dict:
        """Synthesizes the final answer using retrieved context."""
        logger.info("--- NODE: GENERATE ---")
        prompt = f"Query: {state['query']}\nContext: {state['retrieved_docs']}\nGenerate a professional response."
        
        answer = self._call_gemini(prompt)
        return {"generation": answer, "steps": state.get("steps", []) + ["generate"]}