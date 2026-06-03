import os
import time
from loguru import logger
import ollama  # 🟢 Added Ollama import
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from typing import Dict, Any, List  
from src.agent.state import AgentState  

from ddgs import DDGS

class GraphNodes:
    def __init__(self):
        # 🟢 Swap out genai.Client() for a local model pointer
        self.model_name = "llama3.1"
        
        # Connect to localized containerized infrastructure
        self.qdrant = QdrantClient(host=os.getenv("QDRANT_HOST", "localhost"), port=6333)
        self.embed_model = SentenceTransformer("BAAI/bge-small-en-v1.5")

    def _call_gemini(self, prompt: str, max_retries: int = 2) -> str:
        """
        Kept method name identical to prevent breaking your graph architecture.
        Routes all generation traffic directly to your local Ollama runtime.
        """
        for attempt in range(max_retries):
            try:
                # 🟢 Using Ollama's direct chat implementation
                response = ollama.chat(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}]
                )
                return response['message']['content']
                
            except Exception as e:
                err_msg = str(e)
                logger.warning(f"Ollama connection hiccup: {err_msg}. Retrying... (Attempt {attempt + 1}/{max_retries})")
                if attempt == max_retries - 1:
                    logger.error("Ollama engine failed permanently.")
                    raise e
                time.sleep(2)

    def retrieve(self, state: AgentState) -> Dict[str, Any]:
        """Node: Retrieve matching context chunks from local Qdrant DB container."""
        logger.info("--- NODE: RETRIEVE ---")
        query = state["query"]
        
        query_vector = self.embed_model.encode(query).tolist()
        
        search_results = self.qdrant.query_points(
            collection_name="enterprise_knowledge",
            query=query_vector,
            limit=4
        ).points
        
        chunks = [hit.payload["text"] for hit in search_results if hit.payload and "text" in hit.payload]
        logger.info(f"Retrieved {len(chunks)} relevant content fragments from vector pool.")
        
        current_steps = state.get("steps", [])
        current_steps.append("retrieve")
        
        return {"retrieved_docs": chunks, "steps": current_steps}
    
    def grade_documents(self, state: AgentState) -> Dict[str, Any]:
        """Filters out non-relevant context chunks using structured LLM evaluations."""
        logger.info("--- NODE: GRADE DOCUMENTS ---")
        chunks = state.get("retrieved_docs", [])
        current_steps = state.get("steps", [])

        if not chunks:
            logger.warning("No documents found in state to evaluate.")
            current_steps.append("grade_documents:no")  # 🟢 Append verdict to steps list
            return {"retrieved_docs": [], "steps": current_steps}

        context_str = "\n\n".join(chunks)
        prompt = (
            f"[SYSTEM: You are a strict grading filter. You must respond with exactly one word: 'yes' or 'no'. Do not explain your reasoning.]\n\n"
            f"Query: {state['query']}\n\n"
            f"Context Document Fragments:\n{context_str}\n\n"
            f"Is the context relevant? Answer:"
        )
        
        response_text = self._call_gemini(prompt).strip().lower()
        logger.info(f"Grading evaluation verdict received: '{response_text}'")
        
        # 🟢 Calculate and append the dynamic verification label right into the active state list
        verdict_step = "grade_documents:yes" if "yes" in response_text else "grade_documents:no"
        current_steps.append(verdict_step)
        
        return {"retrieved_docs": chunks, "steps": current_steps}
    
    def transform_query(self, state: AgentState) -> Dict[str, Any]:
        """Optimizes the query string if the retrieved context was deemed irrelevant."""
        logger.info("--- NODE: TRANSFORM QUERY ---")
        current_steps = state.get("steps", [])
        current_steps.append("transform_query")

        # 🟢 Force local model to output ONLY the search terms, nothing else
        prompt = (
            f"[SYSTEM: You are a vector database keyword optimizer. You must respond ONLY with the optimized plain-text query keywords. Do not include introductory remarks, markdown formatting, explanations, or label headers.]\n\n"
            f"Original Query: {state['query']}\n\n"
            f"Optimized plain-text query terms:"
        )
        
        better_query = self._call_gemini(prompt).strip()
        
        # Defensive check to clean up standard markdown wrapper code blocks if model ignores instructions
        better_query = better_query.replace("`", "").replace('"', '').replace("'", "")
        if "\n" in better_query:
            better_query = better_query.split("\n")[-1] # Grabs final row item if it creates lists
            
        logger.info(f"Optimized query engineered: '{better_query}'")
        return {"query": better_query, "steps": current_steps}

    def generate(self, state: AgentState) -> Dict[str, Any]:
        """Synthesizes the final answer using retrieved context."""
        logger.info("--- NODE: GENERATE ---")
        current_steps = state.get("steps", [])
        current_steps.append("generate")

        chunks = state.get("retrieved_docs", [])
        context_str = "\n\n".join(chunks)
        
        prompt = (
            f"You are a helpful academic AI assistant.\n"
            f"Using the verified source documentation context provided below, construct a comprehensive, "
            f"factual, and detailed answer to the query. If the context does not contain the answer, "
            f"state clearly that the information was missing.\n\n"
            f"Query: {state['query']}\n\n"
            f"Context:\n{context_str}\n\n"
            f"Answer:"
        )
        
        answer = self._call_gemini(prompt)
        return {"generation": answer, "steps": current_steps}
    
    def web_search(self, state: AgentState) -> Dict[str, Any]:
        """Node: Queries the live internet when local database documentation is missing."""
        logger.info("--- NODE: LIVE WEB SEARCH FALLBACK ---")
        query = state["query"]
        current_steps = state.get("steps", [])
        current_steps.append("web_search")

        try:
            # 🟢 Clean, updated syntax using the modern 'ddgs' library
            with DDGS() as ddgs:
                logger.info(f"Executing live internet search query for: '{query}'")
                # Direct call to fetch text results natively
                search_results = ddgs.text(query, max_results=3)
                
            if search_results:
                # Format the verified results into string context blocks
                web_chunks = [
                    f"Source Link: {res.get('href', '')}\nContent: {res.get('body', '')}" 
                    for res in search_results if res.get('body')
                ]
                
                if web_chunks:
                    logger.success(f"Live web search successfully fetched {len(web_chunks)} external snippets.")
                    return {"retrieved_docs": web_chunks, "steps": current_steps}
                    
        except Exception as e:
            logger.error(f"Live web search fallback execution failed: {str(e)}")
            
        # Fallback snippet if scraping fails or returns nothing
        return {"retrieved_docs": ["Error: Live internet fallback search failed to extract data points."], "steps": current_steps}