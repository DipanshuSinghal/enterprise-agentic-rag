import os
import yaml
from typing import List, Dict, Any
from qdrant_client import QdrantClient
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer
from loguru import logger

class DataEmbedder:
    """Generates vector embeddings locally and manages Qdrant collections."""
    
    def __init__(self):
        # Load configuration details
        config_path = os.path.join(os.path.dirname(__file__), "../../config/settings.yaml")
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
            
        self.collection_name = self.config["database"]["collection_name"]
        self.vector_dim = self.config["embeddings"]["dimension"]
        
        # Initialize local SentenceTransformer model
        logger.info(f"Loading local embedding model: {self.config['embeddings']['model_name']}")
        self.model = SentenceTransformer(self.config["embeddings"]["model_name"])
        
        # Initialize Qdrant Client connecting to Docker container
        self.client = QdrantClient(
            host=os.getenv("QDRANT_HOST", "localhost"), 
            port=int(os.getenv("QDRANT_PORT", 6333))
        )
        
        self._ensure_collection_exists()

    def _ensure_collection_exists(self):
        """Creates the vector collection in Qdrant if it doesn't already exist."""
        collections = self.client.get_collections().collections
        exists = any(c.name == self.collection_name for c in collections)
        
        if not exists:
            logger.info(f"Creating new Qdrant collection: {self.collection_name}")
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=self.vector_dim, 
                    distance=models.Distance.COSINE
                )
            )
        else:
            logger.debug(f"Qdrant collection '{self.collection_name}' already exists.")

    async def embed_and_upsert(self, structured_chunks: List[Dict[str, Any]]):
        """Converts chunks to embeddings and stores them inside Qdrant."""
        if not structured_chunks:
            logger.warning("No chunks provided for ingestion.")
            return

        logger.info(f"Generating vectors for {len(structured_chunks)} chunks...")
        
        # Extract pure strings for batch embedding generation
        texts = [chunk["text"] for chunk in structured_chunks]
        embeddings = self.model.encode(texts, show_progress_bar=False).tolist()
        
        points = []
        for idx, chunk in enumerate(structured_chunks):
            # FIX: Create a deterministic clean integer ID for Qdrant using hash combination
            unique_string_id = f"{chunk['metadata']['source']}-{chunk['metadata']['chunk_id']}"
            # Abs ensures it is always an unsigned positive integer
            numeric_id = abs(hash(unique_string_id)) % (10 ** 10)
            
            points.append(
                models.PointStruct(
                    id=numeric_id, # Pure positive integer ID mapping
                    vector=embeddings[idx],
                    payload={
                        "text": chunk["text"],
                        "source": chunk["metadata"]["source"],
                        "chunk_id": chunk["metadata"]["chunk_id"]
                    }
                )
            )
            
        # Push vectors to database via fast batch insertion
        logger.info(f"Uploading vectors to Qdrant cluster...")
        self.client.upsert(
            collection_name=self.collection_name,
            wait=True,
            points=points
        )
        logger.success(f"Successfully indexed {len(points)} vectors in Qdrant!")