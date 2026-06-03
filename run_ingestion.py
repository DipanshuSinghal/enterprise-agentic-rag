import os
import glob
from pypdf import PdfReader
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
from loguru import logger

# --- 1. CONFIGURATION ---
DATA_DIR = "./data"
COLLECTION_NAME = "enterprise_knowledge"
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))

def extract_text_from_pdfs(data_dir: str):
    """Scans the directory for PDFs and extracts text page by page."""
    documents = []
    pdf_files = glob.glob(os.path.join(data_dir, "*.pdf"))
    
    if not pdf_files:
        logger.warning(f"No PDF files found in '{data_dir}' directory. Please drop some PDFs there!")
        return documents

    for pdf_path in pdf_files:
        file_name = os.path.basename(pdf_path)
        logger.info(f"Parsing document: {file_name}")
        try:
            reader = PdfReader(pdf_path)
            for page_num, page in enumerate(reader.pages):
                text = page.extract_text()
                if text and text.strip():
                    # We store the text along with source metadata (file name and page number)
                    documents.append({
                        "text": text.strip(),
                        "metadata": {
                            "source": file_name,
                            "page": page_num + 1
                        }
                    })
            logger.success(f"Successfully extracted {len(reader.pages)} pages from {file_name}")
        except Exception as e:
            logger.error(f"Failed to read PDF {file_name}: {e}")
            
    return documents

def chunk_text_data(documents: list, chunk_size: int = 500, chunk_overlap: int = 50):
    """Splits text pages into overlapping chunks to keep contextual boundaries intact."""
    chunks = []
    for doc in documents:
        text = doc["text"]
        words = text.split()
        
        for i in range(0, len(words), chunk_size - chunk_overlap):
            chunk_words = words[i:i + chunk_size]
            chunk_text = " ".join(chunk_words)
            
            # Keep the source information linked to each individual chunk
            chunks.append({
                "text": chunk_text,
                "metadata": {
                    "source": doc["metadata"]["source"],
                    "page": doc["metadata"]["page"]
                }
            })
    return chunks

def main():
    logger.info("Starting Enterprise Multi-Document PDF Ingestion Engine...")
    
    # 1. Read and extract text from all local PDFs
    raw_docs = extract_text_from_pdfs(DATA_DIR)
    if not raw_docs:
        logger.error("No text could be extracted. Ingestion aborted.")
        return

    # 2. Chunk text pages
    logger.info("Tokenizing and creating semantic overlapping chunks...")
    chunks = chunk_text_data(raw_docs)
    logger.info(f"Generated total of {len(chunks)} structural context chunks.")

    # 3. Initialize Local Embedding Transformer Model
    logger.info("Loading local BAAI/bge-small-en-v1.5 sentence transformer...")
    embed_model = SentenceTransformer("BAAI/bge-small-en-v1.5")

    # 4. Initialize Connection to Qdrant Docker Container
    logger.info(f"Connecting to Qdrant vector database at {QDRANT_HOST}:{QDRANT_PORT}...")
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

    # 5. Create or Reset Collection
    # 5. Create Collection Safely (Appends instead of wiping out old files)
    if not client.collection_exists(COLLECTION_NAME):
        logger.info(f"Creating a fresh vector collection: '{COLLECTION_NAME}'...")
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE),
        )
    else:
        logger.info(f"Collection '{COLLECTION_NAME}' already exists. Safely appending new data points...")

    # 6. Generate Embeddings and Upsert to Database
    logger.info("Computing dense vectors and indexing into Qdrant...")
    points = []
    for idx, chunk in enumerate(chunks):
        # Generate 384-dimensional floating point vector
        vector = embed_model.encode(chunk["text"]).tolist()
        
        # Structure payload to store both text and file mapping fields
        payload = {
            "text": chunk["text"],
            "source_file": chunk["metadata"]["source"],
            "page_number": chunk["metadata"]["page"]
        }
        
        points.append(PointStruct(id=idx, vector=vector, payload=payload))

    # Push points in batches to handle larger documents efficiently
    batch_size = 100
    for i in range(0, len(points), batch_size):
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=points[i:i + batch_size]
        )
        
    logger.success(f"Ingestion complete! Successfully indexed {len(points)} vector nodes into Qdrant.")

if __name__ == "__main__":
    main()