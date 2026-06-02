import asyncio
import os
from dotenv import load_dotenv
from loguru import logger

# Load environment variables from .env file
load_dotenv()

from src.ingestion.parser import DocumentParser
from src.ingestion.chunker import DocumentChunker
from src.ingestion.embedder import DataEmbedder

async def main():
    logger.info("Starting Ingestion Pipeline Pipeline Test...")
    
    # 1. Define target folder and find files
    docs_dir = "./data/raw_documents"
    files = [f for f in os.listdir(docs_dir) if f.endswith(('.txt', '.md', '.pdf'))]
    
    if not files:
        logger.warning(f"No files found in '{docs_dir}'. Please drop a .txt, .md, or .pdf file there first!")
        return

    target_file = os.path.join(docs_dir, files[0])
    logger.info(f"Found sample file to process: {target_file}")

    # 2. Initialize our modules
    parser = DocumentParser()
    chunker = DocumentChunker(chunk_size=500, chunk_overlap=50)
    embedder = DataEmbedder()

    # 3. Run the ETL process
    # Extract
    raw_text = await parser.extract_text(target_file)
    
    # Transform
    structured_chunks = await chunker.split_document(raw_text, meta_source=files[0])
    
    # Load
    await embedder.embed_and_upsert(structured_chunks)
    
    logger.success("Ingestion pipeline test executed flawlessly!")

if __name__ == "__main__":
    asyncio.run(main())