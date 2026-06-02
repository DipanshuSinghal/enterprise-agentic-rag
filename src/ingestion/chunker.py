from typing import List, Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter
from loguru import logger

class DocumentChunker:
    """Splits raw text into optimized, overlapping chunks for semantic vector mapping."""
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False,
            separators=["\n\n", "\n", " ", ""]
        )

    async def split_document(self, text: str, meta_source: str) -> List[Dict[Any, Any]]:
        """
        Splits text and returns a list of dictionaries containing 
        the chunk content and structured metadata.
        """
        if not text.strip():
            return []
            
        logger.info(f"Chunking document source: {meta_source}")
        chunks = self.splitter.split_text(text)
        
        structured_chunks = []
        for index, chunk in enumerate(chunks):
            structured_chunks.append({
                "text": chunk,
                "metadata": {
                    "source": meta_source,
                    "chunk_id": index,
                    "length": len(chunk)
                }
            })
            
        logger.info(f"Generated {len(structured_chunks)} chunks for {meta_source}")
        return structured_chunks