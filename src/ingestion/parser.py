import os
from pathlib import Path
from pypdf import PdfReader
from loguru import logger

class DocumentParser:
    """Handles extraction of raw text from various file formats."""
    
    @staticmethod
    async def parse_pdf(file_path: Path) -> str:
        """Extracts text from a PDF file page by page."""
        logger.info(f"Parsing PDF file: {file_path.name}")
        text_content = []
        try:
            reader = PdfReader(file_path)
            for page_num, page in enumerate(reader.pages):
                page_text = page.extract_text()
                if page_text:
                    text_content.append(page_text)
            return "\n".join(text_content)
        except Exception as e:
            logger.error(f"Failed to parse PDF {file_path.name}: {str(e)}")
            raise e

    @staticmethod
    async def parse_txt(file_path: Path) -> str:
        """Extracts text from plain text or markdown files."""
        logger.info(f"Parsing text/md file: {file_path.name}")
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            logger.error(f"Failed to read text file {file_path.name}: {str(e)}")
            raise e

    async def extract_text(self, file_path: str) -> str:
        """Gateway method to automatically detect extension and parse."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found at: {file_path}")
            
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            return await self.parse_pdf(path)
        elif suffix in [".txt", ".md"]:
            return await self.parse_txt(path)
        else:
            logger.warning(f"Unsupported file extension '{suffix}' for {path.name}. Skipping.")
            return ""