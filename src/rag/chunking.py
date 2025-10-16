from typing import List, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter
from src.config import get_chunking_config

"""
Chunking Pipeline Module

This module handles text chunking for documents.
It splits large documents into manageable chunks using LangChain's RecursiveCharacterTextSplitter.
"""

class ChunkingPipeline:
    def __init__(self, chunk_size: int = None, chunk_overlap: int = None, separators: list = None):
        # Load configuration
        chunking_config = get_chunking_config()
        
        # Use config values or provided parameters
        self.chunk_size = chunk_size or chunking_config.chunk_size
        self.chunk_overlap = chunk_overlap or chunking_config.chunk_overlap
        self.separators = separators or chunking_config.separators
        
        print(f"[INFO] Chunking config - Size: {self.chunk_size}, Overlap: {self.chunk_overlap}")

    def chunk_documents(self, documents: List[Any]) -> List[Any]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=self.separators
        )
        chunks = splitter.split_documents(documents)
        print(f"[INFO] Split {len(documents)} documents into {len(chunks)} chunks.")
        return chunks

# Example usage
if __name__ == "__main__":
    from .data_loader import load_all_documents
    
    docs = load_all_documents("data")
    chunk_pipe = ChunkingPipeline()
    chunks = chunk_pipe.chunk_documents(docs)
    print(f"[INFO] Chunked into {len(chunks)} chunks.")