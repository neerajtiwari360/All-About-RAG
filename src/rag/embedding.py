from typing import List, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import numpy as np
from .data_loader import load_all_documents
from src.config import get_chunking_config, get_embedding_config

"""
Embedding Pipeline Module

This module handles text chunking and embedding generation for documents.
It splits large documents into manageable chunks and converts them into
vector embeddings using SentenceTransformer models for efficient retrieval.
"""

class EmbeddingPipeline:
    def __init__(self, model_name: str = None, chunk_size: int = None, chunk_overlap: int = None):
        # Load configuration
        chunking_config = get_chunking_config()
        embedding_config = get_embedding_config()
        
        # Use config values or provided parameters
        self.chunk_size = chunk_size or chunking_config.chunk_size
        self.chunk_overlap = chunk_overlap or chunking_config.chunk_overlap
        self.separators = chunking_config.separators
        
        model_name = model_name or embedding_config.model_name
        self.model = SentenceTransformer(model_name, device=embedding_config.device)
        self.batch_size = embedding_config.batch_size
        self.show_progress_bar = embedding_config.show_progress_bar
        
        print(f"[INFO] Loaded embedding model: {model_name}")
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

    def embed_chunks(self, chunks: List[Any]) -> np.ndarray:
        texts = [chunk.page_content for chunk in chunks]
        print(f"[INFO] Generating embeddings for {len(texts)} chunks...")
        embeddings = self.model.encode(
            texts, 
            show_progress_bar=self.show_progress_bar,
            batch_size=self.batch_size
        )
        print(f"[INFO] Embeddings shape: {embeddings.shape}")
        return embeddings

# Example usage
if __name__ == "__main__":
    
    docs = load_all_documents("data")
    emb_pipe = EmbeddingPipeline()
    chunks = emb_pipe.chunk_documents(docs)
    embeddings = emb_pipe.embed_chunks(chunks)
    print("[INFO] Example embedding:", embeddings[0] if len(embeddings) > 0 else None)
