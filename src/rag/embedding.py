from typing import List, Any
from sentence_transformers import SentenceTransformer
import numpy as np
from .data_loader import load_all_documents
from src.config import get_embedding_config

"""
Embedding Pipeline Module

This module handles embedding generation for text chunks.
It converts text chunks into vector embeddings using SentenceTransformer models for efficient retrieval.
"""

class EmbeddingPipeline:
    def __init__(self, model_name: str = None):
        # Load configuration
        embedding_config = get_embedding_config()
        
        model_name = model_name or embedding_config.model_name
        self.model = SentenceTransformer(model_name, device=embedding_config.device)
        self.batch_size = embedding_config.batch_size
        self.show_progress_bar = embedding_config.show_progress_bar
        
        print(f"[INFO] Loaded embedding model: {model_name}")

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
    # Note: For full pipeline, also use ChunkingPipeline
    print("[INFO] Embedding pipeline ready. Use ChunkingPipeline for chunking first.")
