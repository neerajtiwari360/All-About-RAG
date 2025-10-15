import os
import faiss
import numpy as np
import pickle
from typing import List, Any
from sentence_transformers import SentenceTransformer
from .embedding import EmbeddingPipeline
from src.config import get_vectorstore_config, get_embedding_config, get_search_config

class FaissVectorStore:
    def __init__(self, persist_dir: str = None, embedding_model: str = None, chunk_size: int = None, chunk_overlap: int = None):
        # Load configuration
        vectorstore_config = get_vectorstore_config()
        embedding_config = get_embedding_config()
        
        self.persist_dir = persist_dir or vectorstore_config.persist_directory
        os.makedirs(self.persist_dir, exist_ok=True)
        
        self.index = None
        self.metadata = []
        self.embedding_model = embedding_model or embedding_config.model_name
        self.model = SentenceTransformer(self.embedding_model, device=embedding_config.device)
        self.chunk_size = chunk_size or embedding_config.batch_size
        self.chunk_overlap = chunk_overlap
        
        print(f"[INFO] Loaded embedding model: {self.embedding_model}")
        print(f"[INFO] Vector store directory: {self.persist_dir}")

    def build_from_documents(self, documents: List[Any]):
        print(f"[INFO] Building vector store from {len(documents)} raw documents...")
        emb_pipe = EmbeddingPipeline(
            model_name=self.embedding_model, 
            chunk_size=self.chunk_size, 
            chunk_overlap=self.chunk_overlap
        )
        chunks = emb_pipe.chunk_documents(documents)
        embeddings = emb_pipe.embed_chunks(chunks)
        metadatas = [{"text": chunk.page_content} for chunk in chunks]
        self.add_embeddings(np.array(embeddings).astype('float32'), metadatas)
        self.save()
        print(f"[INFO] Vector store built and saved to {self.persist_dir}")

    def add_embeddings(self, embeddings: np.ndarray, metadatas: List[Any] = None):
        dim = embeddings.shape[1]
        if self.index is None:
            self.index = faiss.IndexFlatL2(dim)
        self.index.add(embeddings)
        if metadatas:
            self.metadata.extend(metadatas)
        print(f"[INFO] Added {embeddings.shape[0]} vectors to Faiss index.")

    def save(self):
        faiss_path = os.path.join(self.persist_dir, "faiss.index")
        meta_path = os.path.join(self.persist_dir, "metadata.pkl")
        if self.index is None:
            raise ValueError("Vector store has no index to save. Add embeddings before calling save().")
        faiss.write_index(self.index, faiss_path)
        with open(meta_path, "wb") as f:
            pickle.dump(self.metadata, f)
        print(f"[INFO] Saved Faiss index and metadata to {self.persist_dir}")

    def load(self):
        faiss_path = os.path.join(self.persist_dir, "faiss.index")
        meta_path = os.path.join(self.persist_dir, "metadata.pkl")
        missing_files = [path for path in (faiss_path, meta_path) if not os.path.exists(path)]
        if missing_files:
            missing_str = ", ".join(missing_files)
            raise FileNotFoundError(
                f"Persisted vector store files not found: {missing_str}. Build the store before loading."
            )
        self.index = faiss.read_index(faiss_path)
        with open(meta_path, "rb") as f:
            self.metadata = pickle.load(f)
        print(f"[INFO] Loaded Faiss index and metadata from {self.persist_dir}")

    def search(self, query_embedding: np.ndarray, top_k: int = None):
        # Use configuration for default top_k
        search_config = get_search_config()
        if top_k is None:
            top_k = search_config.default_top_k
        
        # Validate top_k range
        top_k = max(search_config.min_top_k, min(top_k, search_config.max_top_k))
        
        D, I = self.index.search(query_embedding, top_k)
        results = []
        for idx, dist in zip(I[0], D[0]):
            meta = self.metadata[idx] if idx < len(self.metadata) else None
            result = {"index": idx, "metadata": meta}
            
            # Include distance if configured
            if search_config.include_distances:
                result["distance"] = float(dist)
            
            results.append(result)
        return results

    def query(self, query_text: str, top_k: int = None):
        search_config = get_search_config()
        if top_k is None:
            top_k = search_config.default_top_k
            
        print(f"[INFO] Querying vector store for: '{query_text}' (top_k={top_k})")
        query_emb = self.model.encode([query_text]).astype('float32')
        return self.search(query_emb, top_k=top_k)

# Example usage
if __name__ == "__main__":
    from data_loader import load_all_documents
    docs = load_all_documents("data")
    store = FaissVectorStore("faiss_store")
    store.build_from_documents(docs)
    store.load()
    print(store.query("What is the doc about??", top_k=3))
