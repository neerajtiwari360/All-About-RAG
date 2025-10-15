from src.rag.data_loader import load_all_documents
from src.rag.vectorstore import FaissVectorStore
from src.rag.search import RAGSearch

# Example usage
if __name__ == "__main__":
    
    docs = load_all_documents("data")
    store = FaissVectorStore("faiss_store")
    try:
        store.load()
    except FileNotFoundError:
        print("[WARN] No existing Faiss index found. Building a new vector store...")
        store.build_from_documents(docs)
    #print(store.query("What is attention mechanism?", top_k=3))
    rag_search = RAGSearch()
    query = "What is the doc about?"
    summary = rag_search.search_and_summarize(query, top_k=3)
    print("Summary:", summary)
