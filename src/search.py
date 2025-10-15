import os
from dotenv import load_dotenv
from src.vectorstore import FaissVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI
from src.config import get_llm_config, get_vectorstore_config, get_search_config, get_embedding_config, config

load_dotenv()

class RAGSearch:
    def __init__(
        self,
        persist_dir: str = None,
        embedding_model: str = None,
        llm_model: str = None
    ):
        # Load configuration
        llm_config = get_llm_config()
        vectorstore_config = get_vectorstore_config()
        embedding_config = get_embedding_config()
        
        # Use config values or provided parameters
        persist_dir = persist_dir or vectorstore_config.persist_directory
        embedding_model = embedding_model or embedding_config.model_name
        llm_model = llm_model or llm_config.model_name
        
        self.vectorstore = FaissVectorStore(persist_dir, embedding_model)
        
        # Load or build vectorstore
        faiss_path = os.path.join(persist_dir, "faiss.index")
        meta_path = os.path.join(persist_dir, "metadata.pkl")
        if not (os.path.exists(faiss_path) and os.path.exists(meta_path)):
            from src.data_loader import load_all_documents
            doc_config = config.get_document_config()
            docs = load_all_documents(doc_config.data_directory)
            if docs:
                self.vectorstore.build_from_documents(docs)
        else:
            self.vectorstore.load()
        
        # Initialize LLM with configuration
        gemini_api_key = os.getenv(llm_config.api_key_env) or os.getenv(llm_config.fallback_api_key_env)
        if not gemini_api_key:
            raise EnvironmentError(f"API key not found. Set {llm_config.api_key_env} or {llm_config.fallback_api_key_env} in your environment.")
        
        # Initialize LLM with config parameters
        self.llm = ChatGoogleGenerativeAI(
            model=llm_model, 
            google_api_key=gemini_api_key,
            temperature=llm_config.temperature,
            max_tokens=llm_config.max_tokens
        )
        print(f"[INFO] Gemini LLM initialized: {llm_model}")
        print(f"[INFO] LLM config - Temperature: {llm_config.temperature}, Max tokens: {llm_config.max_tokens}")

    def search_and_summarize(self, query: str, top_k: int = None) -> str:
        search_config = get_search_config()
        if top_k is None:
            top_k = search_config.default_top_k
        
        # Validate top_k
        top_k = max(search_config.min_top_k, min(top_k, search_config.max_top_k))
        
        results = self.vectorstore.query(query, top_k=top_k)
        texts = [r["metadata"].get("text", "") for r in results if r["metadata"]]
        context = "\n\n".join(texts)
        
        if not context:
            return "No relevant documents found."
        
        # Use configured prompt template
        prompt_template = config.get_prompt_template()
        prompt = prompt_template.format(context=context, query=query)
        
        response = self.llm.invoke(prompt)
        return getattr(response, "content", str(response))

# Example usage
if __name__ == "__main__":
    rag_search = RAGSearch()
    query = "What is the doc about?"
    summary = rag_search.search_and_summarize(query, top_k=3)
    print("Summary:", summary)
