# AI Coding Agent Instructions for All-About-RAG

## Project Overview
This is a Retrieval-Augmented Generation (RAG) system that processes documents (PDF, TXT, CSV, Excel, Word, JSON) into searchable knowledge base. Users upload documents via web UI or API, ask questions, and get AI-generated answers grounded in document content.

## Architecture
- **Data Flow**: Documents → LangChain loaders → RecursiveCharacterTextSplitter (1000 chars, 200 overlap) → SentenceTransformer embeddings → FAISS vector store → Cosine similarity search → Google Gemini LLM summarization
- **Service Boundaries**: `src/rag/` contains core RAG logic; `api.py` provides FastAPI endpoints; `static/` serves web UI
- **Why FAISS**: Chosen for fast L2 similarity search on CPU, persistent storage with metadata

## Key Components
- `src/rag/data_loader.py`: Multi-format document loading using LangChain loaders
- `src/rag/chunking.py`: Text chunking using RecursiveCharacterTextSplitter
- `src/rag/embedding.py`: Text chunk embedding using SentenceTransformer models
- `src/rag/vectorstore.py`: FAISS index management with pickle metadata storage
- `src/rag/search.py`: Query embedding + vector search + LLM summarization
- `api.py`: FastAPI with CORS, file upload, search endpoints
- `config.yaml`: Centralized YAML config with dataclasses in `src/config.py`

## Development Workflow
1. **Setup**: If `.venv` doesn't exist: `python -m venv .venv`. Then: `.venv\Scripts\activate` (Windows) or `source .venv/bin/activate` (Linux/Mac); `pip install -r requirements.txt`
2. **Environment**: Create `.env` with `GEMINI_API_KEY` or `GOOGLE_API_KEY`
3. **Run API**: `uvicorn api:app --reload --host 0.0.0.0 --port 8000`
4. **Access**: Web UI at `http://localhost:8000`, API docs at `/docs`
5. **Add Documents**: Place files in `data/` folder or upload via API/web UI

## Configuration Patterns
- Use `src.config` functions like `get_embedding_config()` instead of direct YAML access
- Config values overrideable via constructor params (e.g., `EmbeddingPipeline(model_name="custom")`)
- Device auto-detection for embeddings (CUDA > MPS > CPU)

## API Patterns
- Pydantic models for requests/responses (e.g., `SearchRequest` with `query` and `top_k`)
- Global RAG components initialized lazily in `initialize_rag()`
- Error handling with HTTPException and JSONResponse
- CORS enabled for web UI integration

## Code Conventions
- Logging via `print(f"[INFO] ...")` statements throughout
- Type hints with `List[Any]`, `Optional[int]`
- Relative imports within `src.rag` package
- Example usage in `if __name__ == "__main__":` blocks
- Config validation in dataclass `__post_init__` methods

## Examples
**Add new document loader**:
```python
# In data_loader.py
if ".md" in doc_config.supported_formats:
    md_files = list(data_path.glob('**/*.md'))
    for md_file in md_files:
        loader = TextLoader(str(md_file))
        documents.extend(loader.load())
```

**Custom embedding model**:
```python
# Override in config.yaml or constructor
emb_pipe = EmbeddingPipeline(model_name="all-mpnet-base-v2")
```

**New API endpoint**:
```python
# In api.py
@app.post("/custom-search")
async def custom_search(request: SearchRequest):
    return await perform_search(request.query, request.top_k)
```</content>
<parameter name="filePath">d:\GITHUB\All-About-RAG\.github\copilot-instructions.md