import os
import shutil
from pathlib import Path
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn
from dotenv import load_dotenv

from src.rag.data_loader import load_all_documents
from src.rag.vectorstore import FaissVectorStore
from src.rag.search import RAGSearch
from src.config import config, get_server_config, get_document_config, get_search_config, get_llm_config

# Load environment variables
load_dotenv()

# Load configuration
server_config = get_server_config()
doc_config = get_document_config()
search_config = get_search_config()
llm_config = get_llm_config()

# Initialize FastAPI app
app = FastAPI(
    title=server_config.title,
    description=server_config.description,
    version=server_config.version,
    docs_url=server_config.docs_url,
    redoc_url=server_config.redoc_url
)

# Add CORS middleware
cors_config = config.get_cors_config()
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_config["allow_origins"],
    allow_credentials=cors_config["allow_credentials"],
    allow_methods=cors_config["allow_methods"],
    allow_headers=cors_config["allow_headers"],
)

# Mount static files
static_dir = Path("static")
if static_dir.exists():
    app.mount("/static", StaticFiles(directory="static"), name="static")

# Global variables for RAG components
rag_search: Optional[RAGSearch] = None
vectorstore: Optional[FaissVectorStore] = None

# Pydantic models for request/response
class SearchRequest(BaseModel):
    query: str
    top_k: Optional[int] = search_config.default_top_k
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "query": "What are the main topics discussed in the documents?",
                "top_k": 3
            }
        }
    }
    
class SearchResponse(BaseModel):
    query: str
    summary: str
    sources: List[Dict[str, Any]]
    total_chunks: int

class HealthResponse(BaseModel):
    status: str
    vectorstore_exists: bool
    total_documents: Optional[int] = None

class DocumentInfo(BaseModel):
    filename: str
    size: int
    type: str

class DocumentsResponse(BaseModel):
    documents: List[DocumentInfo]
    total_count: int

class UploadResponse(BaseModel):
    message: str
    uploaded_files: List[str]
    vectorstore_rebuilt: bool

def initialize_rag():
    """Initialize RAG components"""
    global rag_search, vectorstore
    try:
        rag_search = RAGSearch()
        vectorstore = rag_search.vectorstore
        return True
    except Exception as e:
        print(f"[ERROR] Failed to initialize RAG: {e}")
        return False

def get_documents_info() -> List[DocumentInfo]:
    """Get information about documents in the data folder"""
    data_path = Path(doc_config.data_directory)
    documents = []
    
    if data_path.exists():
        for file_path in data_path.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in doc_config.supported_formats:
                documents.append(DocumentInfo(
                    filename=file_path.name,
                    size=file_path.stat().st_size,
                    type=file_path.suffix.lower()
                ))
    
    return documents

@app.on_event("startup")
async def startup_event():
    """Initialize RAG on startup"""
    print("[INFO] Starting RAG API...")
    success = initialize_rag()
    if success:
        print("[INFO] RAG API initialized successfully")
    else:
        print("[WARN] RAG API started but initialization failed")

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main HTML interface"""
    static_file = Path("static/index.html")
    if static_file.exists():
        try:
            return HTMLResponse(content=static_file.read_text(encoding='utf-8'), status_code=200)
        except Exception as e:
            print(f"[ERROR] Failed to read HTML file: {e}")
            return HTMLResponse(content=f"""
            <html>
                <body>
                    <h1>RAG API is running</h1>
                    <p>API documentation available at <a href="/docs">/docs</a></p>
                    <p>Health check available at <a href="/health">/health</a></p>
                    <p>Error loading interface: {str(e)}</p>
                </body>
            </html>
            """, status_code=200)
    else:
        return HTMLResponse(content="""
        <html>
            <body>
                <h1>RAG API is running</h1>
                <p>API documentation available at <a href="/docs">/docs</a></p>
                <p>Health check available at <a href="/health">/health</a></p>
                <p>Static files not found. Expected at: static/index.html</p>
            </body>
        </html>
        """, status_code=200)

@app.get("/api", response_model=Dict[str, str])
async def api_info():
    """API information endpoint"""
    return {
        "message": "RAG API is running",
        "docs": "/docs",
        "health": "/health",
        "version": "1.0.0"
    }

@app.get("/favicon.ico")
async def favicon():
    """Return a simple favicon to prevent 404 errors"""
    return Response(status_code=204)

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    global vectorstore
    
    vectorstore_exists = False
    total_documents = None
    
    # Check if vectorstore exists
    vectorstore_config = config.get_vectorstore_config()
    faiss_path = Path(vectorstore_config.persist_directory) / "faiss.index"
    meta_path = Path(vectorstore_config.persist_directory) / "metadata.pkl"
    
    if faiss_path.exists() and meta_path.exists():
        vectorstore_exists = True
        if vectorstore and vectorstore.metadata:
            total_documents = len(vectorstore.metadata)
    
    return HealthResponse(
        status="healthy" if rag_search else "degraded",
        vectorstore_exists=vectorstore_exists,
        total_documents=total_documents
    )

@app.get("/documents", response_model=DocumentsResponse)
async def list_documents():
    """List all documents in the data folder"""
    documents = get_documents_info()
    return DocumentsResponse(
        documents=documents,
        total_count=len(documents)
    )

@app.post("/documents/upload", response_model=UploadResponse)
async def upload_documents(files: List[UploadFile] = File(...)):
    """Upload documents and rebuild vectorstore"""
    global rag_search, vectorstore
    
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")
    
    # Validate file count
    if len(files) > doc_config.max_files_per_upload:
        raise HTTPException(
            status_code=400, 
            detail=f"Too many files. Maximum {doc_config.max_files_per_upload} files allowed per upload."
        )
    
    # Create data directory if it doesn't exist
    data_path = Path(doc_config.data_directory)
    data_path.mkdir(exist_ok=True)
    
    uploaded_files = []
    
    try:
        # Save uploaded files
        for file in files:
            if not file.filename:
                continue
                
            # Check file extension
            file_ext = Path(file.filename).suffix.lower()
            if file_ext not in doc_config.supported_formats:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Unsupported file type: {file_ext}. Supported: {', '.join(doc_config.supported_formats)}"
                )
            
            # Check file size
            content = await file.read()
            file_size_mb = len(content) / (1024 * 1024)
            if file_size_mb > doc_config.max_file_size_mb:
                raise HTTPException(
                    status_code=400,
                    detail=f"File {file.filename} is too large ({file_size_mb:.1f}MB). Maximum size: {doc_config.max_file_size_mb}MB"
                )
            
            file_path = data_path / file.filename
            
            # Save file
            with open(file_path, "wb") as buffer:
                buffer.write(content)
            
            uploaded_files.append(file.filename)
        
        # Rebuild vectorstore
        print("[INFO] Rebuilding vectorstore with new documents...")
        docs = load_all_documents(doc_config.data_directory)
        
        if not docs:
            raise HTTPException(status_code=400, detail="No valid documents found to process")
        
        # Initialize new vectorstore
        vectorstore_config = config.get_vectorstore_config()
        vectorstore = FaissVectorStore(vectorstore_config.persist_directory)
        vectorstore.build_from_documents(docs)
        
        # Reinitialize RAG search
        rag_search = RAGSearch()
        
        return UploadResponse(
            message=f"Successfully uploaded {len(uploaded_files)} files and rebuilt vectorstore",
            uploaded_files=uploaded_files,
            vectorstore_rebuilt=True
        )
        
    except Exception as e:
        # Clean up uploaded files on error
        for filename in uploaded_files:
            file_path = data_path / filename
            if file_path.exists():
                file_path.unlink()
        
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.post("/search", response_model=SearchResponse)
async def search_documents(request: SearchRequest):
    """Search documents using RAG"""
    global rag_search
    
    if not rag_search:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    # Validate top_k
    top_k = request.top_k or search_config.default_top_k
    if top_k < search_config.min_top_k or top_k > search_config.max_top_k:
        raise HTTPException(
            status_code=400, 
            detail=f"top_k must be between {search_config.min_top_k} and {search_config.max_top_k}"
        )
    
    try:
        # Perform RAG search
        summary = rag_search.search_and_summarize(request.query, top_k=top_k)
        
        # Get detailed source information
        results = rag_search.vectorstore.query(request.query, top_k=top_k)
        
        sources = []
        for i, result in enumerate(results):
            text = result["metadata"]["text"]
            text_preview = text[:search_config.text_preview_length]
            if len(text) > search_config.text_preview_length:
                text_preview += "..."
            
            source = {
                "chunk_id": i,
                "text_preview": text_preview,
                "full_text": text
            }
            
            # Include distance if configured
            if search_config.include_distances and "distance" in result:
                source["distance"] = float(result["distance"])
            
            sources.append(source)
        
        return SearchResponse(
            query=request.query,
            summary=summary,
            sources=sources,
            total_chunks=len(rag_search.vectorstore.metadata) if rag_search.vectorstore.metadata else 0
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@app.delete("/vectorstore")
async def clear_vectorstore():
    """Clear the vectorstore and all processed data"""
    global rag_search, vectorstore
    
    try:
        # Remove vectorstore files
        vectorstore_config = config.get_vectorstore_config()
        vectorstore_path = Path(vectorstore_config.persist_directory)
        if vectorstore_path.exists():
            shutil.rmtree(vectorstore_path)
        
        # Reset global variables
        rag_search = None
        vectorstore = None
        
        return {"message": "Vectorstore cleared successfully"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear vectorstore: {str(e)}")

@app.delete("/documents")
async def clear_documents():
    """Clear all documents from the data folder"""
    try:
        data_path = Path(doc_config.data_directory)
        if data_path.exists():
            for file_path in data_path.iterdir():
                if file_path.is_file():
                    file_path.unlink()
        
        return {"message": "All documents cleared successfully"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear documents: {str(e)}")

if __name__ == "__main__":
    # Check for required environment variables
    gemini_api_key = os.getenv(llm_config.api_key_env) or os.getenv(llm_config.fallback_api_key_env)
    if not gemini_api_key:
        print(f"[ERROR] API key not found. Set {llm_config.api_key_env} or {llm_config.fallback_api_key_env} in your environment.")
        exit(1)
    
    # Run the application
    print("[INFO] Starting RAG API server...")
    print(f"[INFO] Configuration loaded from config.yaml")
    print(f"[INFO] Server: {server_config.host}:{server_config.port}")
    print(f"[INFO] Data directory: {doc_config.data_directory}")
    print(f"[INFO] Supported formats: {doc_config.supported_formats}")
    
    uvicorn.run(
        "api:app",
        host=server_config.host,
        port=server_config.port,
        reload=server_config.reload,
        log_level=server_config.log_level
    )