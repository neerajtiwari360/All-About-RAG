import os
import shutil
from pathlib import Path
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn
from dotenv import load_dotenv

from src.data_loader import load_all_documents
from src.vectorstore import FaissVectorStore
from src.search import RAGSearch

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="RAG API",
    description="Retrieval-Augmented Generation API for document search and question answering",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure as needed for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for RAG components
rag_search: Optional[RAGSearch] = None
vectorstore: Optional[FaissVectorStore] = None

# Pydantic models for request/response
class SearchRequest(BaseModel):
    query: str
    top_k: Optional[int] = 3
    
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
    data_path = Path("data")
    documents = []
    
    if data_path.exists():
        for file_path in data_path.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in ['.pdf', '.txt', '.csv', '.xlsx', '.docx', '.json']:
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

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information"""
    return {
        "message": "RAG API is running",
        "docs": "/docs",
        "health": "/health",
        "version": "1.0.0"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    global vectorstore
    
    vectorstore_exists = False
    total_documents = None
    
    # Check if vectorstore exists
    faiss_path = Path("faiss_store/faiss.index")
    meta_path = Path("faiss_store/metadata.pkl")
    
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
    
    # Create data directory if it doesn't exist
    data_path = Path("data")
    data_path.mkdir(exist_ok=True)
    
    uploaded_files = []
    
    try:
        # Save uploaded files
        for file in files:
            if not file.filename:
                continue
                
            # Check file extension
            file_ext = Path(file.filename).suffix.lower()
            if file_ext not in ['.pdf', '.txt', '.csv', '.xlsx', '.docx', '.json']:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Unsupported file type: {file_ext}. Supported: .pdf, .txt, .csv, .xlsx, .docx, .json"
                )
            
            file_path = data_path / file.filename
            
            # Save file
            with open(file_path, "wb") as buffer:
                content = await file.read()
                buffer.write(content)
            
            uploaded_files.append(file.filename)
        
        # Rebuild vectorstore
        print("[INFO] Rebuilding vectorstore with new documents...")
        docs = load_all_documents("data")
        
        if not docs:
            raise HTTPException(status_code=400, detail="No valid documents found to process")
        
        # Initialize new vectorstore
        vectorstore = FaissVectorStore("faiss_store")
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
    
    try:
        # Perform RAG search
        summary = rag_search.search_and_summarize(request.query, top_k=request.top_k)
        
        # Get detailed source information
        results = rag_search.vectorstore.query(request.query, top_k=request.top_k)
        
        sources = []
        for i, result in enumerate(results):
            sources.append({
                "chunk_id": i,
                "distance": float(result["distance"]),
                "text_preview": result["metadata"]["text"][:200] + "..." if len(result["metadata"]["text"]) > 200 else result["metadata"]["text"],
                "full_text": result["metadata"]["text"]
            })
        
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
        vectorstore_path = Path("faiss_store")
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
        data_path = Path("data")
        if data_path.exists():
            for file_path in data_path.iterdir():
                if file_path.is_file():
                    file_path.unlink()
        
        return {"message": "All documents cleared successfully"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear documents: {str(e)}")

if __name__ == "__main__":
    # Check for required environment variables
    gemini_api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not gemini_api_key:
        print("[ERROR] Gemini API key not found. Set GEMINI_API_KEY or GOOGLE_API_KEY in your environment.")
        exit(1)
    
    # Run the application
    print("[INFO] Starting RAG API server...")
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )