# All-About-RAG

A comprehensive implementation of Retrieval-Augmented Generation (RAG) from basic to advanced concepts with practical evaluation methods.

## 🤖 What is RAG?

Retrieval-Augmented Generation (RAG) is a powerful AI technique that combines the strengths of large language models with external knowledge retrieval. Instead of relying solely on the model's training data, RAG retrieves relevant information from external documents and uses that context to generate more accurate, up-to-date, and factually grounded responses.

## 🔄 How RAG Works: Step-by-Step Implementation

This repository demonstrates a complete RAG pipeline with the following components:

### Step 1: Document Loading (`src/data_loader.py`)

**Purpose**: Load and parse various document formats into a unified structure.

```python
# Supported formats: PDF, TXT, CSV, Excel, Word, JSON
docs = load_all_documents("data")
```

**What happens here**:
- 📄 **Document Discovery**: Recursively finds all supported files in the data directory
- 🔄 **Format Parsing**: Uses specialized loaders for each file type (PyPDFLoader, TextLoader, etc.)
- 📝 **Content Extraction**: Converts raw files into LangChain document objects with text content and metadata
- ✅ **Error Handling**: Gracefully handles corrupted or unsupported files

**Key Features**:
- Multi-format support (PDF, TXT, CSV, Excel, Word, JSON)
- Recursive directory scanning
- Robust error handling with detailed logging

### Step 2: Text Chunking and Embedding (`src/embedding.py`)

**Purpose**: Break documents into manageable chunks and convert them into numerical vectors.

```python
# Initialize embedding pipeline
emb_pipe = EmbeddingPipeline(model_name="all-MiniLM-L6-v2", chunk_size=1000, chunk_overlap=200)

# Split documents into chunks
chunks = emb_pipe.chunk_documents(docs)

# Generate embeddings
embeddings = emb_pipe.embed_chunks(chunks)
```

**What happens here**:

#### 2a. Text Chunking
- 📏 **Smart Splitting**: Uses `RecursiveCharacterTextSplitter` to break documents into chunks
- 🔄 **Overlap Strategy**: Maintains context between chunks with configurable overlap (default: 200 chars)
- 📊 **Size Control**: Ensures chunks fit within model context limits (default: 1000 chars)

#### 2b. Embedding Generation
- 🧠 **Semantic Encoding**: Converts text chunks into high-dimensional vectors using sentence-transformers
- 🎯 **Model Choice**: Uses `all-MiniLM-L6-v2` for fast, efficient embeddings (384 dimensions)
- 📈 **Batch Processing**: Processes multiple chunks efficiently with progress tracking

### Step 3: Vector Storage (`src/vectorstore.py`)

**Purpose**: Store embeddings in a searchable vector database for fast similarity retrieval.

```python
# Create and build vector store
store = FaissVectorStore("faiss_store")
store.build_from_documents(docs)
```

**What happens here**:
- 🗄️ **Vector Database**: Uses FAISS (Facebook AI Similarity Search) for efficient similarity search
- 💾 **Persistence**: Saves both the vector index and metadata to disk
- 🔍 **Indexing**: Creates a searchable index optimized for L2 distance similarity
- 📋 **Metadata Storage**: Maintains mapping between vectors and original text content

**Key Features**:
- Fast similarity search with FAISS
- Persistent storage (survives application restarts)
- Scalable to millions of vectors
- Metadata preservation for result context

### Step 4: Retrieval and Generation (`src/search.py`)

**Purpose**: Retrieve relevant documents and generate contextual responses using an LLM.

```python
# Initialize RAG search system
rag_search = RAGSearch()

# Search and summarize
query = "What is the doc about?"
summary = rag_search.search_and_summarize(query, top_k=3)
```

**What happens here**:

#### 4a. Query Processing
- 🔤 **Query Embedding**: Converts user query into the same vector space as documents
- 📊 **Similarity Search**: Finds most relevant document chunks using cosine similarity
- 🎯 **Top-K Retrieval**: Returns the most relevant chunks (configurable, default: 3-5)

#### 4b. Context Assembly
- 📝 **Context Building**: Combines retrieved chunks into coherent context
- 🔗 **Relevance Ranking**: Orders results by similarity score
- 📏 **Context Limiting**: Ensures context fits within LLM token limits

#### 4c. Response Generation
- 🤖 **LLM Integration**: Uses Google's Gemini model for natural language generation
- 🎯 **Prompted Generation**: Provides structured prompts with query and context
- ✨ **Contextual Responses**: Generates answers grounded in retrieved documents

## 🚀 Quick Start

### Prerequisites

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Environment Setup

Create a `.env` file with your API keys:
```bash
GEMINI_API_KEY=your_gemini_api_key_here
# or
GOOGLE_API_KEY=your_google_api_key_here
```

### Usage Options

#### Option 1: Command Line Interface (CLI)

1. **Add your documents** to the `data/` folder (supports PDF, TXT, CSV, Excel, Word, JSON)

2. **Run the application**:
```python
python app.py
```

3. **What happens when you run it**:
   - 📚 Loads all documents from the `data/` directory
   - 🔄 Checks for existing vector store, builds new one if needed
   - 🔍 Processes your query against the document knowledge base
   - 📝 Returns AI-generated summary based on relevant document content

#### Option 2: FastAPI Web Service

1. **Install additional dependencies** (already included in requirements.txt):
```bash
pip install fastapi uvicorn python-multipart aiofiles
```

2. **Start the API server**:
```bash
python api.py
# or
uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

3. **Access the API**:
   - 🌐 **API Server**: http://localhost:8000
   - 📚 **Interactive Docs**: http://localhost:8000/docs
   - 📖 **ReDoc Documentation**: http://localhost:8000/redoc

## 🔌 API Endpoints

### Health & Information

| Method | Endpoint | Description | Response |
|--------|----------|-------------|----------|
| `GET` | `/` | API information and available endpoints | Basic API info |
| `GET` | `/health` | Health check and system status | System health status |

### Document Management

| Method | Endpoint | Description | Request Body | Response |
|--------|----------|-------------|--------------|----------|
| `GET` | `/documents` | List all uploaded documents | None | Document list with metadata |
| `POST` | `/documents/upload` | Upload new documents | `multipart/form-data` with files | Upload confirmation |
| `DELETE` | `/documents` | Clear all documents | None | Deletion confirmation |

### Search & RAG

| Method | Endpoint | Description | Request Body | Response |
|--------|----------|-------------|--------------|----------|
| `POST` | `/search` | Perform RAG search and get AI summary | JSON with query and top_k | Search results and summary |

### Vector Store Management

| Method | Endpoint | Description | Request Body | Response |
|--------|----------|-------------|--------------|----------|
| `DELETE` | `/vectorstore` | Clear the vector database | None | Deletion confirmation |

## 📋 API Usage Examples

### 1. Health Check
```bash
curl -X GET "http://localhost:8000/health"
```

**Response:**
```json
{
  "status": "healthy",
  "vectorstore_exists": true,
  "total_documents": 15
}
```

### 2. Upload Documents
```bash
curl -X POST "http://localhost:8000/documents/upload" \
  -F "files=@document1.pdf" \
  -F "files=@document2.txt"
```

**Response:**
```json
{
  "message": "Successfully uploaded 2 files and rebuilt vectorstore",
  "uploaded_files": ["document1.pdf", "document2.txt"],
  "vectorstore_rebuilt": true
}
```

### 3. Search Documents
```bash
curl -X POST "http://localhost:8000/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the main technologies mentioned?",
    "top_k": 3
  }'
```

**Response:**
```json
{
  "query": "What are the main technologies mentioned?",
  "summary": "Based on the documents, the main technologies mentioned include...",
  "sources": [
    {
      "chunk_id": 0,
      "distance": 0.245,
      "text_preview": "The document discusses various technologies including Python, FastAPI...",
      "full_text": "Complete text of the relevant chunk..."
    }
  ],
  "total_chunks": 15
}
```

### 4. List Documents
```bash
curl -X GET "http://localhost:8000/documents"
```

**Response:**
```json
{
  "documents": [
    {
      "filename": "document1.pdf",
      "size": 245760,
      "type": ".pdf"
    }
  ],
  "total_count": 1
}
```

## 🧪 Testing with Postman

### Import the Collection

1. **Download Postman** from [postman.com](https://www.postman.com/)

2. **Import the collection**:
   - Open Postman
   - Click "Import"
   - Select `RAG_API_Collection.postman_collection.json`
   - Select `RAG_API_Environment.postman_environment.json`

3. **Set the environment**:
   - Click the environment dropdown (top right)
   - Select "RAG API Environment"

### Available Test Collections

#### 🏥 Health & Status
- **Root - API Info**: Get basic API information
- **Health Check**: Check system health and vectorstore status

#### 📄 Document Management
- **List Documents**: View all uploaded documents
- **Upload Documents**: Upload new files (PDF, TXT, CSV, Excel, Word, JSON)
- **Clear All Documents**: Remove all documents from the system

#### 🔍 Search & RAG
- **Basic Search**: Standard RAG query with default parameters
- **Search with High Precision**: Single best result (top_k=1)
- **Search with More Context**: Broader context search (top_k=5)
- **Technical Query Example**: Specialized technical question

#### 🗄️ Vector Store Management
- **Clear Vector Store**: Reset the entire vector database

#### ⚠️ Error Handling Tests
- **Search with Empty Query**: Test validation
- **Search with Invalid top_k**: Test parameter validation
- **Upload Unsupported File Type**: Test file type validation

### Testing Workflow

1. **Start the API server**:
   ```bash
   python api.py
   ```

2. **Check health**:
   - Run "Health Check" request
   - Verify `vectorstore_exists: false` initially

3. **Upload documents**:
   - Use "Upload Documents" request
   - Select test files from your system
   - Verify successful upload and vectorstore rebuild

4. **Test search functionality**:
   - Run various search requests
   - Try different `top_k` values
   - Test with different query types

5. **Validate error handling**:
   - Test edge cases with the error handling collection
   - Verify appropriate error messages

## 📊 API Schemas

### Request Models

#### SearchRequest
```json
{
  "query": "string (required)",
  "top_k": "integer (optional, default: 3)"
}
```

#### File Upload
- **Content-Type**: `multipart/form-data`
- **Field**: `files` (supports multiple files)
- **Supported formats**: `.pdf`, `.txt`, `.csv`, `.xlsx`, `.docx`, `.json`

### Response Models

#### SearchResponse
```json
{
  "query": "string",
  "summary": "string",
  "sources": [
    {
      "chunk_id": "integer",
      "distance": "float",
      "text_preview": "string",
      "full_text": "string"
    }
  ],
  "total_chunks": "integer"
}
```

#### HealthResponse
```json
{
  "status": "healthy|degraded",
  "vectorstore_exists": "boolean",
  "total_documents": "integer|null"
}
```

#### DocumentsResponse
```json
{
  "documents": [
    {
      "filename": "string",
      "size": "integer",
      "type": "string"
    }
  ],
  "total_count": "integer"
}
```

#### UploadResponse
```json
{
  "message": "string",
  "uploaded_files": ["string"],
  "vectorstore_rebuilt": "boolean"
}
```

## 🛠️ Development Setup

### Running in Development Mode

```bash
# Start with auto-reload
uvicorn api:app --reload --host 0.0.0.0 --port 8000

# Start with custom host/port
uvicorn api:app --host 127.0.0.1 --port 3000
```

### Environment Variables

| Variable | Required | Description | Default |
|----------|----------|-------------|---------|
| `GEMINI_API_KEY` | Yes | Google Gemini API key | None |
| `GOOGLE_API_KEY` | Yes | Alternative to GEMINI_API_KEY | None |

### Docker Support (Optional)

Create a `Dockerfile`:
```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
```

Run with Docker:
```bash
docker build -t rag-api .
docker run -p 8000:8000 -e GEMINI_API_KEY=your_key_here rag-api
```

## 🏗️ Architecture Overview

```
📁 Data Sources (PDF, TXT, etc.)
    ↓
📄 Document Loader (data_loader.py)
    ↓
✂️ Text Chunking (embedding.py)
    ↓
🧠 Embedding Generation (embedding.py)
    ↓
🗄️ Vector Storage (vectorstore.py)
    ↓
🔍 Similarity Search (search.py)
    ↓
🤖 LLM Generation (search.py)
    ↓
✨ Final Response
```

## 📊 Key Components

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Document Loading** | LangChain Loaders | Parse multiple file formats |
| **Text Splitting** | RecursiveCharacterTextSplitter | Intelligent document chunking |
| **Embeddings** | sentence-transformers | Convert text to vectors |
| **Vector DB** | FAISS | Fast similarity search |
| **LLM** | Google Gemini | Natural language generation |

## 🔧 Configuration Options

### Embedding Pipeline
```python
EmbeddingPipeline(
    model_name="all-MiniLM-L6-v2",  # Embedding model
    chunk_size=1000,                # Characters per chunk
    chunk_overlap=200               # Overlap between chunks
)
```

### Vector Store
```python
FaissVectorStore(
    persist_dir="faiss_store",      # Storage directory
    embedding_model="all-MiniLM-L6-v2"  # Must match embedding pipeline
)
```

### RAG Search
```python
RAGSearch(
    persist_dir="faiss_store",      # Vector store location
    embedding_model="all-MiniLM-L6-v2",  # Embedding model
    llm_model="gemini-2.5-flash"   # LLM for generation
)
```

## 🎯 Advanced Features

- **Multi-format Support**: Handles PDF, TXT, CSV, Excel, Word, and JSON files
- **Persistent Storage**: Vector indices survive application restarts
- **Configurable Chunking**: Adjust chunk size and overlap for your use case
- **Error Handling**: Robust error handling with detailed logging
- **Scalable Architecture**: Easily extensible for additional file formats or LLMs

## 🔮 Future Enhancements

- [ ] **Advanced RAG Techniques**: Query expansion, re-ranking, hybrid search
- [ ] **Evaluation Metrics**: Implement RAGAS, faithfulness, and relevance scoring
- [ ] **Multiple LLM Support**: OpenAI, Anthropic, local models
- [ ] **Advanced Chunking**: Semantic chunking, document structure awareness
- [ ] **Monitoring**: Performance metrics and query analytics
- [ ] **API Interface**: REST API for integration with other applications

## 🤝 Contributing

Feel free to contribute by:
- Adding support for new document formats
- Implementing advanced RAG techniques
- Adding evaluation metrics
- Improving documentation

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
