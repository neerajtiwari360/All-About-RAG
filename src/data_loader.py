from pathlib import Path
from typing import List, Any
from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders.excel import UnstructuredExcelLoader
from langchain_community.document_loaders import JSONLoader
from src.config import get_document_config

def load_all_documents(data_dir: str = None) -> List[Any]:
    """
    Load all supported files from the data directory and convert to LangChain document structure.
    Supported formats are defined in config.yaml
    """
    # Load configuration
    doc_config = get_document_config()
    
    # Use project root data folder from config or parameter
    data_dir = data_dir or doc_config.data_directory
    data_path = Path(data_dir).resolve()
    print(f"[DEBUG] Data path: {data_path}")
    print(f"[DEBUG] Supported formats: {doc_config.supported_formats}")
    
    documents = []

    # PDF files
    if ".pdf" in doc_config.supported_formats:
        pdf_files = list(data_path.glob('**/*.pdf'))
        print(f"[DEBUG] Found {len(pdf_files)} PDF files: {[str(f) for f in pdf_files]}")
        for pdf_file in pdf_files:
            print(f"[DEBUG] Loading PDF: {pdf_file}")
            try:
                loader = PyPDFLoader(str(pdf_file))
                loaded = loader.load()
                print(f"[DEBUG] Loaded {len(loaded)} PDF docs from {pdf_file}")
                documents.extend(loaded)
            except Exception as e:
                print(f"[ERROR] Failed to load PDF {pdf_file}: {e}")

    # TXT files
    if ".txt" in doc_config.supported_formats:
        txt_files = list(data_path.glob('**/*.txt'))
        print(f"[DEBUG] Found {len(txt_files)} TXT files: {[str(f) for f in txt_files]}")
        for txt_file in txt_files:
            print(f"[DEBUG] Loading TXT: {txt_file}")
            try:
                loader = TextLoader(str(txt_file))
                loaded = loader.load()
                print(f"[DEBUG] Loaded {len(loaded)} TXT docs from {txt_file}")
                documents.extend(loaded)
            except Exception as e:
                print(f"[ERROR] Failed to load TXT {txt_file}: {e}")

    # CSV files
    if ".csv" in doc_config.supported_formats:
        csv_files = list(data_path.glob('**/*.csv'))
        print(f"[DEBUG] Found {len(csv_files)} CSV files: {[str(f) for f in csv_files]}")
        for csv_file in csv_files:
            print(f"[DEBUG] Loading CSV: {csv_file}")
            try:
                loader = CSVLoader(str(csv_file))
                loaded = loader.load()
                print(f"[DEBUG] Loaded {len(loaded)} CSV docs from {csv_file}")
                documents.extend(loaded)
            except Exception as e:
                print(f"[ERROR] Failed to load CSV {csv_file}: {e}")

    # Excel files
    if ".xlsx" in doc_config.supported_formats:
        excel_files = list(data_path.glob('**/*.xlsx'))
        print(f"[DEBUG] Found {len(excel_files)} Excel files: {[str(f) for f in excel_files]}")
        for excel_file in excel_files:
            print(f"[DEBUG] Loading Excel: {excel_file}")
            try:
                loader = UnstructuredExcelLoader(str(excel_file))
                loaded = loader.load()
                print(f"[DEBUG] Loaded {len(loaded)} Excel docs from {excel_file}")
                documents.extend(loaded)
            except Exception as e:
                print(f"[ERROR] Failed to load Excel {excel_file}: {e}")

    # Word files
    if ".docx" in doc_config.supported_formats:
        word_files = list(data_path.glob('**/*.docx'))
        print(f"[DEBUG] Found {len(word_files)} Word files: {[str(f) for f in word_files]}")
        for word_file in word_files:
            print(f"[DEBUG] Loading Word: {word_file}")
            try:
                loader = Docx2txtLoader(str(word_file))
                loaded = loader.load()
                print(f"[DEBUG] Loaded {len(loaded)} Word docs from {word_file}")
                documents.extend(loaded)
            except Exception as e:
                print(f"[ERROR] Failed to load Word {word_file}: {e}")

    # JSON files
    if ".json" in doc_config.supported_formats:
        json_files = list(data_path.glob('**/*.json'))
        print(f"[DEBUG] Found {len(json_files)} JSON files: {[str(f) for f in json_files]}")
        for json_file in json_files:
            print(f"[DEBUG] Loading JSON: {json_file}")
            try:
                loader = JSONLoader(str(json_file), jq_schema=".", text_content=False)
                loaded = loader.load()
                print(f"[DEBUG] Loaded {len(loaded)} JSON docs from {json_file}")
                documents.extend(loaded)
            except Exception as e:
                print(f"[ERROR] Failed to load JSON {json_file}: {e}")

    print(f"[DEBUG] Total loaded documents: {len(documents)}")
    return documents

# Example usage
if __name__ == "__main__":
    docs = load_all_documents()
    print(f"Loaded {len(docs)} documents")
    for i, doc in enumerate(docs[:3]):  # Show first 3 documents
        print(f"Document {i+1}: {doc.page_content[:200]}...")