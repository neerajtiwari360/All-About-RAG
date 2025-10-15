"""
Configuration management for the RAG system
Loads and provides access to configuration from config.yaml
"""

import yaml
import os
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass

@dataclass
class ServerConfig:
    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = True
    log_level: str = "info"
    title: str = "RAG API"
    description: str = "Retrieval-Augmented Generation API"
    version: str = "1.0.0"
    docs_url: str = "/docs"
    redoc_url: str = "/redoc"

@dataclass
class ChunkingConfig:
    chunk_size: int = 1000
    chunk_overlap: int = 200
    separators: list = None
    length_function: str = "len"
    
    def __post_init__(self):
        if self.separators is None:
            self.separators = ["\n\n", "\n", " ", ""]

@dataclass
class EmbeddingConfig:
    model_name: str = "all-MiniLM-L6-v2"
    batch_size: int = 32
    show_progress_bar: bool = True
    device: str = "auto"
    
    def __post_init__(self):
        # Auto-detect device if set to "auto"
        if self.device == "auto":
            try:
                import torch
                if torch.cuda.is_available():
                    self.device = "cuda"
                elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    self.device = "mps"
                else:
                    self.device = "cpu"
                print(f"[INFO] Auto-detected device: {self.device}")
            except ImportError:
                self.device = "cpu"
                print(f"[INFO] PyTorch not available, using CPU")

@dataclass
class VectorStoreConfig:
    persist_directory: str = "faiss_store"
    index_type: str = "IndexFlatL2"

@dataclass
class SearchConfig:
    default_top_k: int = 3
    max_top_k: int = 20
    min_top_k: int = 1
    include_distances: bool = True
    include_metadata: bool = True
    text_preview_length: int = 200

@dataclass
class LLMConfig:
    provider: str = "gemini"
    model_name: str = "gemini-2.5-flash"
    api_key_env: str = "GEMINI_API_KEY"
    fallback_api_key_env: str = "GOOGLE_API_KEY"
    temperature: float = 0.3
    max_tokens: int = 1000
    top_p: float = 0.9

@dataclass
class DocumentConfig:
    data_directory: str = "data"
    supported_formats: list = None
    max_file_size_mb: int = 50
    max_files_per_upload: int = 10
    
    def __post_init__(self):
        if self.supported_formats is None:
            self.supported_formats = [".pdf", ".txt", ".csv", ".xlsx", ".docx", ".json"]

class ConfigManager:
    """Centralized configuration management"""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = Path(config_path)
        self._config: Dict[str, Any] = {}
        self._load_config()
    
    def _load_config(self):
        """Load configuration from YAML file"""
        if not self.config_path.exists():
            print(f"[WARN] Config file {self.config_path} not found. Using default values.")
            self._config = self._get_default_config()
            return
        
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self._config = yaml.safe_load(f) or {}
            print(f"[INFO] Loaded configuration from {self.config_path}")
        except Exception as e:
            print(f"[ERROR] Failed to load config file: {e}")
            print("[INFO] Using default configuration")
            self._config = self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            "server": {},
            "chunking": {},
            "embedding": {},
            "vectorstore": {},
            "search": {},
            "llm": {},
            "documents": {},
            "prompts": {
                "rag_template": """Context information from relevant documents:
{context}

Question: {query}

Based on the context above, provide a helpful and accurate answer to the question.
If the context doesn't contain sufficient information to answer the question, please state that clearly.

Answer:"""
            }
        }
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by dot notation key"""
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def get_server_config(self) -> ServerConfig:
        """Get server configuration"""
        server_data = self._config.get("server", {})
        return ServerConfig(**server_data)
    
    def get_chunking_config(self) -> ChunkingConfig:
        """Get chunking configuration"""
        chunking_data = self._config.get("chunking", {})
        return ChunkingConfig(**chunking_data)
    
    def get_embedding_config(self) -> EmbeddingConfig:
        """Get embedding configuration"""
        embedding_data = self._config.get("embedding", {})
        return EmbeddingConfig(**embedding_data)
    
    def get_vectorstore_config(self) -> VectorStoreConfig:
        """Get vector store configuration"""
        vectorstore_data = self._config.get("vectorstore", {})
        return VectorStoreConfig(**vectorstore_data)
    
    def get_search_config(self) -> SearchConfig:
        """Get search configuration"""
        search_data = self._config.get("search", {})
        return SearchConfig(**search_data)
    
    def get_llm_config(self) -> LLMConfig:
        """Get LLM configuration"""
        llm_data = self._config.get("llm", {})
        
        # Handle provider-specific configs
        provider = llm_data.get("provider", "gemini")
        provider_config = llm_data.get(provider, {})
        
        # Merge general LLM config with provider-specific config
        merged_config = {**llm_data, **provider_config}
        merged_config["provider"] = provider
        
        return LLMConfig(**{k: v for k, v in merged_config.items() 
                          if k in LLMConfig.__dataclass_fields__})
    
    def get_document_config(self) -> DocumentConfig:
        """Get document configuration"""
        doc_data = self._config.get("documents", {})
        return DocumentConfig(**doc_data)
    
    def get_prompt_template(self, template_name: str = "rag_template") -> str:
        """Get prompt template"""
        prompts = self._config.get("prompts", {})
        return prompts.get(template_name, """Context: {context}\n\nQuestion: {query}\n\nAnswer:""")
    
    def get_cors_config(self) -> Dict[str, Any]:
        """Get CORS configuration"""
        return self._config.get("cors", {
            "allow_origins": ["*"],
            "allow_credentials": True,
            "allow_methods": ["*"],
            "allow_headers": ["*"]
        })
    
    def reload(self):
        """Reload configuration from file"""
        self._load_config()
        print("[INFO] Configuration reloaded")
    
    def save_config(self, config_path: Optional[str] = None):
        """Save current configuration to file"""
        path = Path(config_path) if config_path else self.config_path
        
        try:
            with open(path, 'w', encoding='utf-8') as f:
                yaml.dump(self._config, f, default_flow_style=False, indent=2)
            print(f"[INFO] Configuration saved to {path}")
        except Exception as e:
            print(f"[ERROR] Failed to save config: {e}")
    
    def update_config(self, key: str, value: Any):
        """Update configuration value"""
        keys = key.split('.')
        config = self._config
        
        # Navigate to the parent of the target key
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        # Set the value
        config[keys[-1]] = value
        print(f"[INFO] Updated config: {key} = {value}")

# Global configuration instance
config = ConfigManager()

# Convenience functions for backward compatibility
def get_server_config() -> ServerConfig:
    return config.get_server_config()

def get_chunking_config() -> ChunkingConfig:
    return config.get_chunking_config()

def get_embedding_config() -> EmbeddingConfig:
    return config.get_embedding_config()

def get_vectorstore_config() -> VectorStoreConfig:
    return config.get_vectorstore_config()

def get_search_config() -> SearchConfig:
    return config.get_search_config()

def get_llm_config() -> LLMConfig:
    return config.get_llm_config()

def get_document_config() -> DocumentConfig:
    return config.get_document_config()

# Example usage
if __name__ == "__main__":
    # Test configuration loading
    print("Testing configuration loading...")
    
    server_config = config.get_server_config()
    print(f"Server config: {server_config}")
    
    chunking_config = config.get_chunking_config()
    print(f"Chunking config: {chunking_config}")
    
    embedding_config = config.get_embedding_config()
    print(f"Embedding config: {embedding_config}")
    
    llm_config = config.get_llm_config()
    print(f"LLM config: {llm_config}")
    
    # Test template
    template = config.get_prompt_template()
    print(f"RAG template: {template[:100]}...")