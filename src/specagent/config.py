"""
Configuration management using Pydantic Settings.

All configuration is loaded from environment variables with sensible defaults.
Use a .env file for local development.

Environment Variables:
    HF_API_KEY: HuggingFace API key (optional, for embeddings API)
    EMBEDDING_MODEL: fastembed model ID for embeddings (e.g. nomic-ai/nomic-embed-text-v1.5)
    LLM_MODEL_PATH: Path to local GGUF model file
    CHUNK_SIZE: Token size for document chunks
    CHUNK_OVERLAP: Overlap between chunks
    LANCEDB_URI: Path to LanceDB storage directory
    LOG_LEVEL: Logging level (DEBUG, INFO, WARNING, ERROR)
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal, Optional

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ==========================================================================
    # API Keys (optional - only needed for HF API embeddings)
    # ==========================================================================
    hf_api_key: Optional[SecretStr] = Field(
        default=None,
        description="HuggingFace API key (optional, for API-based embeddings)",
    )

    # ==========================================================================
    # Model Configuration
    # ==========================================================================
    embedding_model: str = Field(
        default="nomic-ai/nomic-embed-text-v1.5",
        description="fastembed model ID for document/query embeddings",
    )
    embedding_dimension: int = Field(
        default=768,
        description="Vector dimension (must match embedding_model output)",
    )

    # LLM Configuration
    use_local_llm: bool = Field(
        default=False,
        description="Use local GGUF model instead of HF Inference API",
    )
    use_custom_endpoint: bool = Field(
        default=True,
        description="Use custom OpenAI-compatible endpoint instead of HuggingFace",
    )
    custom_endpoint_url: str = Field(
        default="http://qwen3-4b-predictor.ml-serving.10.0.1.2.sslip.io:30750/v1/chat/completions",
        description="Custom inference endpoint URL (OpenAI-compatible)",
    )
    
    # HuggingFace Inference API (when use_local_llm=False)
    llm_model: str = Field(
        default="Qwen/Qwen2.5-3B-Instruct",
        description="HuggingFace model ID for Inference API",
    )
    
    # Local GGUF model (when use_local_llm=True)
    llm_model_path: Path = Field(
        default=Path("/models/qwen3-4b-instruct.Q4_K_M.gguf"),
        description="Path to local GGUF model file",
    )
    llm_model_name: str = Field(
        default="qwen3-4b-instruct",
        description="Model name for logging/identification",
    )
    llm_n_ctx: int = Field(
        default=4096,
        ge=512,
        le=32768,
        description="Context window size for LLM",
    )
    llm_n_gpu_layers: int = Field(
        default=0,
        ge=0,
        description="Number of layers to offload to GPU (0 for CPU only)",
    )
    llm_temperature: float = Field(
        default=0.1,
        ge=0.0,
        le=2.0,
        description="Temperature for LLM generation (lower = more deterministic)",
    )
    llm_max_tokens: int = Field(
        default=1024,
        ge=1,
        le=4096,
        description="Maximum tokens for LLM response",
    )

    # ==========================================================================
    # Chunking Configuration
    # ==========================================================================
    chunk_size: int = Field(
        default=512,
        ge=128,
        le=2048,
        description="Target size in tokens for document chunks",
    )
    chunk_overlap: int = Field(
        default=64,
        ge=0,
        le=256,
        description="Overlap between consecutive chunks",
    )

    # ==========================================================================
    # LanceDB Configuration
    # ==========================================================================
    lancedb_uri: Path = Field(
        default=Path("data/lancedb"),
        description="Path to LanceDB storage directory",
    )
    lancedb_table_name: str = Field(
        default="documents",
        description="LanceDB table name",
    )
    default_library: str = Field(
        default="3gpp-specs",
        description="Default library name for ingested documents",
    )

    # ==========================================================================
    # Ingestion Configuration
    # ==========================================================================
    docs_dir: Path = Field(
        default=Path("data/docs"),
        description="Directory where user places input documents",
    )
    chunk_size_tokens: int = Field(
        default=512,
        ge=64,
        le=2048,
        description="Target chunk size in tokens",
    )
    chunk_overlap_tokens: int = Field(
        default=64,
        ge=0,
        le=512,
        description="Overlap in tokens between consecutive chunks",
    )
    chunk_min_tokens: int = Field(
        default=50,
        ge=1,
        le=256,
        description="Minimum chunk size; shorter chunks kept as-is",
    )
    embedding_batch_size: int = Field(
        default=32,
        ge=1,
        le=256,
        description="Batch size for fastembed inference",
    )
    max_ingest_concurrency: int = Field(
        default=4,
        ge=1,
        le=32,
        description="Maximum concurrent files during folder ingestion",
    )

    # ==========================================================================
    # Search Configuration
    # ==========================================================================
    hybrid_search_enabled: bool = Field(
        default=True,
        description="Enable BM25 full-text + vector hybrid search",
    )
    search_refine_factor: int = Field(
        default=10,
        ge=1,
        le=100,
        description="ANN re-ranking candidates (higher = better recall)",
    )

    # ==========================================================================
    # HTTP Configuration (for URL ingestion)
    # ==========================================================================
    http_timeout_seconds: int = Field(
        default=10,
        ge=1,
        le=300,
        description="Timeout for URL fetch requests in seconds",
    )
    http_user_agent: str = Field(
        default="specagent/1.0",
        description="User-Agent header for HTTP requests",
    )

    # ==========================================================================
    # Retrieval Configuration
    # ==========================================================================
    retrieval_top_k: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Number of chunks to retrieve",
    )
    similarity_threshold: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Minimum cosine similarity for retrieved chunks",
    )

    # ==========================================================================
    # Agent Configuration
    # ==========================================================================
    max_rewrites: int = Field(
        default=1,
        ge=0,
        le=5,
        description="Maximum number of query rewrites before giving up",
    )
    grader_confidence_threshold: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="Minimum average confidence to skip rewriting",
    )
    min_relevant_chunk_percentage: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Minimum percentage of relevant chunks required to skip rewriting",
    )
    high_similarity_threshold: float = Field(
        default=0.85,
        ge=0.0,
        le=1.0,
        description="Similarity threshold for top-3 chunks to skip rewriting (fast heuristic)",
    )

    # ==========================================================================
    # API Configuration
    # ==========================================================================
    api_host: str = Field(
        default="0.0.0.0",
        description="Host to bind API server",
    )
    api_port: int = Field(
        default=8000,
        ge=1,
        le=65535,
        description="Port for API server",
    )
    api_workers: int = Field(
        default=1,
        ge=1,
        le=4,
        description="Number of uvicorn workers",
    )

    # ==========================================================================
    # Observability Configuration
    # ==========================================================================
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        default="INFO",
        description="Logging level",
    )
    phoenix_endpoint: str = Field(
        default="http://localhost:6006",
        description="Arize Phoenix collector endpoint",
    )
    enable_tracing: bool = Field(
        default=True,
        description="Enable OpenTelemetry tracing to Phoenix",
    )

    # ==========================================================================
    # Data Paths
    # ==========================================================================
    data_dir: Path = Field(
        default=Path("data"),
        description="Root directory for data files",
    )
    raw_data_dir: Path = Field(
        default=Path("data/raw"),
        description="Directory for raw TSpec-LLM markdown files",
    )
    processed_data_dir: Path = Field(
        default=Path("data/processed"),
        description="Directory for processed chunks",
    )

    # ==========================================================================
    # Validators
    # ==========================================================================
    @field_validator("chunk_overlap")
    @classmethod
    def validate_chunk_overlap(cls, v: int, info) -> int:
        """Ensure overlap is less than chunk size."""
        chunk_size = info.data.get("chunk_size", 512)
        if v >= chunk_size:
            raise ValueError(f"chunk_overlap ({v}) must be less than chunk_size ({chunk_size})")
        return v

    @field_validator("lancedb_uri", "docs_dir", "data_dir", "raw_data_dir", "processed_data_dir")
    @classmethod
    def resolve_path(cls, v: Path) -> Path:
        """Resolve paths to absolute paths."""
        return v.resolve()


@lru_cache
def get_settings() -> Settings:
    """
    Get cached settings instance.
    
    Uses LRU cache to ensure settings are only loaded once.
    Call `get_settings.cache_clear()` to reload settings.
    
    Returns:
        Settings: Application settings instance
    """
    return Settings()


# Convenience alias
settings = get_settings()
