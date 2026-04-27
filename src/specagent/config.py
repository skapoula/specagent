"""
Configuration management using Pydantic Settings.

All configuration is loaded from environment variables with sensible defaults.
Use a .env file for local development.

Environment Variables:
    GROQ_API_KEY: Groq cloud API key (required when llm_provider='groq')
    EMBEDDING_MODEL: fastembed model ID for embeddings (e.g. nomic-ai/nomic-embed-text-v1.5)
    LANCEDB_URI: Path to LanceDB storage directory
    LOG_LEVEL: Logging level (DEBUG, INFO, WARNING, ERROR)
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict

# Recognised LLM backend identifiers.
LLMProvider = Literal["custom_endpoint", "groq"]


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
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
    # Env var: LLM_PROVIDER — selects LLM backend.
    # "groq":           Groq cloud inference API (default).
    # "custom_endpoint": OpenAI-compatible self-hosted endpoint.
    # "local":          Local GGUF model (not yet implemented).
    llm_provider: LLMProvider = Field(
        default="groq",
        description="LLM backend to use: groq | custom_endpoint | local",
    )
    use_custom_endpoint: bool = Field(
        default=False,
        description="Legacy: set llm_provider='custom_endpoint' instead.",
    )
    custom_endpoint_url: str = Field(
        default="http://qwen3-4b-predictor.ml-serving.10.0.1.2.sslip.io:30750/v1/chat/completions",
        description="Custom inference endpoint URL (OpenAI-compatible)",
    )

    # ---- Groq-specific (only used when llm_provider == "groq") ----
    # API key for the Groq inference API.  Required when llm_provider is "groq".
    # Env var: GROQ_API_KEY.
    groq_api_key: str = Field(
        default="",
        repr=False,  # prevent key from appearing in Settings repr/logs
        description="Groq cloud API key. Required when llm_provider='groq'.",
    )
    # Model served by Groq.
    # Free-tier limits (RPM / TPM / TPD) as of 2025-03:
    #   llama-4-scout-17b-16e-instruct : 30 / 30K / 500K  ← default (best free-tier budget)
    #   llama-3.3-70b-versatile        : 30 / 12K / 100K  (higher quality, tighter limits)
    #   llama-3.1-8b-instant           : 30 /  6K / 500K  (fastest, lowest per-minute budget)
    # specagent makes up to 5 LLM calls per query (~6K tokens total), so 30K TPM
    # allows ~4 concurrent queries/min vs ~1 for llama-3.3-70b-versatile.
    # Override via GROQ_MODEL env var.
    groq_model: str = Field(
        default="meta-llama/llama-4-scout-17b-16e-instruct",
        description=(
            "Groq model name. Default is llama-4-scout for best free-tier TPM/TPD. "
            "Use llama-3.3-70b-versatile for higher quality at the cost of tighter limits."
        ),
    )
    # Groq reasoning effort level.  Maps to Groq's reasoning_effort parameter;
    # only supported by reasoning models (e.g. qwq-32b).  Empty string disables it.
    # Env var: GROQ_REASONING_EFFORT
    groq_reasoning_effort: Literal["", "low", "medium", "high"] = Field(
        default="",
        description="Groq reasoning effort (low/medium/high). Leave empty for non-reasoning models.",
    )
    # Max completion tokens for Groq calls.  1024 is sufficient for all specagent
    # nodes (generator answers with citations, grader/router/hallucination return
    # small JSON).  Keeping this low conserves the free-tier TPM budget.
    groq_max_tokens: int = Field(
        default=1024,
        ge=1,
        le=32768,
        description="Maximum output tokens per Groq call. Keep low to stay within free-tier TPM limits.",
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
        default=0.5,
        ge=0.0,
        le=1.0,
        description=(
            "Minimum percentage of relevant chunks required to skip rewriting. "
            "With grader capped at top-3 chunks, 0.5 requires at least 2/3 chunks "
            "to be relevant before proceeding."
        ),
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
    cors_allow_origins: list[str] = Field(
        default=["http://localhost:3000"],
        description=(
            "Allowed CORS origins for the REST API. "
            "Set to the frontend URL(s) in production. "
            "Env var: CORS_ALLOW_ORIGINS (comma-separated)."
        ),
    )
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

    # Query journal — writes a JSONL record per completed query for offline analysis.
    # Disabled by default to avoid disk writes in CI / test environments.
    enable_query_journal: bool = Field(
        default=False,
        description="Write a JSONL query journal to journal_dir after each query.",
    )
    journal_dir: Path = Field(
        default=Path("data/journal"),
        description="Directory for JSONL query journal files.",
    )

    # LangSmith tracing (https://smith.langchain.com)
    # Set LANGCHAIN_API_KEY to activate. setup_langsmith_tracing() reads these.
    enable_langsmith: bool = Field(
        default=True,
        description="Enable LangSmith tracing. Requires LANGCHAIN_API_KEY to be set.",
    )
    langchain_api_key: str = Field(
        default="",
        repr=False,  # prevent key from appearing in Settings repr/logs
        description="LangSmith API key (env var: LANGCHAIN_API_KEY).",
    )
    langchain_project: str = Field(
        default="3gpp-specagent",
        description="LangSmith project name (env var: LANGCHAIN_PROJECT).",
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
    # Vision / OCR Configuration
    # ==========================================================================
    enable_docx_ocr: bool = Field(
        default=False,
        description=(
            "Enable two-pass OCR for .docx files using Groq vision. "
            "Requires GROQ_API_KEY. Set ENABLE_DOCX_OCR=true to activate."
        ),
    )
    vision_model: str = Field(
        default="meta-llama/llama-4-scout-17b-16e-instruct",
        description="Groq vision model for image analysis.",
    )
    vision_rpm_limit: int = Field(
        default=30,
        ge=1,
        le=600,
        description="Groq vision API requests-per-minute limit (free tier: 30).",
    )
    vision_rpd_limit: int = Field(
        default=1000,
        ge=1,
        description="Groq vision API requests-per-day limit (free tier: 1000).",
    )
    vision_tpm_limit: int = Field(
        default=30000,
        ge=1,
        description=(
            "Groq vision API tokens-per-minute limit (free tier: 30000 for llama-4-scout). "
            "Pre-flight TPM budgeting slows calls when the estimated token budget is nearly full. "
            "Env var: VISION_TPM_LIMIT."
        ),
    )
    vision_tokens_per_call_estimate: int = Field(
        default=2000,
        ge=1,
        description=(
            "Conservative token estimate per vision API call used for pre-flight TPM budgeting. "
            "Each call uses ~1024 output + ~1000 input tokens; 2000 is a safe upper bound. "
            "Env var: VISION_TOKENS_PER_CALL_ESTIMATE."
        ),
    )
    vision_min_image_bytes: int = Field(
        default=10 * 1024,  # 10 KB
        ge=1,
        description=(
            "Minimum image size in bytes to send to the vision API. "
            "Images smaller than this (logos, icons, decorative elements) are skipped. "
            "Env var: VISION_MIN_IMAGE_BYTES. Default: 10 KB."
        ),
    )
    vision_max_image_bytes: int = Field(
        default=20 * 1024 * 1024,  # 20 MB
        ge=1,
        description="Maximum image size in bytes to send to vision API. Larger images are skipped.",
    )
    vision_max_tokens: int = Field(
        default=4096,
        ge=256,
        le=32768,
        description=(
            "Max output tokens for Groq vision API calls. "
            "3GPP call-flow diagrams with 30–50 steps require ~2000 tokens. "
            "Env var: VISION_MAX_TOKENS."
        ),
    )
    vision_max_retries: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Maximum tenacity retries for transient Groq vision API errors.",
    )
    vision_cache_path: Path = Field(
        default=Path("data/vision_cache/results.json"),
        description=(
            "Path to the disk-backed JSON cache for Groq vision API results. "
            "Keyed by image content SHA-256 to avoid re-submitting unchanged images. "
            "Env var: VISION_CACHE_PATH."
        ),
    )
    vision_max_calls_per_run: int = Field(
        default=0,
        ge=0,
        description=(
            "Maximum vision API calls allowed per ingest_folder run. "
            "0 = unlimited. Env var: VISION_MAX_CALLS_PER_RUN."
        ),
    )
    vision_diagram_types: list[str] = Field(
        default=[
            "call_flow",
            "state_machine",
            "block_diagram",
            "flowchart",
            "network_topology",
        ],
        description=(
            "Diagram types for which Mermaid output is requested from the vision model. "
            "Env var: VISION_DIAGRAM_TYPES (comma-separated)."
        ),
    )

    # ==========================================================================
    # Groq LLM Rate Limit Configuration
    # Defaults match the free-tier limits for meta-llama/llama-4-scout-17b-16e-instruct.
    # ==========================================================================
    groq_llm_rpm_limit: int = Field(
        default=30,
        ge=1,
        le=600,
        description=(
            "Groq LLM API requests-per-minute limit. "
            "Free-tier default: 30. Env var: GROQ_LLM_RPM_LIMIT."
        ),
    )
    groq_llm_rpd_limit: int = Field(
        default=1000,
        ge=1,
        description=(
            "Groq LLM API requests-per-day limit. "
            "Free-tier default: 1000. Env var: GROQ_LLM_RPD_LIMIT."
        ),
    )
    groq_llm_tpm_limit: int = Field(
        default=30000,
        ge=1,
        description=(
            "Groq LLM API tokens-per-minute limit. "
            "Free-tier default: 30000. Env var: GROQ_LLM_TPM_LIMIT."
        ),
    )
    groq_llm_tpd_limit: int = Field(
        default=500000,
        ge=1,
        description=(
            "Groq LLM API tokens-per-day limit. "
            "Free-tier default: 500000. Env var: GROQ_LLM_TPD_LIMIT."
        ),
    )
    groq_llm_tokens_per_call_estimate: int = Field(
        default=6000,
        ge=1,
        description=(
            "Conservative token estimate per LLM call for pre-flight TPM budgeting. "
            "specagent makes up to 5 LLM calls per query at ~1200 tokens each. "
            "Env var: GROQ_LLM_TOKENS_PER_CALL_ESTIMATE."
        ),
    )
    groq_llm_max_retries: int = Field(
        default=6,
        ge=1,
        le=10,
        description=(
            "Maximum tenacity retries for 429/503/504 Groq LLM API errors. "
            "Env var: GROQ_LLM_MAX_RETRIES."
        ),
    )

    # ==========================================================================
    # Mermaid Validation Configuration
    # ==========================================================================
    mermaid_validate_with_mmdc: bool = Field(
        default=False,
        description=(
            "Enable Tier 2 Mermaid validation via mmdc subprocess. "
            "Requires @mermaid-js/mermaid-cli to be installed on PATH. "
            "Env var: MERMAID_VALIDATE_WITH_MMDC."
        ),
    )
    mermaid_mmdc_timeout: int = Field(
        default=10,
        ge=1,
        le=120,
        description=(
            "Timeout in seconds for mmdc subprocess validation. "
            "Timeouts are treated as valid (tool absence must not fail pipeline). "
            "Env var: MERMAID_MMDC_TIMEOUT."
        ),
    )

    # ==========================================================================
    # Kuzu Embedded Graph DB / DAG Store Configuration
    # ==========================================================================
    kuzu_db_path: Path = Field(
        default=Path("data/dag_store"),
        description=(
            "Path to the Kuzu embedded graph database directory for DAG storage. "
            "Created automatically on first use. Env var: KUZU_DB_PATH."
        ),
    )
    enable_dag_storage: bool = Field(
        default=False,
        description=(
            "Enable DAG storage for detected call-flow diagrams during .docx OCR ingest. "
            "Env var: ENABLE_DAG_STORAGE."
        ),
    )
    enable_dag_retrieval: bool = Field(
        default=False,
        description=(
            "Enable DAG-augmented retrieval in the RAG pipeline. "
            "Requires ENABLE_DAG_STORAGE=true. Env var: ENABLE_DAG_RETRIEVAL."
        ),
    )
    dag_retrieval_top_k: int = Field(
        default=1,
        ge=1,
        le=10,
        description="Max number of DAG results to inject per query. Env var: DAG_RETRIEVAL_TOP_K.",
    )
    dag_retrieval_score: float = Field(
        default=0.70,
        ge=0.0,
        le=1.0,
        description=(
            "Similarity score assigned to DAG-derived chunks injected into retrieval results. "
            "Env var: DAG_RETRIEVAL_SCORE."
        ),
    )

    # ==========================================================================
    # Validators
    # ==========================================================================
    @field_validator("chunk_overlap_tokens")
    @classmethod
    def validate_chunk_overlap_tokens(cls, v: int, info) -> int:
        """Ensure token overlap is less than token chunk size."""
        chunk_size_tokens = info.data.get("chunk_size_tokens", 512)
        if v >= chunk_size_tokens:
            raise ValueError(
                f"chunk_overlap_tokens ({v}) must be less than chunk_size_tokens ({chunk_size_tokens})"
            )
        return v

    @field_validator(
        "lancedb_uri",
        "docs_dir",
        "data_dir",
        "raw_data_dir",
        "processed_data_dir",
        "journal_dir",
        "kuzu_db_path",
        "vision_cache_path",
    )
    @classmethod
    def resolve_path(cls, v: Path) -> Path:
        """Resolve paths to absolute paths."""
        return v.resolve()

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Load configuration from .env file only — shell environment variables are excluded.

        This prevents ambient shell exports (e.g. EMBEDDING_MODEL set by another
        project sharing the same devcontainer) from silently overriding .env values.
        Constructor kwargs still take priority, which allows tests to pass overrides
        directly without touching the filesystem.

        IMPORTANT: Because shell exports are ignored, secrets such as GROQ_API_KEY
        must be supplied via a ``.env`` file (e.g. ``echo 'GROQ_API_KEY=...' >> .env``),
        not via ``export GROQ_API_KEY=...`` in the shell. Shell exports will be
        silently ignored.
        """
        return (init_settings, dotenv_settings, file_secret_settings)


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
