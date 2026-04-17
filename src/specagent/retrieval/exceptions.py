"""Domain exception classes for specagent retrieval pipeline."""


class UnsupportedFormatError(Exception):
    """Raised when a file extension or MIME type is not supported by the converter."""


class IngestionError(Exception):
    """Raised when the ingestion pipeline fails (fetch, conversion, or store write)."""


class StoreError(Exception):
    """Raised when a LanceDB read or write operation fails."""


class EmbeddingError(Exception):
    """Raised when the embedding model fails to produce vectors."""


class ConfigurationError(Exception):
    """Raised at startup when required settings are missing or inconsistent."""


class VisionError(Exception):
    """Raised when the Groq vision API call fails after all retries."""


class DagStoreError(Exception):
    """Raised when a Kuzu DAG store read or write operation fails."""


class LLMRateLimitError(Exception):
    """Raised when the Groq LLM API daily quota is exhausted and retry would exceed threshold."""
