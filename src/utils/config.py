from pathlib import Path

from pydantic_settings import BaseSettings

# Resolve .env relative to this file so it is found regardless of the
# working directory the process is started from.
_ENV_FILE = Path(__file__).resolve().parent.parent.parent / ".env"


class Settings(BaseSettings):
    """Typed, validated application settings."""

    # Remote APIs / DB
    DB_URL: str = ""
    OPEN_ALEX_API_KEY: str = ""
    S2_API_Key: str = ""
    OPEN_ALEX_EMAIL: str = ""
    OPEN_ROUTER_API_KEY: str = ""
    GEMINI_API_KEY: str = ""
    QDRANT_URL: str = ""

    # External services
    GROBID_URL: str = "http://localhost:8070"
    OPENALEX_BASE_URL: str = "https://api.openalex.org"

    # Embedding and indexing defaults
    QDRANT_COLLECTION_NAME: str = "papers"
    DENSE_MODEL: str = "intfloat/multilingual-e5-large-instruct"
    SPARSE_MODEL: str = "prithivida/Splade_PP_en_v1"

    # Pipeline model defaults (overridable at call site)
    RERANKER_MODEL: str = "BAAI/bge-reranker-v2-m3"
    DECOMPOSER_MODEL: str = "gemini-3-flash-preview"
    CLASSIFIER_MODEL: str = "gemini-3-flash-preview"
    CLASSIFIER_BACKUP: list[str] = [
        "gemini-3.1-flash-lite",
        "gemini-3.1-flash-lite-preview",
        "gemini-2.5-flash",
        "gemini-2.5-flash-lite"
    ]

    model_config = {
        "env_file": str(_ENV_FILE),
        "env_file_encoding": "utf-8",
        # Allow extra env vars in .env without erroring
        "extra": "ignore",
    }

    def validate_required(self, *fields: str) -> None:
        """Raise RuntimeError immediately if any listed setting is empty.

        Call this from entry points so missing secrets fail at startup with a
        useful message instead of surfacing as an obscure 401/refused-connection
        the first time the field is read.
        """
        missing = [f for f in fields if not str(getattr(self, f, "")).strip()]
        if missing:
            raise RuntimeError(
                "Required configuration is missing or empty: "
                + ", ".join(missing)
                + ". Set these in your .env file."
            )


config = Settings()
