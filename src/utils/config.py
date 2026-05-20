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

    # External services
    GROBID_URL: str = "http://localhost:8070"
    OPENALEX_BASE_URL: str = "https://api.openalex.org"

    # Embedder
    EMBEDDER_MODEL_NAME: str = "dunzhang/stella_en_1.5B_v5"
    EMBEDDER_BATCH_SIZE: int = 16
    EMBEDDER_DIM: int = 1024
    # Empty string = auto-detect (cuda → mps → cpu). Override with
    # "cuda", "cuda:0", "cpu", "mps", etc.
    EMBEDDER_DEVICE: str = ""
    # Stella performs best on this symmetric sentence-retrieval task when
    # queries and database contexts are both embedded raw (without prompts).
    # Empty string disables prompt wrapping.
    EMBEDDER_QUERY_PROMPT_NAME: str = ""

    model_config = {
        "env_file": str(_ENV_FILE),
        "env_file_encoding": "utf-8",
        # Allow extra env vars in .env without erroring
        "extra": "ignore",
    }

config = Settings()
