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

    model_config = {
        "env_file": str(_ENV_FILE),
        "env_file_encoding": "utf-8",
        # Allow extra env vars in .env without erroring
        "extra": "ignore",
    }

config = Settings()
