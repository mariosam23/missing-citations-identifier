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
    EMBEDDER_MODEL_NAME: str = "BAAI/bge-large-en-v1.5"
    EMBEDDER_BATCH_SIZE: int = 16
    EMBEDDER_DIM: int = 1024
    # Empty string = auto-detect (cuda → mps → cpu). Override with
    # "cuda", "cuda:0", "cpu", "mps", etc.
    EMBEDDER_DEVICE: str = ""
    # BGE-large performs best on this asymmetric sentence-retrieval task when
    # queries are wrapped in the query instruction prefix, and database contexts
    # are embedded raw.
    EMBEDDER_QUERY_PROMPT_NAME: str = "query"

    # Citation-need identifier (LLM-based binary "should this sentence cite?"
    # decision). Pin the model + run at temperature 0 for reproducible results.
    # NOTE: the gemini-2.0-* models return quota 0 on this account; only the
    # 2.5 / 3.1 flash tiers actually serve requests, so the rotation lists those.
    GEMINI_MODEL_NAME: str = "gemini-2.5-flash-lite"
    # Comma-separated fallbacks the rotating client cycles through to spread
    # per-model rate limits. The primary model above is tried first.
    GEMINI_FALLBACK_MODELS: str = "gemini-3.1-flash-lite,gemini-2.5-flash"
    CITATION_NEED_MIN_CONFIDENCE: float = 0.5
    CITATION_NEED_MAX_SENTENCES: int = 400
    # Skip sentences below this word count before spending an API call on them.
    CITATION_NEED_MIN_WORDS: int = 5

    def gemini_model_rotation(self) -> list[str]:
        """Ordered, de-duplicated model list: primary first, then fallbacks."""
        ordered = [
            self.GEMINI_MODEL_NAME.strip(),
            *(m.strip() for m in self.GEMINI_FALLBACK_MODELS.split(",")),
        ]
        seen: set[str] = set()
        rotation: list[str] = []
        for model in ordered:
            if model and model not in seen:
                seen.add(model)
                rotation.append(model)
        return rotation

    model_config = {
        "env_file": str(_ENV_FILE),
        "env_file_encoding": "utf-8",
        # Allow extra env vars in .env without erroring
        "extra": "ignore",
    }

config = Settings()
