from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Typed, validated application settings."""

    # Remote APIs / DB
    DB_URL: str = ""
    OPEN_ALEX_API_KEY: str = ""
    OPEN_ALEX_EMAIL: str = ""
    OPEN_ROUTER_API_KEY: str = ""
    QDRANT_URL: str = ""

    # Embedding and indexing defaults
    QDRANT_COLLECTION_NAME: str = "papers"
    DENSE_MODEL: str = "intfloat/multilingual-e5-large-instruct"
    SPARSE_MODEL: str = "prithivida/Splade_PP_en_v1"

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        # Allow extra env vars in .env without erroring
        "extra": "ignore",
    }


config = Settings()
