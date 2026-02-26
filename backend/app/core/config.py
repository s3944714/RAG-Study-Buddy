from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    env: str = "development"
    debug: bool = False

    # Database (used in later modules)
    database_url: str = "sqlite+aiosqlite:///./dev.db"

    # LLM / embedding — loaded from env, never hard-coded
    openai_api_key: str = ""
    embedding_model: str = "text-embedding-3-small"
    llm_model: str = "gpt-4o-mini"

    # Vector store
    chroma_host: str = "localhost"
    chroma_port: int = 8000


settings = Settings()