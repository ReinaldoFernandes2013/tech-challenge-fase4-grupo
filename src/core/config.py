from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configurações centrais da aplicação e do pipeline RAG."""
    
    # Caminhos do Projeto
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent
    DATA_RAW_DIR: Path = BASE_DIR / "data" / "raw"
    DATA_PROCESSED_DIR: Path = BASE_DIR / "data" / "processed"
    VECTOR_DB_DIR: Path = DATA_PROCESSED_DIR / "chroma_db"
    
    # Configurações de Modelo
    OPENAI_API_KEY: str = ""
    LLM_MODEL: str = "gpt-4o-mini"
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    TEMPERATURE: float = 0.0
    
    # Parâmetros de Recuperação (RRF e Re-ranking)
    COLLECTION_NAME: str = "olist_customer_reviews"
    BM25_TOP_K: int = 15
    VECTOR_TOP_K: int = 15
    RERANK_TOP_K: int = 5
    RRF_K: int = 60
    
    # Observabilidade (LangSmith)
    LANGCHAIN_TRACING_V2: bool = False
    LANGCHAIN_ENDPOINT: str = "https://api.smith.langchain.com"
    LANGCHAIN_API_KEY: str = ""
    LANGCHAIN_PROJECT: str = "olist-voice-of-customer-rag"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


settings = Settings()