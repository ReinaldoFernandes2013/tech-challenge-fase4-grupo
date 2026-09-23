from typing import Any, Dict, List
from pydantic import BaseModel, Field
from src.schemas.rag_schema import InsightResponse


class HealthCheckResponse(BaseModel):
    """Status operacional do serviço."""
    status: str = Field(default="healthy")
    version: str = Field(default="1.0.0")
    engine: str = Field(default="Hybrid BM25 + ChromaDB + FlashRank + Gemini")


class QueryResponse(BaseModel):
    """Resposta padronizada da API contendo a inteligência gerada."""
    success: bool = Field(default=True)
    latency_seconds: float = Field(..., description="Tempo total de execução do pipeline.")
    data: InsightResponse