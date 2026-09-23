from typing import Optional
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """Requisição de inferência RAG para Voice of Customer."""
    query: str = Field(
        ...,
        min_length=3,
        description="Pergunta executiva ou dor de negócio a ser investigada.",
        examples=["Quais os principais problemas relatados sobre produtos com defeito?"]
    )
    retrieval_k: Optional[int] = Field(
        default=15,
        ge=1,
        le=50,
        description="Número de documentos candidatos a buscar no motor híbrido."
    )
    rerank_n: Optional[int] = Field(
        default=5,
        ge=1,
        le=15,
        description="Número final de evidências filtradas após o Cross-Encoder."
    )