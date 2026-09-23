from typing import List, Optional
from pydantic import BaseModel, Field


class RAGTriadMetric(BaseModel):
    """Métricas pontuais para uma avaliação individual da Tríade."""
    context_relevance: float = Field(
        ..., ge=0.0, le=1.0,
        description="Grau de relevância dos documentos recuperados em relação à pergunta (0 a 1)."
    )
    groundedness: float = Field(
        ..., ge=0.0, le=1.0,
        description="Grau em que a resposta é suportada estritamente pelo contexto recuperado (0 a 1)."
    )
    answer_relevance: float = Field(
        ..., ge=0.0, le=1.0,
        description="Grau de alinhamento e completude da resposta gerada em relação à pergunta (0 a 1)."
    )
    justification: str = Field(
        ...,
        description="Racional analítico detalhando as pontuações atribuídas."
    )


class SingleEvaluationResult(BaseModel):
    """Resultado consolidado de inferência + avaliação."""
    query: str
    latency_seconds: float
    retrieved_docs_count: int
    triad: RAGTriadMetric


class BenchmarkReport(BaseModel):
    """Relatório final de benchmark para o comitê acadêmico."""
    total_evaluations: int
    mean_latency_seconds: float
    mean_context_relevance: float
    mean_groundedness: float
    mean_answer_relevance: float
    results: List[SingleEvaluationResult]