from typing import List, Optional
from pydantic import BaseModel, Field


class CitationEvidence(BaseModel):
    """Evidência documental que fundamenta a resposta."""
    review_id: str = Field(description="ID único da avaliação citada.")
    review_score: int = Field(description="Classificação por estrelas (1 a 5).")
    delivery_delay_days: Optional[float] = Field(
        default=0.0,
        description="Dias de atraso na entrega associados a esta avaliação."
    )
    excerpt: str = Field(description="Trecho literal relevante do comentário.")


class InsightResponse(BaseModel):
    """Contrato de resposta executiva de inteligência sobre o cliente."""
    query: str = Field(description="Pergunta original do analista.")
    executive_summary: str = Field(
        description="Resumo executivo direto respondendo à dor ou dúvida levantada."
    )
    sentiment_trend: str = Field(
        description="Tendência geral do sentimento: 'Crítico/Negativo', 'Neutro' ou 'Positivo'."
    )
    key_root_causes: List[str] = Field(
        description="Causas raiz identificadas nos comentários (ex: Logística, Qualidade, SAC)."
    )
    actionable_recommendations: List[str] = Field(
        description="Ações práticas recomendadas para a operação e vendedores da Olist."
    )
    citations: List[CitationEvidence] = Field(
        description="Lista de evidências reais extraídas diretamente do dataset que justificam a resposta."
    )
    groundedness_score: float = Field(
        default=1.0,
        description="Score interno de ancoragem (0 a 1), baseado na fidelidade às evidências."
    )