import pytest
from pydantic import ValidationError
from src.schemas.rag_schema import CitationEvidence, InsightResponse


def test_citation_evidence_valid():
    """Valida a instanciação correta de uma evidência com dados íntegros."""
    citation = CitationEvidence(
        review_id="rev_12345",
        review_score=1,
        delivery_delay_days=5.5,
        excerpt="Produto atrasou e veio com defeito.",
    )
    assert citation.review_id == "rev_12345"
    assert citation.review_score == 1
    assert citation.delivery_delay_days == 5.5
    assert "atrasou" in citation.excerpt


def test_citation_evidence_invalid_score():
    """Garante que review_score não aceita valores fora do tipo inteiro."""
    with pytest.raises(ValidationError):
        CitationEvidence(
            review_id="rev_999",
            review_score="muito_ruim",  # Tipo inválido intencional
            excerpt="Texto de teste.",
        )


def test_insight_response_structure():
    """Valida a integridade do payload consolidado de resposta executiva."""
    payload = {
        "query": "Qual o motivo dos atrasos?",
        "executive_summary": "Concentração de falhas na última milha de transporte.",
        "sentiment_trend": "Crítico/Negativo",
        "key_root_causes": ["Logística de terceiros", "Atraso no despacho"],
        "actionable_recommendations": ["Revisar SLA de transportadoras parceiras"],
        "citations": [
            {
                "review_id": "rev_001",
                "review_score": 1,
                "delivery_delay_days": 12.0,
                "excerpt": "Não recebi a encomenda no prazo previsto.",
            }
        ],
        "groundedness_score": 1.0,
    }
    insight = InsightResponse(**payload)
    assert insight.sentiment_trend == "Crítico/Negativo"
    assert len(insight.key_root_causes) == 2
    assert len(insight.citations) == 1
    assert insight.groundedness_score == 1.0