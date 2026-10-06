import pytest
from pydantic import ValidationError
from src.schemas.rag_schema import CitationEvidence, InsightResponse, LLMOutputSchema
from src.schemas.request import QueryRequest


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
    """Garante que review_score não aceita tipos inválidos nem valores fora de [1, 5]."""
    # Tipo inválido
    with pytest.raises(ValidationError):
        CitationEvidence(
            review_id="rev_999",
            review_score="muito_ruim",
            excerpt="Texto de teste.",
        )

    # Valor acima do permitido (> 5)
    with pytest.raises(ValidationError):
        CitationEvidence(
            review_id="rev_999",
            review_score=6,
            excerpt="Texto de teste.",
        )

    # Valor abaixo do permitido (< 1)
    with pytest.raises(ValidationError):
        CitationEvidence(
            review_id="rev_999",
            review_score=0,
            excerpt="Texto de teste.",
        )


def test_insight_response_structure_and_bounds():
    """Valida a integridade do payload consolidado e as restrições de limite."""
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

    # Groundedness acima do limite máximo (> 1.0) deve falhar
    payload_invalid = payload.copy()
    payload_invalid["groundedness_score"] = 1.2
    with pytest.raises(ValidationError):
        InsightResponse(**payload_invalid)


def test_query_request_validation():
    """Valida as regras de fronteira de entrada da API REST."""
    # Consulta válida
    req = QueryRequest(query="Problemas de entrega")
    assert req.query == "Problemas de entrega"
    assert req.retrieval_k == 15

    # Consulta curta demais (< 3 caracteres)
    with pytest.raises(ValidationError):
        QueryRequest(query="oi")


def test_llm_output_schema():
    """Garante que a extração estruturada do LLM respeita o contrato base."""
    data = {
        "executive_summary": "Entrega rápida e produto conforme anúncio.",
        "sentiment_trend": "Positivo",
        "key_root_causes": ["Agilidade logística"],
        "actionable_recommendations": ["Manter parcerias com correios"],
        "cited_review_ids": ["rev_100", "rev_101"],
    }
    output = LLMOutputSchema(**data)
    assert len(output.cited_review_ids) == 2
    assert output.sentiment_trend == "Positivo"