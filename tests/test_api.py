from unittest.mock import patch
import pytest
from fastapi.testclient import TestClient
from src.api.main import app
from src.schemas.rag_schema import CitationEvidence, InsightResponse


def test_health_check_endpoint():
    """Verifica se o endpoint /health responde status 200 e estrutura esperada."""
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data.get("status") == "healthy"
        assert "version" in data


def test_query_endpoint_validation_error():
    """Garante retorno HTTP 422 caso o payload enviado seja vazio/inválido."""
    with TestClient(app) as client:
        response = client.post("/api/v1/query", json={})
        assert response.status_code == 422


def test_query_endpoint_mock_execution():
    """Valida se o endpoint /api/v1/query processa a requisição com sucesso e retorna o contrato correto."""
    mocked_response = InsightResponse(
        query="Problemas com atraso na entrega?",
        executive_summary="Identificadas falhas pontuais no cumprimento de prazos logísticos.",
        sentiment_trend="Crítico/Negativo",
        key_root_causes=["Atraso no transporte", "Falta de informação no rastreio"],
        actionable_recommendations=["Auditar centros de distribuição críticos"],
        citations=[
            CitationEvidence(
                review_id="rev_teste_001",
                review_score=1,
                delivery_delay_days=6.0,
                excerpt="A entrega atrasou mais de uma semana.",
            )
        ],
        groundedness_score=1.0,
    )

    with patch("src.rag.pipeline.OlistRAGPipeline.generate_insight", return_value=mocked_response):
        with TestClient(app) as client:
            payload = {
                "query": "Problemas com atraso na entrega?",
                "retrieval_k": 5,
                "rerank_n": 2,
            }
            response = client.post("/api/v1/query", json=payload)
            assert response.status_code == 200
            body = response.json()
            assert body.get("success") is True
            assert "data" in body
            assert "latency_seconds" in body
            assert isinstance(body["data"]["key_root_causes"], list)
            assert isinstance(body["data"]["citations"], list)
            assert body["data"]["citations"][0]["review_id"] == "rev_teste_001"