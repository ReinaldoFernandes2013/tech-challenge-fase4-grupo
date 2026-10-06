from unittest.mock import MagicMock, patch
import pandas as pd
import pytest

from src.rag.pipeline import OlistRAGPipeline
from src.schemas.rag_schema import InsightResponse


@pytest.fixture
def mock_df():
    return pd.DataFrame(
        {
            "review_id": ["rev_1", "rev_2"],
            "order_id": ["ord_1", "ord_2"],
            "clean_comment": ["entrega atrasou muito", "produto quebrado"],
            "review_score": [1, 1],
            "delivery_delay_days": [10.0, 0.0],
        }
    )


class TestOlistRAGPipelineMocked:

    @patch("src.rag.pipeline.CrossEncoderReranker")
    @patch("src.rag.pipeline.HybridSearchEngine")
    def test_pipeline_out_of_scope_insufficient_evidence(
        self, mock_hybrid_cls, mock_reranker_cls, mock_df
    ):
        """Valida formalmente se perguntas fora do escopo acionam a abstenção segura (Fallback 1)
        quando o score das evidências for menor que o limiar empírico de 0.10.
        """
        # 1. Instância do pipeline
        pipeline = OlistRAGPipeline(mock_df)

        # 2. Mock do reranker devolvendo score irrelevante / baixo (< 0.10)
        pipeline.reranker.rerank.return_value = [
            {
                "review_id": "rev_irrelevante",
                "text": "comentário não relacionado",
                "rerank_score": 0.04,  # abaixo do threshold de 0.10
                "review_score": 1,
                "delivery_delay_days": 0.0,
            }
        ]

        # Evita busca em cache durante o teste
        pipeline.cache.get = MagicMock(return_value=None)

        query = "Qual é a receita de bolo de cenoura com cobertura de chocolate?"
        response = pipeline.generate_insight(query=query)

        # Comprovações formais da abstenção segura
        assert isinstance(response, InsightResponse)
        assert response.query == query
        assert "Não encontramos evidências suficientes" in response.executive_summary
        assert len(response.citations) == 0, "O pipeline não deve gerar citações ao se abster."
        assert response.groundedness_score == 0.0, "O groundedness deve ser 0.0 na abstenção."
        assert response.sentiment_trend == "Neutro", "O sentimento na recusa deve ser neutro."

    @patch("src.rag.pipeline.CrossEncoderReranker")
    @patch("src.rag.pipeline.HybridSearchEngine")
    def test_pipeline_valid_evidence_generation(
        self, mock_hybrid_cls, mock_reranker_cls, mock_df
    ):
        """Valida o fluxo com evidências acima do limiar sem invocar API real de LLM."""
        pipeline = OlistRAGPipeline(mock_df)

        pipeline.reranker.rerank.return_value = [
            {
                "review_id": "rev_100",
                "text": "O prazo de entrega atrasou mais de 2 semanas.",
                "rerank_score": 0.85,  # acima do threshold
                "review_score": 1,
                "delivery_delay_days": 14.0,
            }
        ]

        pipeline.cache.get = MagicMock(return_value=None)

        query = "Quais são os principais motivos de atraso na entrega?"
        response = pipeline.generate_insight(query=query, llm=None)

        assert isinstance(response, InsightResponse)
        assert response.query == query
        assert len(response.citations) > 0
        assert response.citations[0].review_id == "rev_100"
        assert response.citations[0].review_score == 1

    @patch("src.rag.pipeline.CrossEncoderReranker")
    @patch("src.rag.pipeline.HybridSearchEngine")
    def test_pipeline_llm_failure_positive_sentiment_resilience(
        self, mock_hybrid_cls, mock_reranker_cls, mock_df
    ):
        """Valida se o fallback determinístico sob falha da API de LLM reflete adequadamente comentários positivos."""
        pipeline = OlistRAGPipeline(mock_df)

        # Simula recuperação de avaliações 5 estrelas
        pipeline.reranker.rerank.return_value = [
            {
                "review_id": "rev_elogio_1",
                "text": "Entrega super rápida e produto impecável!",
                "rerank_score": 0.92,
                "review_score": 5,
                "delivery_delay_days": 0.0,
            }
        ]

        pipeline.cache.get = MagicMock(return_value=None)

        # Simula explicitamente a falha de conexão / API do LLM
        mock_failing_llm = MagicMock()
        mock_failing_llm.with_structured_output.side_effect = RuntimeError("API Gemini Indisponível / Quota Excedida")

        query = "Quais são os elogios sobre a entrega rápida?"
        response = pipeline.generate_insight(query=query, llm=mock_failing_llm)

        assert isinstance(response, InsightResponse)
        assert response.sentiment_trend == "Positivo", "O sentimento deve ser Positivo para avaliações favoráveis."
        assert len(response.citations) == 1
        assert response.citations[0].review_id == "rev_elogio_1"
        assert "eficiência no cumprimento de prazos" in response.executive_summary