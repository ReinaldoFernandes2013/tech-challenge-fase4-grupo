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
        """Valida se perguntas fora do escopo acionam a recusa graciosa (Fallback 1)

        quando o score das evidências for menor que o limiar de 0.10.
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

        query = "Qual é o melhor smartphone para comprar hoje?"
        response = pipeline.generate_insight(query=query)

        assert isinstance(response, InsightResponse)
        assert response.query == query
        assert (
            "Não encontramos evidências suficientes" in response.executive_summary
        )

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
        # Chama sem LLM para acionar a síntese determinística estruturada auditada
        response = pipeline.generate_insight(query=query, llm=None)

        assert isinstance(response, InsightResponse)
        assert response.query == query
        assert len(response.citations) > 0
        assert response.citations[0].review_id == "rev_100"