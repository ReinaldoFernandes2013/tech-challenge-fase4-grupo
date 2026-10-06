from unittest.mock import MagicMock, patch
import pandas as pd
import pytest

from src.indexing.bm25_retriever import BM25RetrieverOlist
from src.indexing.hybrid_indexer import HybridSearchEngine


@pytest.fixture
def sample_reviews_df():
    return pd.DataFrame(
        {
            "review_id": ["rev_1", "rev_2", "rev_3"],
            "order_id": ["ord_1", "ord_2", "ord_3"],
            "clean_comment": [
                "produto excelente entrega muito rápida recomendo",
                "produto veio com defeito e quebrado péssimo atendimento",
                "demorou para chegar mas o produto é bom",
            ],
            "review_score": [5, 1, 3],
        }
    )


class TestBM25Retriever:

    def test_bm25_search_returns_relevant_document(self, sample_reviews_df):
        retriever = BM25RetrieverOlist(sample_reviews_df)
        results = retriever.search("entrega rápida produto", top_k=2)

        assert len(results) > 0
        assert results[0]["review_id"] == "rev_1"

    def test_bm25_search_empty_query_returns_empty(self, sample_reviews_df):
        retriever = BM25RetrieverOlist(sample_reviews_df)
        results = retriever.search("", top_k=2)
        assert results == []

    def test_bm25_search_no_match_returns_empty(self, sample_reviews_df):
        """Valida que termos inexistentes não retornam documentos de pontuação zero."""
        retriever = BM25RetrieverOlist(sample_reviews_df)
        results = retriever.search("palavrainexistentenocorpusxyz", top_k=2)
        assert results == []


class TestHybridSearchEngineMocked:

    @patch("src.indexing.hybrid_indexer.OlistVectorStore")
    @patch("src.indexing.hybrid_indexer.BM25RetrieverOlist")
    def test_hybrid_search_rrf_fusion(self, mock_bm25_cls, mock_vector_cls, sample_reviews_df):
        # Mock do BM25
        mock_bm25_instance = MagicMock()
        mock_bm25_instance.search.return_value = [
            {"review_id": "rev_1", "clean_comment": "produto excelente", "score": 12.5},
            {"review_id": "rev_2", "clean_comment": "produto com defeito", "score": 8.0},
        ]
        mock_bm25_cls.return_value = mock_bm25_instance

        # Mock do Vector Store (sem ChromaDB e sem carregar modelo pesado)
        mock_vector_instance = MagicMock()
        mock_vector_instance.similarity_search.return_value = [
            {"review_id": "rev_2", "clean_comment": "produto com defeito", "score": 0.88},
            {"review_id": "rev_1", "clean_comment": "produto excelente", "score": 0.75},
        ]
        mock_vector_cls.return_value = mock_vector_instance

        engine = HybridSearchEngine(sample_reviews_df)
        results = engine.search("produto", top_k=2, rrf_k=60)

        assert len(results) <= 2
        returned_ids = [doc["review_id"] for doc in results]
        assert "rev_1" in returned_ids
        assert "rev_2" in returned_ids
        # Exigência rigorosa de presença do score consolidado de RRF
        assert "rrf_score" in results[0]