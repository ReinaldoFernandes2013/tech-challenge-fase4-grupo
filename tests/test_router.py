"""
Testes unitários rigorosos para o Router, Semantic Cache e Query Analyzer.
"""
import pytest
import numpy as np
import pandas as pd
from src.rag.router import SemanticCache, QueryAnalyzer


@pytest.fixture
def sample_dataframe():
    """Gera massa de dados sintética simulando o dataset de reviews."""
    return pd.DataFrame({
        "review_id": ["r1", "r2", "r3", "r4"],
        "customer_state": ["SP", "SP", "RJ", "MG"],
        "review_score": [1, 2, 5, 1],
        "text": ["Péssimo atraso", "Demorou", "Excelente entrega", "Não recebi"],
    })


def test_semantic_cache_hit_and_miss():
    """Valida a similaridade de cosseno e recuperação no cache."""
    cache = SemanticCache(threshold=0.90)
    base_vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    payload = {"executive_summary": "Problema logístico em SP"}

    assert cache.get(base_vec) is None
    cache.put(base_vec, payload)

    similar_vec = np.array([0.98, 0.02, 0.0], dtype=np.float32)
    hit_result = cache.get(similar_vec)
    assert hit_result is not None
    assert hit_result["executive_summary"] == "Problema logístico em SP"

    different_vec = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    assert cache.get(different_vec) is None


def test_semantic_cache_zero_vector():
    """Garante robustez caso receba vetor nulo."""
    cache = SemanticCache()
    zero_vec = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    assert cache.get(zero_vec) is None


@pytest.mark.parametrize("query,expected_state,expected_score", [
    ("Quais os principais problemas de entrega em SP?", "SP", None),
    ("Extravio de mercadorias no RJ com nota 1", "RJ", 1),
    ("Relatos péssimos sobre pedidos de MG", "MG", 1),
    ("Produtos excelentes com 5 estrelas em Curitiba", None, 5),
    ("Atrasos gerais no pós-venda", None, None),
])
def test_query_analyzer_extract_filters(query, expected_state, expected_score):
    """Testa a extração precisa de metadados geográficos e de sentimento."""
    filters = QueryAnalyzer.extract_filters(query)
    assert filters.get("customer_state") == expected_state
    assert filters.get("review_score") == expected_score


@pytest.mark.parametrize("query,expected_is_quant", [
    ("Qual a média de notas em SP?", True),
    ("Quantos pedidos tiveram atraso?", True),
    ("Qual o volume total de reclamações?", True),
    ("Quais as causas de insatisfação do cliente?", False),
    ("Explique por que os produtos chegam avariados", False),
])
def test_query_analyzer_is_quantitative(query, expected_is_quant):
    """Valida a separação entre queries quantitativas e semânticas."""
    assert QueryAnalyzer.is_quantitative_query(query) == expected_is_quant


def test_compute_direct_metrics(sample_dataframe):
    """Garante a integridade do cálculo de métricas tabulares diretas."""
    filters = {"customer_state": "SP"}
    metrics = QueryAnalyzer.compute_direct_metrics(sample_dataframe, filters)
    assert metrics["total_reviews"] == 2
    assert metrics["avg_score"] == 1.5

    filters_score = {"review_score": 1}
    metrics_score = QueryAnalyzer.compute_direct_metrics(sample_dataframe, filters_score)
    assert metrics_score["total_reviews"] == 2
    assert metrics_score["avg_score"] == 1.0