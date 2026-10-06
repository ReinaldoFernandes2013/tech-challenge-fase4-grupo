"""
Módulo de Métricas Determinísticas e Estatísticas para Avaliação de RAG e Recuperação de Informação.
"""
from typing import List, Sequence, Set
import numpy as np


def precision_at_k(retrieved_ids: Sequence[str], relevant_ids: Set[str], k: int) -> float:
    """Calcula a métrica Precision@K para a recuperação léxica e vetorial."""
    if not retrieved_ids or not relevant_ids or k <= 0:
        return 0.0

    top_k = retrieved_ids[:k]
    hits = sum(1 for doc_id in top_k if doc_id in relevant_ids)
    return float(hits / min(k, len(top_k)))


def recall_at_k(retrieved_ids: Sequence[str], relevant_ids: Set[str], k: int) -> float:
    """Calcula a métrica Recall@K medindo a cobertura das evidências relevantes."""
    if not retrieved_ids or not relevant_ids or k <= 0:
        return 0.0

    top_k = retrieved_ids[:k]
    hits = sum(1 for doc_id in top_k if doc_id in relevant_ids)
    return float(hits / len(relevant_ids))


def mean_reciprocal_rank(ranked_results: List[Sequence[str]], ground_truth_ids: List[Set[str]]) -> float:
    """Calcula o MRR (Mean Reciprocal Rank) sobre uma bateria de consultas."""
    if not ranked_results or not ground_truth_ids or len(ranked_results) != len(ground_truth_ids):
        return 0.0

    reciprocal_ranks = []
    for retrieved, relevant in zip(ranked_results, ground_truth_ids):
        rr = 0.0
        for rank, doc_id in enumerate(retrieved, start=1):
            if doc_id in relevant:
                rr = 1.0 / rank
                break
        reciprocal_ranks.append(rr)

    return float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0


def calculate_triad_statistics(scores: List[float]) -> dict:
    """Calcula agregados estatísticos (média, mediana, desvio padrão) para métricas de RAG."""
    if not scores:
        return {"mean": 0.0, "median": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}

    arr = np.asarray(scores, dtype=np.float64)
    return {
        "mean": round(float(np.mean(arr)), 4),
        "median": round(float(np.median(arr)), 4),
        "std": round(float(np.std(arr)), 4),
        "min": round(float(np.min(arr)), 4),
        "max": round(float(np.max(arr)), 4),
    }