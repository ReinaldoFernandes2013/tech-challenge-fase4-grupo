import os

# 1. Blindagem de concorrência antes de importar qualquer biblioteca de tensores
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from pathlib import Path
from typing import Any, Dict, List
import pandas as pd
from src.core.config import settings
from src.core.logging import logger
from src.indexing.bm25_retriever import BM25RetrieverOlist
from src.indexing.vector_store import OlistVectorStore
from src.rag.reranker import CrossEncoderReranker


class HybridSearchEngine:
    """Motor de busca híbrida com fusão RRF (BM25 + Dense Vectors)."""

    def __init__(self, df: pd.DataFrame):
        self.bm25_retriever = BM25RetrieverOlist(df)
        self.vector_store = OlistVectorStore()

    def search(
        self,
        query: str,
        top_k: int = 10,
        rrf_k: int = 60,
    ) -> List[Dict[str, Any]]:
        """Executa recuperação híbrida e combina rankings via RRF."""
        # 1. Recuperação em paralelo/conjunta
        bm25_results = self.bm25_retriever.search(query, top_k=settings.BM25_TOP_K)
        vector_results = self.vector_store.similarity_search(
            query, top_k=settings.VECTOR_TOP_K
        )

        # 2. Fusão RRF: Score = 1 / (k + rank)
        rrf_scores: Dict[str, float] = {}
        doc_map: Dict[str, Dict[str, Any]] = {}

        for rank, item in enumerate(bm25_results):
            rid = item["review_id"]
            rrf_scores[rid] = rrf_scores.get(rid, 0.0) + (1.0 / (rrf_k + rank + 1))
            doc_map[rid] = item

        for rank, item in enumerate(vector_results):
            rid = item["review_id"]
            rrf_scores[rid] = rrf_scores.get(rid, 0.0) + (1.0 / (rrf_k + rank + 1))
            if rid not in doc_map:
                doc_map[rid] = item

        # 3. Ordenação decrescente pelo score consolidado
        sorted_ids = sorted(
            rrf_scores.keys(), key=lambda rid: rrf_scores[rid], reverse=True
        )[:top_k]

        final_docs = []
        for rid in sorted_ids:
            entry = doc_map[rid].copy()
            entry["rrf_score"] = rrf_scores[rid]
            final_docs.append(entry)

        logger.info(
            f"Recuperação híbrida concluída. Retornados {len(final_docs)} documentos consolidados."
        )
        return final_docs



