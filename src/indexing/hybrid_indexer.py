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


if __name__ == "__main__":
    parquet_path = settings.DATA_PROCESSED_DIR / "olist_reviews_clean.parquet"
    if not parquet_path.exists():
        raise FileNotFoundError(
            f"Parquet não encontrado em: {parquet_path}. Execute o loader primeiro."
        )

    logger.info("A carregar base processada...")
    df_all = pd.read_parquet(parquet_path)

    # Amostra controlada para teste rápido e seguro
    sample_size = min(300, len(df_all))
    df_sample = df_all.sample(n=sample_size, random_state=42)
    logger.info(f"Amostra selecionada para validação: {sample_size} registos.")

    # 1. Indexação em lotes pequenos e controlados (50 em 50)
    vstore = OlistVectorStore()
    vstore.index_dataframe(df_sample, batch_size=50)

    # 2. Inicializar motor híbrido e re-ranker
    hybrid_engine = HybridSearchEngine(df_sample)
    reranker = CrossEncoderReranker()

    # 3. Teste de consulta
    query_teste = "O produto chegou com defeito e o atendimento não ajudou"
    logger.info(f"\n--- A testar recuperação para a pergunta: '{query_teste}' ---")

    candidatos = hybrid_engine.search(query_teste, top_k=10)
    top_evidencias = reranker.rerank(query_teste, candidatos, top_n=3)

    print("\n" + "=" * 60)
    print("RESULTADO DAS 3 MELHORES EVIDÊNCIAS APÓS RE-RANKING:")
    print("=" * 60)
    for idx, ev in enumerate(top_evidencias, 1):
        print(
            f"\n[{idx}] Review ID: {ev['review_id']} | Avaliação: {ev['review_score']} estrelas"
        )
        print(
            f"    RRF Score: {ev.get('rrf_score', 0):.4f} | Rerank Score: {ev.get('rerank_score', 0):.4f}"
        )
        print(f"    Texto: {ev['text']}")