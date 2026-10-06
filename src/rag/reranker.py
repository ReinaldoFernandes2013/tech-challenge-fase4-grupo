from pathlib import Path
from typing import Any, Dict, List
from flashrank import Ranker, RerankRequest
from src.core.logging import logger

CACHE_DIR = str(Path("./data/models").resolve())


class CrossEncoderReranker:
    """Re-ranker local ultrarrápido baseado em FlashRank (Cross-Encoder leve)."""

    def __init__(self, model_name: str = "ms-marco-MiniLM-L-12-v2"):
        logger.info(f"A inicializar o modelo de re-ranking: {model_name}")
        self.ranker = Ranker(model_name=model_name, cache_dir=CACHE_DIR)

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_n: int = 5,
    ) -> List[Dict[str, Any]]:
        """Reclassifica os candidatos recuperados pelo motor híbrido."""
        if not candidates:
            return []

        # Formato esperado pelo FlashRank com garantia de string não vazia
        passages = []
        for i, doc in enumerate(candidates):
            text_content = str(doc.get("text") or "").strip()
            if not text_content:
                text_content = "Comentário sem texto explícito."
            passages.append({"id": str(i), "text": text_content, "meta": doc})

        rerank_request = RerankRequest(query=query, passages=passages)
        ranked_passages = self.ranker.rerank(rerank_request)

        results = []
        for item in ranked_passages[:top_n]:
            doc_data = item["meta"].copy()
            doc_data["rerank_score"] = float(item["score"])
            results.append(doc_data)

        logger.info(
            f"Re-ranking concluído: {len(candidates)} candidatos reduzidos para os top-{len(results)} mais relevantes."
        )
        return results