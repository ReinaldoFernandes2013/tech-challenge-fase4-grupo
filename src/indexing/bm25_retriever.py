from typing import List, Dict, Any
import pandas as pd
from rank_bm25 import BM25Okapi
from src.core.logging import logger
from src.data.preprocessor import TextCleanerPTBR


class BM25RetrieverOlist:
    """Recuperador léxico baseado em BM25 ajustado para avaliações em PT-BR."""

    def __init__(self, df: pd.DataFrame):
        self.df = df.reset_index(drop=True)
        self._build_index()

    def _tokenize(self, text: str) -> List[str]:
        cleaned = TextCleanerPTBR.clean(text)
        return cleaned.split()

    def _build_index(self) -> None:
        logger.info("A construir índice BM25 sobre os comentários...")
        tokenized_corpus = [
            self._tokenize(doc) for doc in self.df["clean_comment"]
        ]
        self.bm25 = BM25Okapi(tokenized_corpus)
        logger.info(f"Índice BM25 construído com {len(tokenized_corpus):,} documentos.")

    def search(self, query: str, top_k: int = 15) -> List[Dict[str, Any]]:
        """Recupera os documentos mais relevantes com base em pontuação BM25."""
        tokenized_query = self._tokenize(query)
        if not tokenized_query:
            return []

        scores = self.bm25.get_scores(tokenized_query)
        top_indices = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )[:top_k]

        results = []
        for idx in top_indices:
            row = self.df.iloc[idx]
            results.append({
                "review_id": str(row.get("review_id", "")),
                "order_id": str(row.get("order_id", "")),
                "review_score": int(row.get("review_score", 0)),
                "text": str(row.get("full_comment", "")),
                "delivery_delay_days": row.get("delivery_delay_days"),
                "is_delayed": row.get("is_delayed"),
                "score_bm25": float(scores[idx]),
                "retriever_type": "bm25"
            })
        return results