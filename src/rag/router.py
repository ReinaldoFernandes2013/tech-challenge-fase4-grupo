"""
Módulo Avançado de Roteamento Semântico, Cache Vetorial e Observabilidade MLOps.
"""
import time
import re
import unicodedata
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
from src.core.logging import logger


def normalize_text(text: str) -> str:
    """Remove acentos e converte para minúsculas para robustez nas buscas."""
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join([c for c in nfkd if not unicodedata.combining(c)]).lower()


class SemanticCache:
    """Cache semântico local baseado em embeddings e similaridade de cosseno."""

    def __init__(self, threshold: float = 0.90):
        self.threshold = threshold
        self.cache: list[Dict[str, Any]] = []

    def get(self, query_vector: np.ndarray) -> Optional[Dict[str, Any]]:
        if not self.cache or query_vector is None:
            return None

        query_vec = np.asarray(query_vector, dtype=np.float32)
        query_norm = np.linalg.norm(query_vec)
        if query_norm == 0:
            return None

        best_score = -1.0
        best_payload = None

        for entry in self.cache:
            cached_vec = np.asarray(entry["vector"], dtype=np.float32)
            cached_norm = np.linalg.norm(cached_vec)
            if cached_norm > 0:
                sim = float(np.dot(query_vec, cached_vec) / (query_norm * cached_norm))
                if sim > best_score:
                    best_score = sim
                    best_payload = entry["payload"]

        if best_score >= self.threshold:
            logger.info("Semantic Cache HIT: similaridade=%.4f", best_score)
            return best_payload

        return None

    def put(self, query_vector: np.ndarray, payload: Dict[str, Any]) -> None:
        self.cache.append({
            "vector": np.asarray(query_vector, dtype=np.float32),
            "payload": payload,
            "created_at": time.time(),
        })


class QueryAnalyzer:
    """Extração de intenções, filtros de metadados e estatísticas diretas."""

    UF_LIST = ["SP", "RJ", "MG", "RS", "PR", "SC", "BA", "PE", "CE", "DF", "GO", "ES"]

    @classmethod
    def extract_filters(cls, query: str) -> Dict[str, Any]:
        """Extrai UF e faixas de satisfação mencionadas na consulta com suporte a acentuação."""
        filters: Dict[str, Any] = {}
        normalized = normalize_text(query)
        tokens_upper = normalized.upper().split()

        for uf in cls.UF_LIST:
            if (
                uf in tokens_upper
                or f"EM {uf}" in normalized.upper()
                or f"DE {uf}" in normalized.upper()
                or f"NO {uf}" in normalized.upper()
                or f"NA {uf}" in normalized.upper()
            ):
                filters["customer_state"] = uf
                break

        if re.search(
            r"\b(pessim[oa]s?|ruim|ruins|pior|piores|critico|criticos|critica|criticas|nota 1|1 estrela)\b",
            normalized,
        ):
            filters["review_score"] = 1
        elif re.search(
            r"\b(excelente|excelentes|otim[oa]s?|5 estrelas|nota 5)\b",
            normalized,
        ):
            filters["review_score"] = 5

        return filters

    @classmethod
    def is_quantitative_query(cls, query: str) -> bool:
        """Verifica se a query exige agregação tabular ou causa-raiz qualitativa."""
        normalized = normalize_text(query)
        pattern = r"\b(quantos|quantas|total|media|mediana|porcentagem|estatistica|volume|quantidade)\b"
        return bool(re.search(pattern, normalized))

    @classmethod
    def compute_direct_metrics(cls, df: pd.DataFrame, filters: Dict[str, Any]) -> Dict[str, Any]:
        """Calcula métricas agregadas instantâneas a partir da base Parquet."""
        filtered_df = df.copy()
        if "customer_state" in filters:
            filtered_df = filtered_df[filtered_df["customer_state"] == filters["customer_state"]]
        if "review_score" in filters:
            filtered_df = filtered_df[filtered_df["review_score"] == filters["review_score"]]

        total = len(filtered_df)
        avg_score = float(filtered_df["review_score"].mean()) if total > 0 else 0.0

        return {
            "total_reviews": total,
            "avg_score": round(avg_score, 2),
            "state_filter": filters.get("customer_state", "Todos"),
            "score_filter": filters.get("review_score", "Todos"),
        }