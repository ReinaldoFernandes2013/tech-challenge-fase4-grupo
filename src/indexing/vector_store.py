from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from src.core.config import settings
from src.core.logging import logger
from src.indexing.embeddings import get_embedding_function


class OlistVectorStore:
    """Gestor do índice vetorial ChromaDB persistente."""

    def __init__(self, persist_dir: Optional[Path] = None):
        self.persist_dir = str(persist_dir or settings.VECTOR_DB_DIR)
        self.embedding_fn = get_embedding_function()
        self.collection_name = settings.COLLECTION_NAME

    def get_store(self) -> Chroma:
        """Obtém a instância da vector store local."""
        return Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embedding_fn,
            persist_directory=self.persist_dir,
        )

    def index_dataframe(self, df: pd.DataFrame, batch_size: int = 1000) -> None:
        """Cria e persiste embeddings a partir do DataFrame processado."""
        logger.info(f"A preparar documentos para indexação vetorial ({len(df):,} registos)...")

        documents: List[Document] = []
        for _, row in df.iterrows():
            metadata = {
                "review_id": str(row.get("review_id", "")),
                "order_id": str(row.get("order_id", "")),
                "review_score": int(row.get("review_score", 0)),
                "delivery_delay_days": float(row.get("delivery_delay_days")) if pd.notnull(row.get("delivery_delay_days")) else 0.0,
                "is_delayed": bool(row.get("is_delayed")) if pd.notnull(row.get("is_delayed")) else False,
            }
            # O texto a ser vetorizado é o comentário limpo, preservando o original nos metadados
            doc = Document(
                page_content=str(row.get("clean_comment", "")),
                metadata=metadata,
            )
            documents.append(doc)

        store = self.get_store()
        total_batches = (len(documents) + batch_size - 1) // batch_size
        logger.info(f"A gravar no ChromaDB em {total_batches} lotes...")

        for i in range(0, len(documents), batch_size):
            batch = documents[i : i + batch_size]
            ids = [doc.metadata["review_id"] for doc in batch]
            store.add_documents(batch, ids=ids)
            logger.info(f"Lote {i // batch_size + 1}/{total_batches} concluído.")

        logger.info("Indexação vetorial finalizada com sucesso.")

    def similarity_search(self, query: str, top_k: int = 15) -> List[Dict[str, Any]]:
        """Executa busca semântica por cosseno."""
        store = self.get_store()
        docs = store.similarity_search_with_score(query, k=top_k)

        results = []
        for doc, score in docs:
            results.append({
                "review_id": doc.metadata.get("review_id", ""),
                "order_id": doc.metadata.get("order_id", ""),
                "review_score": doc.metadata.get("review_score", 0),
                "text": doc.page_content,
                "delivery_delay_days": doc.metadata.get("delivery_delay_days"),
                "is_delayed": doc.metadata.get("is_delayed"),
                "score_vector": float(score),
                "retriever_type": "vector",
            })
        return results


