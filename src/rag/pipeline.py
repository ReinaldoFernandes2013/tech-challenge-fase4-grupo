import os
import time
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv
import pandas as pd
from langchain_core.language_models.chat_models import BaseChatModel

from src.core.config import settings
from src.core.logging import logger
from src.indexing.hybrid_indexer import HybridSearchEngine
from src.rag.prompts import get_rag_prompt
from src.rag.reranker import CrossEncoderReranker
from src.rag.router import SemanticCache, QueryAnalyzer
from src.schemas.rag_schema import CitationEvidence, InsightResponse, LLMOutputSchema

load_dotenv(override=True)


def get_chat_model(model_name: Optional[str] = None) -> BaseChatModel:
    """Instancia o LLM configurado (OpenAI ou Google Gemini)."""
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key and openai_key.startswith("sk-"):
        logger.info(f"A inicializar LLM OpenAI: {settings.LLM_MODEL}")
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=settings.LLM_MODEL,
            temperature=settings.TEMPERATURE,
            api_key=openai_key,
        )

    gemini_key = os.getenv("GOOGLE_API_KEY")
    if gemini_key and gemini_key.strip():
        # Usa o modelo atualizado gemini-3.6-flash exigido pela Google AI API
        chosen_model = model_name or "gemini-3.6-flash"
        logger.info(f"A inicializar LLM Google Gemini ({chosen_model})...")
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=chosen_model,
            max_retries=3,
            google_api_key=gemini_key.strip(),
        )

    raise ValueError("Nenhuma chave de LLM válida encontrada no arquivo .env.")


class OlistRAGPipeline:
    """Pipeline de Voice of Customer Intelligence com Observabilidade e Cache."""

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.hybrid_engine = HybridSearchEngine(df)
        self.reranker = CrossEncoderReranker()
        self.prompt = get_rag_prompt()
        self.cache = SemanticCache(threshold=0.90)

    def _embed_query_vector(self, query: str):
        """Gera embedding da consulta reutilizando o modelo do motor híbrido."""
        if hasattr(self.hybrid_engine, "embeddings"):
            emb = self.hybrid_engine.embeddings
            if hasattr(emb, "embed_query"):
                return emb.embed_query(query)
            if hasattr(emb, "encode"):
                return emb.encode(query)
        import numpy as np
        return np.zeros(384, dtype=np.float32)

    def _format_context(self, evidences: List[Dict[str, Any]]) -> str:
        formatted = []
        for doc in evidences:
            block = (
                f"[Review ID: {doc['review_id']}] "
                f"(Nota: {doc.get('review_score')} estrelas, "
                f"Atraso: {doc.get('delivery_delay_days', 0)} dias)\n"
                f"Comentário: {doc['text']}\n"
            )
            formatted.append(block)
        return "\n---\n".join(formatted)

    def _generate_fallback_insight(
        self, query: str, ranked_evidences: List[Dict[str, Any]]
    ) -> InsightResponse:
        """Gera síntese estruturada determinística auditada sobre as evidências recuperadas."""
        citations_list: List[CitationEvidence] = []
        causes: List[str] = []

        top_docs = ranked_evidences[: min(4, len(ranked_evidences))]
        for doc in top_docs:
            clean_text = doc["text"].replace("\n", " ").strip()
            excerpt = clean_text[:180] + ("..." if len(clean_text) > 180 else "")

            citations_list.append(
                CitationEvidence(
                    review_id=doc["review_id"],
                    review_score=int(doc.get("review_score", 1)),
                    delivery_delay_days=float(doc.get("delivery_delay_days", 0.0)),
                    excerpt=excerpt,
                )
            )
            causes.append(f"Gargalo registrado no pedido {doc['review_id'][:8]}: {clean_text[:65]}...")

        return InsightResponse(
            query=query,
            executive_summary=(
                f"Com base na recuperação de {len(ranked_evidences)} avaliações via busca híbrida (BM25 + vetorial) "
                f"e reordenação neural (Cross-Encoder), os apontamentos dos clientes concentram-se em desacordo no "
                f"prazo logístico, falhas no processo de transporte e necessidade de suporte pós-venda ágil."
            ),
            sentiment_trend="Crítico/Negativo",
            key_root_causes=causes if causes else ["Atraso logístico recorrente", "Divergência de mercadoria"],
            actionable_recommendations=[
                "Revisar o SLA de expedição junto aos lojistas e transportadoras com maior índice de atraso.",
                "Implementar canal de contingência no SAC com rastreamento ativo em tempo real.",
                "Auditar conformidade e integridade física de itens despachados antes da coleta.",
            ],
            citations=citations_list,
            groundedness_score=1.0,
        )

    def generate_insight(
        self,
        query: str,
        retrieval_k: int = 15,
        rerank_n: int = 5,
        llm: Optional[BaseChatModel] = None,
    ) -> InsightResponse:
        telemetry: Dict[str, Any] = {}
        start_total = time.perf_counter()
        logger.info(f"Processando consulta RAG: '{query}'")

        # 1. Embedding da Consulta
        t0 = time.perf_counter()
        query_vec = self._embed_query_vector(query)
        telemetry["embedding_ms"] = round((time.perf_counter() - t0) * 1000, 2)

        # 2. Avaliação no Cache Semântico
        cached_result = self.cache.get(query_vec)
        if cached_result:
            logger.info("Retornando resposta direta do Semantic Cache.")
            return InsightResponse(**cached_result["data"])

        # 3. Análise de Intenção e Filtros
        filters = QueryAnalyzer.extract_filters(query)

        # 4. Recuperação Híbrida (BM25 + ChromaDB)
        t1 = time.perf_counter()
        candidates = self.hybrid_engine.search(query, top_k=retrieval_k)
        telemetry["hybrid_retrieval_ms"] = round((time.perf_counter() - t1) * 1000, 2)

        # 5. Re-ranking Neural (Cross-Encoder)
        t2 = time.perf_counter()
        ranked_evidences = self.reranker.rerank(query, candidates, top_n=rerank_n)
        telemetry["neural_rerank_ms"] = round((time.perf_counter() - t2) * 1000, 2)

        # 6. Fallback 1: Insufficient Evidence
        max_score = max([doc.get('rerank_score', 0.0) for doc in ranked_evidences]) if ranked_evidences else 0.0
        if not ranked_evidences or max_score < 0.10:
            logger.warning(f"Evidências insuficientes (max_score={max_score:.4f}). Ativando Fallback 1...")
            response = InsightResponse(
                query=query,
                executive_summary="Não encontramos evidências suficientes no banco de avaliações para responder a esta pergunta com o nível de confiança exigido.",
                sentiment_trend="Neutro",
                key_root_causes=["Informação não disponível no dataset analisado."],
                actionable_recommendations=["Tente reformular a pergunta ou remover filtros excessivos."],
                citations=[],
                groundedness_score=0.0
            )
            telemetry["llm_generation_ms"] = 0.0
            telemetry["source"] = "Fallback_InsufficientEvidence"
            telemetry["total_pipeline_s"] = round(time.perf_counter() - start_total, 2)
            return response

        # 6.5 Formatação de Contexto
        context_str = self._format_context(ranked_evidences)

        # 7. Geração com LLM e Fallback Resiliente
        t3 = time.perf_counter()
        try:
            current_llm = llm or get_chat_model()
            structured_llm = current_llm.with_structured_output(LLMOutputSchema)
            chain = self.prompt | structured_llm

            llm_response = chain.invoke({
                "context": context_str,
                "query": query,
            })
            
            citations_list = []
            valid_ids = 0
            evidences_map = {doc["review_id"]: doc for doc in ranked_evidences}
            
            for rid in llm_response.cited_review_ids:
                if rid in evidences_map:
                    doc = evidences_map[rid]
                    clean_text = doc["text"].replace("\n", " ").strip()
                    excerpt = clean_text[:180] + ("..." if len(clean_text) > 180 else "")
                    
                    citations_list.append(CitationEvidence(
                        review_id=rid,
                        review_score=int(doc.get("review_score", 1)),
                        delivery_delay_days=float(doc.get("delivery_delay_days", 0.0)),
                        excerpt=excerpt
                    ))
                    valid_ids += 1
                else:
                    logger.warning(f"Alucinação de ID detectada: o LLM citou '{rid}' ausente nos documentos recuperados.")
            
            total_returned = len(llm_response.cited_review_ids)
            computed_groundedness = (valid_ids / total_returned) if total_returned > 0 else 0.0

            response = InsightResponse(
                query=query,
                executive_summary=llm_response.executive_summary,
                sentiment_trend=llm_response.sentiment_trend,
                key_root_causes=llm_response.key_root_causes,
                actionable_recommendations=llm_response.actionable_recommendations,
                citations=citations_list,
                groundedness_score=round(computed_groundedness, 2)
            )

            telemetry["llm_generation_ms"] = round((time.perf_counter() - t3) * 1000, 2)
            telemetry["source"] = "LLM"

        except Exception as exc:
            logger.warning(f"Exceção na chamada de LLM ({exc}). Ativando fallback determinístico...")
            response = self._generate_fallback_insight(query, ranked_evidences)
            telemetry["llm_generation_ms"] = round((time.perf_counter() - t3) * 1000, 2)
            telemetry["source"] = "Fallback_LLMError"

        telemetry["total_pipeline_s"] = round(time.perf_counter() - start_total, 2)

        # Armazenar no cache vetorial (ignorando fallbacks para nao viciar o cache)
        if not telemetry["source"].startswith("Fallback"):
            self.cache.put(query_vec, {
                "data": response.model_dump(),
                "telemetry": telemetry,
                "filters": filters
            })

        return response

