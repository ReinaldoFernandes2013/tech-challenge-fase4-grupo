import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd

from src.core.config import settings
from src.core.logging import logger
from src.rag.pipeline import OlistRAGPipeline
from src.schemas.request import QueryRequest
from src.schemas.response import HealthCheckResponse, QueryResponse

# Estado global do pipeline em memória
state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Ciclo de vida: carrega o dataset e inicializa o pipeline uma única vez."""
    parquet_path = settings.DATA_PROCESSED_DIR / "olist_reviews_clean.parquet"
    if not parquet_path.exists():
        logger.error(f"Base de dados não encontrada em: {parquet_path}")
        raise RuntimeError(f"Base Parquet ausente: {parquet_path}")

    logger.info("A carregar dataset e a instanciar o pipeline Olist RAG...")
    df_all = pd.read_parquet(parquet_path)
    # Amostra de validação para inicialização rápida e estável
    df_sample = df_all.sample(n=min(500, len(df_all)), random_state=42)

    state["pipeline"] = OlistRAGPipeline(df_sample)
    logger.info("Pipeline RAG inicializado com sucesso e pronto para receber requisições.")
    yield
    state.clear()


app = FastAPI(
    title="Olist Voice of Customer - RAG API",
    description="API de Recuperação Híbrida e Inteligência Estruturada para E-commerce",
    version="1.0.0",
    lifespan=lifespan,
)

# Configuração de CORS para permitir consumo pelo Streamlit / Front-end
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get(
    "/health",
    response_model=HealthCheckResponse,
    status_code=status.HTTP_200_OK,
    tags=["Observabilidade"],
)
async def health_check():
    """Retorna a saúde do serviço e modelos alocados."""
    return HealthCheckResponse()


@app.post(
    "/api/v1/query",
    response_model=QueryResponse,
    status_code=status.HTTP_200_OK,
    tags=["Inteligência RAG"],
)
async def query_pipeline(request: QueryRequest):
    """Executa a busca híbrida, re-ranking e extrai insights executivos validados."""
    pipeline: OlistRAGPipeline = state.get("pipeline")
    if not pipeline:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="O pipeline ainda não foi inicializado.",
        )

    t0 = time.perf_counter()
    try:
        insight = pipeline.generate_insight(
            query=request.query,
            retrieval_k=request.retrieval_k,
            rerank_n=request.rerank_n,
        )
        elapsed = round(time.perf_counter() - t0, 3)

        return QueryResponse(
            success=True,
            latency_seconds=elapsed,
            data=insight,
        )
    except Exception as exc:
        logger.error(f"Erro ao processar consulta: {str(exc)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Falha na inferência do pipeline: {str(exc)}",
        )