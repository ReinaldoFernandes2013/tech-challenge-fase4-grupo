import time
from fastapi import APIRouter, HTTPException, Request, status

from src.core.logging import logger
from src.schemas.request import QueryRequest
from src.schemas.response import HealthCheckResponse, QueryResponse

router = APIRouter()


@router.get(
    "/health",
    response_model=HealthCheckResponse,
    status_code=status.HTTP_200_OK,
    tags=["Observabilidade"],
)
async def health_check():
    """Retorna a saúde operacional do serviço e os motores configurados."""
    return HealthCheckResponse()


@router.post(
    "/api/v1/query",
    response_model=QueryResponse,
    status_code=status.HTTP_200_OK,
    tags=["Inteligência RAG"],
)
async def query_pipeline(request: QueryRequest, http_req: Request):
    """Executa busca híbrida, re-ranking neural e síntese executiva validadas."""
    pipeline = http_req.app.state.pipeline
    if not pipeline:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="O pipeline RAG ainda não foi inicializado no servidor.",
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