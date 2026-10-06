from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd

from src.api.routes import router as api_router
from src.core.config import settings
from src.core.logging import logger
from src.rag.pipeline import OlistRAGPipeline


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Ciclo de vida: carrega o dataset e instancia o pipeline uma única vez."""
    parquet_path = settings.DATA_PROCESSED_DIR / "olist_indexed_sample.parquet"
    if not parquet_path.exists():
        parquet_path = settings.DATA_PROCESSED_DIR / "olist_reviews_clean.parquet"

    if not parquet_path.exists():
        logger.error(f"Base de dados não encontrada em: {parquet_path}")
        raise RuntimeError(f"Base Parquet ausente: {parquet_path}")

    logger.info("A carregar dataset e a instanciar o pipeline Olist RAG...")
    df_indexed = pd.read_parquet(parquet_path)

    # Compatibilização de esquema: assegura que 'clean_comment' e 'text' coexistam
    if "clean_comment" not in df_indexed.columns and "text" in df_indexed.columns:
        df_indexed["clean_comment"] = df_indexed["text"]
    elif "text" not in df_indexed.columns and "clean_comment" in df_indexed.columns:
        df_indexed["text"] = df_indexed["clean_comment"]

    # Atribuição ao app.state para evitar variáveis globais e garantir concorrência limpa
    app.state.pipeline = OlistRAGPipeline(df_indexed)
    
    logger.info("Pipeline RAG inicializado com sucesso e pronto para receber requisições.")
    yield
    app.state.pipeline = None


app = FastAPI(
    title="Olist Voice of Customer - RAG API",
    description="API de Recuperação Híbrida e Inteligência Estruturada para E-commerce",
    version="1.0.0",
    lifespan=lifespan,
)

# Configuração de CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Registo modular de rotas
app.include_router(api_router)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=False)