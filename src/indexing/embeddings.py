import os
from pathlib import Path
import torch
from langchain_core.embeddings import Embeddings
from src.core.config import settings
from src.core.logging import logger

# 1. Blindagem de threads para manter o SO 100% responsivo
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.set_num_threads(4)

# 2. Isola o cache de modelos dentro da pasta do projeto
CACHE_DIR = str(Path("./data/models/hf").resolve())
os.environ["HF_HOME"] = CACHE_DIR
os.environ["TRANSFORMERS_CACHE"] = CACHE_DIR
os.environ["SENTENCE_TRANSFORMERS_HOME"] = CACHE_DIR


def get_embedding_function() -> Embeddings:
    """Instancia embeddings com limites de concorrência e cache isolado."""
    if settings.OPENAI_API_KEY and settings.OPENAI_API_KEY.startswith("sk-"):
        logger.info(f"A inicializar embeddings OpenAI: {settings.EMBEDDING_MODEL}")
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(
            model=settings.EMBEDDING_MODEL,
            openai_api_key=settings.OPENAI_API_KEY,
        )

    logger.info("A inicializar embeddings locais (cache isolado em data/models)...")
    from langchain_huggingface import HuggingFaceEmbeddings

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Dispositivo alocado: {device.upper()}")

    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        cache_folder=CACHE_DIR,
        model_kwargs={"device": device},
        encode_kwargs={"batch_size": 32, "normalize_embeddings": True},
    )