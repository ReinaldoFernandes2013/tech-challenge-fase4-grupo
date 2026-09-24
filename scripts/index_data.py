import sys
from pathlib import Path
import pandas as pd

# Adicionar a raiz do projeto ao sys.path para imports funcionarem
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.core.config import settings
from src.core.logging import logger
from src.indexing.vector_store import OlistVectorStore

def run_indexing():
    logger.info("=== Iniciando Indexacao Unica (ChromaDB + Base BM25) ===")
    
    parquet_path = settings.DATA_PROCESSED_DIR / "olist_reviews_clean.parquet"
    if not parquet_path.exists():
        raise FileNotFoundError(f"Parquet processado nao encontrado: {parquet_path}")

    logger.info("A carregar base processada original...")
    df_all = pd.read_parquet(parquet_path)
    
    # TRATAMENTO DE VALORES AUSENTES (Requisito Obrigatorio da Fase 4)
    # Avaliacoes sem comentario de texto (nulos, NaN ou strings vazias) sao inuteis 
    # para a busca semantica e RAG. Portanto, a estrategia eh DESCARTA-LOS.
    initial_len = len(df_all)
    df_all = df_all.dropna(subset=['clean_comment'])
    df_all = df_all[df_all['clean_comment'].str.strip() != '']
    logger.info(f"Tratamento de ausentes: Removidos {initial_len - len(df_all)} registros com comentarios vazios/nulos.")
    
    # Remover duplicatas por review_id para evitar desincronia entre BM25 e Chroma
    initial_len = len(df_all)
    df_all = df_all.drop_duplicates(subset=["review_id"])
    logger.info(f"Removidas {initial_len - len(df_all)} duplicatas de review_id.")
    
    # Justificativa do tamanho com semente fixa:
    # A base inteira possui ~41k registos. Para viabilizar a avaliacao e a
    # execucao dos modelos de embedding locais na CPU sem estourar tempo/recursos,
    # utilizamos uma amostra fixa de 5.000 registos.
    SAMPLE_SIZE = 5000
    if len(df_all) > SAMPLE_SIZE:
        logger.info(f"Gerando amostra fixa de {SAMPLE_SIZE} documentos (random_state=42)...")
        df_index = df_all.sample(n=SAMPLE_SIZE, random_state=42).copy()
    else:
        df_index = df_all.copy()

    # Salvar a base exata que foi indexada para que a API consuma a MESMA base
    indexed_path = settings.DATA_PROCESSED_DIR / "olist_indexed_sample.parquet"
    df_index.to_parquet(indexed_path, index=False)
    logger.info(f"Base de indexacao ({len(df_index)} docs) salva em: {indexed_path}")
    
    # Indexacao no ChromaDB de forma Idempotente
    logger.info("A inicializar Vector Store e indexar documentos no ChromaDB...")
    vstore = OlistVectorStore()
    
    # Limpa a collection atual para garantir idempotencia (se rodar duas vezes, recria)
    try:
        vstore.get_store().delete_collection()
        logger.info("Collection anterior removida com sucesso para garantir idempotencia.")
    except Exception as e:
        logger.warning(f"Falha ao remover collection anterior (pode ser a primeira execução): {e}")
        
    vstore.index_dataframe(df_index, batch_size=200)
    logger.info("=== Indexacao Unica concluida com sucesso! ===")

if __name__ == "__main__":
    run_indexing()



