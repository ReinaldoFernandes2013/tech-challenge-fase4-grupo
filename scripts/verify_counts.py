import sys
from pathlib import Path
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.core.config import settings
from src.indexing.vector_store import OlistVectorStore
from src.indexing.bm25_retriever import BM25RetrieverOlist

def verify():
    # 1. Contagem no Parquet indexado
    indexed_path = settings.DATA_PROCESSED_DIR / "olist_indexed_sample.parquet"
    if not indexed_path.exists():
        print("Arquivo parquet indexado nao encontrado.")
        return
        
    df = pd.read_parquet(indexed_path)
    parquet_count = len(df)
    print(f"Linhas no Parquet indexado (amostra sem duplicatas): {parquet_count}")
    
    # 2. Contagem no ChromaDB
    vstore = OlistVectorStore()
    store = vstore.get_store()
    chroma_count = store._collection.count()
    print(f"Documentos no ChromaDB: {chroma_count}")
    
    # 3. Contagem no BM25
    bm25 = BM25RetrieverOlist(df)
    bm25_count = bm25.bm25.corpus_size if hasattr(bm25.bm25, 'corpus_size') else len(bm25.bm25.doc_len)
    print(f"Documentos no BM25: {bm25_count}")

    if parquet_count == chroma_count == bm25_count:
        print("\nSUCESSO: As tres contagens sao estritamente iguais!")
    else:
        print("\nERRO: Contagens divergentes!")

if __name__ == "__main__":
    verify()

