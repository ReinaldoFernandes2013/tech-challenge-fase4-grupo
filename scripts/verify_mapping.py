from pathlib import Path
import random
import sys
import warnings

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

import pandas as pd
from src.core.config import settings
from src.indexing.vector_store import OlistVectorStore


def verify_mapping():
    parquet_path = settings.DATA_PROCESSED_DIR / "olist_indexed_sample.parquet"
    if not parquet_path.exists():
        print(f"ERRO: Arquivo não encontrado: {parquet_path}")
        sys.exit(1)

    df = pd.read_parquet(parquet_path)

    # Semente fixa para reprodutibilidade científica da amostragem
    random.seed(42)
    sample_ids = random.sample(df["review_id"].dropna().tolist(), 5)

    print("=== Verificando Mapeamento review_id -> ChromaDB ===")
    print(f"Total de linhas no Parquet: {len(df)}")
    print(f"Total de review_ids ÚNICOS no Parquet: {df['review_id'].nunique()}")

    vstore = OlistVectorStore()
    store = vstore.get_store()

    success = True
    for rid in sample_ids:
        results = store.get(where={"review_id": rid})
        count = len(results.get("ids", []))

        print(f"review_id: {rid} | Documentos encontrados no ChromaDB: {count}")
        if count != 1:
            success = False

    if success:
        print("\nSUCESSO: Cada review_id amostrado possui EXATAMENTE 1 documento no ChromaDB.")
        return 0
    else:
        print("\nERRO: Algum review_id possui 0 ou múltiplos documentos associados no ChromaDB.")
        sys.exit(1)


if __name__ == "__main__":
    verify_mapping()