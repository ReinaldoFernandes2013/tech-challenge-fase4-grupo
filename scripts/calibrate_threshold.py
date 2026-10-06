import json
from pathlib import Path
import sys
import warnings

warnings.filterwarnings("ignore")

# Raiz do projeto
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

import pandas as pd
from src.core.config import settings
from src.indexing.hybrid_indexer import HybridSearchEngine
from src.rag.reranker import CrossEncoderReranker


def main():
    print("=== Iniciando Calibração de Limiar (Threshold) ===")

    questions_path = PROJECT_ROOT / "data" / "calibration_questions.json"
    parquet_path = settings.DATA_PROCESSED_DIR / "olist_indexed_sample.parquet"

    if not parquet_path.exists():
        parquet_path = settings.DATA_PROCESSED_DIR / "olist_reviews_clean.parquet"

    if not questions_path.exists():
        print(f"Erro: Ficheiro de perguntas não encontrado: {questions_path}")
        return

    if not parquet_path.exists():
        print(f"Erro: Ficheiro parquet não encontrado: {parquet_path}")
        return

    print(f"A carregar perguntas versionadas de: {questions_path}")
    with open(questions_path, "r", encoding="utf-8") as f:
        queries = json.load(f)

    print(f"A carregar dataframe base: {parquet_path}")
    df = pd.read_parquet(parquet_path)
    if "clean_comment" not in df.columns and "text" in df.columns:
        df["clean_comment"] = df["text"]

    print("A inicializar Motor Híbrido e Reranker...")
    hybrid_engine = HybridSearchEngine(df)
    reranker = CrossEncoderReranker()

    results = {}

    for category, qs in queries.items():
        print(f"\n--- {category} ---")
        cat_scores = []
        for q in qs:
            candidates = hybrid_engine.search(q, top_k=15)
            if not candidates:
                print(f"[REJECT] Pontuação: 0.0000 (0 docs) | Pergunta: '{q}'")
                cat_scores.append(0.0)
                continue

            ranked = reranker.rerank(q, candidates, top_n=5)
            if not ranked:
                print(f"[REJECT] Pontuação: 0.0000 (0 docs pós-rerank) | Pergunta: '{q}'")
                cat_scores.append(0.0)
                continue

            max_score = max([doc.get("rerank_score", 0.0) for doc in ranked])
            cat_scores.append(max_score)
            status = "ACIMA (PASS)" if max_score >= 0.10 else "ABAIXO (REJECT)"
            print(f"[{status}] Pontuação: {max_score:.4f} | Pergunta: '{q}'")

        results[category] = cat_scores

    print("\n=== Resumo dos Limiares ===")
    for cat, scores in results.items():
        if scores:
            avg_score = sum(scores) / len(scores)
            print(
                f"{cat}:\n  Média = {avg_score:.4f} | Mín = {min(scores):.4f} | Máx = {max(scores):.4f}"
            )

    # Identificação flexível das categorias
    in_scores = []
    out_scores = []
    for k, v in results.items():
        if "IN-DOMAIN" in k.upper():
            in_scores.extend(v)
        elif "OUT-OF-DOMAIN" in k.upper():
            out_scores.extend(v)

    if in_scores and out_scores:
        min_in = min(in_scores)
        max_out = max(out_scores)
        margin = min_in - max_out

        print(f"\nMínimo In-Domain    : {min_in:.4f}")
        print(f"Máximo Out-Domain   : {max_out:.4f}")
        print(f"Margem de Separação : {margin:.4f}")

        if margin > 0:
            proposed = (min_in + max_out) / 2
            print(f"Separação Perfeita  : Limiar sugerido (Ponto Médio) = {proposed:.4f}")
        else:
            print("Aviso: Sobreposição detectada entre caudas. O limiar de 0.1000 prioriza precisão contra alucinação.")

    print(f"Limiar Operacional Configurado: {settings.RERANKER_CONFIDENCE_THRESHOLD:.4f}")


if __name__ == "__main__":
    main()