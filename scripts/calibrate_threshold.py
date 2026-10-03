import json
import os
from pathlib import Path
import sys
import warnings

warnings.filterwarnings("ignore")

# Raiz do projeto
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

import pandas as pd
from src.indexing.hybrid_indexer import HybridSearchEngine
from src.rag.reranker import CrossEncoderReranker


def main():
    print("=== Iniciando Calibração de Threshold ===")

    questions_path = PROJECT_ROOT / "data" / "calibration_questions.json"
    parquet_path = PROJECT_ROOT / "data/processed/olist_indexed_sample.parquet"

    if not questions_path.exists():
        print(f"Erro: Arquivo não encontrado: {questions_path}")
        return

    print(f"Carregando perguntas versionadas de: {questions_path}")
    with open(questions_path, "r", encoding="utf-8") as f:
        queries = json.load(f)

    print(f"Carregando dataframe base: {parquet_path}")
    df = pd.read_parquet(parquet_path)
    if "clean_comment" not in df.columns and "text" in df.columns:
        df["clean_comment"] = df["text"]

    print("Inicializando Motor Híbrido e Reranker...")
    hybrid_engine = HybridSearchEngine(df)
    reranker = CrossEncoderReranker()

    results = {}

    for category, qs in queries.items():
        print(f"\n--- {category} ---")
        cat_scores = []
        for q in qs:
            candidates = hybrid_engine.search(q, top_k=15)
            if not candidates:
                print(f"[REJECT] Score: 0.0000 (0 docs) | Query: '{q}'")
                cat_scores.append(0.0)
                continue

            ranked = reranker.rerank(q, candidates, top_n=5)
            if not ranked:
                print(f"[REJECT] Score: 0.0000 (0 docs pós-rerank) | Query: '{q}'")
                cat_scores.append(0.0)
                continue

            max_score = max([doc.get("rerank_score", 0.0) for doc in ranked])
            cat_scores.append(max_score)
            status = "ACIMA (PASS)" if max_score >= 0.10 else "ABAIXO (REJECT)"
            print(f"[{status}] Score: {max_score:.4f} | Query: '{q}'")

        results[category] = cat_scores

    print("\n=== Resumo dos Limiares ===")
    for cat, scores in results.items():
        if scores:
            avg_score = sum(scores) / len(scores)
            print(
                f"{cat}:\n  Média = {avg_score:.4f} | Mín = {min(scores):.4f} | Máx = {max(scores):.4f}"
            )

    in_scores = results.get("IN-DOMAIN (Olist, E-commerce, Logística)", [])
    out_scores = results.get("OUT-OF-DOMAIN (Ruído, Assuntos não relacionados)", [])

    if in_scores and out_scores:
        min_in = min(in_scores)
        max_out = max(out_scores)
        proposed = (min_in + max_out) / 2
        print(f"\nMínimo In-Domain : {min_in:.4f}")
        print(f"Máximo Out-Domain: {max_out:.4f}")
        print(f"Threshold Sugerido (Ponto Médio): {proposed:.4f}")
        print(f"Threshold Configurado no Pipeline: 0.1000")


if __name__ == "__main__":
    main()