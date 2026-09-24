import os
import sys
from pathlib import Path
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.indexing.hybrid_indexer import HybridSearchEngine
from src.rag.reranker import CrossEncoderReranker

def main():
    print("=== Iniciando Calibração de Threshold ===")
    
    parquet_path = "data/processed/olist_indexed_sample.parquet"
    print(f"Carregando dataframe base: {parquet_path}")
    df = pd.read_parquet(parquet_path)
    
    print("Inicializando Motor Híbrido e Reranker...")
    hybrid_engine = HybridSearchEngine(df)
    reranker = CrossEncoderReranker()
    
    queries = {
        "IN-DOMAIN (Olist, E-commerce, Logística)": [
            "Por que os clientes estao reclamando de atraso em SP?",
            "Qual o motivo da nota 1 na logistica?",
            "Quais sao as principais criticas sobre a qualidade dos produtos?",
            "O que dizem os clientes satisfeitos com a entrega rapida?",
            "Ha problemas no SAC e no atendimento ao cliente?",
            "Reclamacoes sobre produtos falsificados ou quebrados.",
            "Elogios sobre a facilidade de compra.",
            "A transportadora esta entregando no prazo?",
            "Problemas com reembolso e cancelamento de pedidos.",
            "Qual o nivel de satisfacao com os vendedores da Olist?"
        ],
        "OUT-OF-DOMAIN (Ruído, Assuntos não relacionados)": [
            "Como investir em acoes da bolsa de valores?",
            "Qual a receita para um bolo de cenoura fofinho?",
            "Quais sao as regras de impedimento no futebol?",
            "Me explique a teoria da relatividade geral de Einstein.",
            "Qual a capital da Franca e quantos habitantes tem?"
        ]
    }
    
    results = {}
    
    for category, qs in queries.items():
        print(f"\n--- {category} ---")
        results[category] = []
        for q in qs:
            candidates = hybrid_engine.search(q, top_k=15)
            if not candidates:
                print(f"Q: '{q}' | Score Maximo: N/A (0 docs)")
                continue
                
            ranked = reranker.rerank(q, candidates, top_n=5)
            if not ranked:
                print(f"Q: '{q}' | Score Maximo: N/A (0 docs pos-rerank)")
                continue
            
            # Pega o score do documento mais bem rankeado
            max_score = max([doc.get('rerank_score', 0.0) for doc in ranked])
            results[category].append(max_score)
            print(f"Q: '{q}' | Score Maximo: {max_score:.4f}")
            
    print("\n=== Resumo e Proposta de Threshold ===")
    in_scores = results["IN-DOMAIN (Olist, E-commerce, Logística)"]
    out_scores = results["OUT-OF-DOMAIN (Ruído, Assuntos não relacionados)"]
    
    min_in = min(in_scores) if in_scores else 0.0
    max_out = max(out_scores) if out_scores else 0.0
    
    print(f"Minimo In-Domain Score : {min_in:.4f}")
    print(f"Maximo Out-of-Domain   : {max_out:.4f}")
    
    # Heuristica de threshold
    proposed = (min_in + max_out) / 2
    print(f"\nThreshold Sugerido (Ponto Medio): {proposed:.4f}")

if __name__ == '__main__':
    main()