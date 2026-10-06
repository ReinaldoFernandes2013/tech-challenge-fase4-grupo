import json
from pathlib import Path
import time
import pandas as pd
from rich.console import Console
from rich.table import Table

from src.core.config import settings
from src.core.logging import logger
from src.evaluation.metrics import calculate_triad_statistics
from src.evaluation.rag_triad import RAGTriadEvaluator
from src.rag.pipeline import OlistRAGPipeline
from src.schemas.eval_schema import BenchmarkReport, SingleEvaluationResult

console = Console()

# Conjunto de testes de validação para a banca avaliadora
TEST_SUITE = [
    "Quais os principais problemas relatados sobre atraso na entrega e extravio?",
    "Quais as reclamações mais frequentes sobre produtos com defeito de fabricação?",
    "Como os clientes avaliam o atendimento pós-venda e a resolução de reembolsos?",
]


def run_benchmark():
    # 1. Carrega estritamente a mesma base que foi indexada no ChromaDB
    parquet_path = settings.DATA_PROCESSED_DIR / "olist_indexed_sample.parquet"
    if not parquet_path.exists():
        parquet_path = settings.DATA_PROCESSED_DIR / "olist_reviews_clean.parquet"

    if not parquet_path.exists():
        raise FileNotFoundError(f"Arquivo parquet para benchmark ausente: {parquet_path}")

    logger.info(f"Carregando base de benchmark formal a partir de: {parquet_path}")
    df_benchmark = pd.read_parquet(parquet_path)

    pipeline = OlistRAGPipeline(df_benchmark)
    evaluator = RAGTriadEvaluator()

    results = []

    console.print("\n[bold cyan]🚀 Iniciando Benchmark da Tríade de RAG (Olist Voice of Customer)[/bold cyan]\n")

    for idx, query in enumerate(TEST_SUITE, 1):
        console.print(f"[yellow]({idx}/{len(TEST_SUITE)}) Avaliando pergunta:[/yellow] '{query}'")

        t0 = time.perf_counter()
        insight = pipeline.generate_insight(query=query, retrieval_k=15, rerank_n=5)
        elapsed = round(time.perf_counter() - t0, 3)

        # Formatação do contexto com proteção para abstenção formal
        if insight.citations:
            context_str = "\n".join([f"[{c.review_id}] {c.excerpt}" for c in insight.citations])
        else:
            context_str = "Nenhum documento recuperado acima do limiar semântico (Abstenção acionada)."

        answer_str = (
            f"Resumo: {insight.executive_summary}\n"
            f"Causas: {', '.join(insight.key_root_causes)}\n"
            f"Ações: {', '.join(insight.actionable_recommendations)}"
        )

        triad_metric = evaluator.evaluate(query=query, context=context_str, answer=answer_str)

        results.append(
            SingleEvaluationResult(
                query=query,
                latency_seconds=elapsed,
                retrieved_docs_count=len(insight.citations),
                triad=triad_metric,
            )
        )

    # Estatísticas consolidadas
    latencies = [r.latency_seconds for r in results]
    contexts = [r.triad.context_relevance for r in results]
    groundedness_scores = [r.triad.groundedness for r in results]
    answers = [r.triad.answer_relevance for r in results]

    mean_latency = round(float(pd.Series(latencies).mean()), 3)
    mean_context = calculate_triad_statistics(contexts)["mean"]
    mean_groundedness = calculate_triad_statistics(groundedness_scores)["mean"]
    mean_answer = calculate_triad_statistics(answers)["mean"]

    report = BenchmarkReport(
        total_evaluations=len(results),
        mean_latency_seconds=mean_latency,
        mean_context_relevance=mean_context,
        mean_groundedness=mean_groundedness,
        mean_answer_relevance=mean_answer,
        results=results,
    )

    # Apresentação no Terminal
    table = Table(title="📊 Relatório de Auditoria: Tríade de RAG (Olist)", show_header=True, header_style="bold magenta")
    table.add_column("Query Investigada", style="dim", width=40)
    table.add_column("Latência", justify="right")
    table.add_column("Context Rel.", justify="right")
    table.add_column("Groundedness", justify="right")
    table.add_column("Answer Rel.", justify="right")

    for r in results:
        table.add_row(
            r.query[:38] + "...",
            f"{r.latency_seconds}s",
            f"{r.triad.context_relevance:.2f}",
            f"{r.triad.groundedness:.2f}",
            f"{r.triad.answer_relevance:.2f}",
        )

    console.print("\n")
    console.print(table)

    summary_table = Table(title="🎯 Médias Globais do Sistema", show_header=True, header_style="bold green")
    summary_table.add_column("Métrica", style="bold")
    summary_table.add_column("Score Médio", justify="center")
    summary_table.add_column("Meta Acadêmica", justify="center")
    summary_table.add_column("Status", justify="center")

    summary_table.add_row("Context Relevance", f"{mean_context * 100:.1f}%", ">= 75.0%", "✅ Aprovado" if mean_context >= 0.75 else "⚠️ Ajustar")
    summary_table.add_row("Groundedness (Zero Alucinação)", f"{mean_groundedness * 100:.1f}%", ">= 90.0%", "✅ Aprovado" if mean_groundedness >= 0.90 else "⚠️️ Ajustar")
    summary_table.add_row("Answer Relevance", f"{mean_answer * 100:.1f}%", ">= 80.0%", "✅ Aprovado" if mean_answer >= 0.80 else "⚠️ Ajustar")
    summary_table.add_row("Tempo Médio de Inferência", f"{mean_latency}s", "< 30.0s", "✅ Aprovado" if mean_latency < 30.0 else "⚠️ Alerta")

    console.print("\n")
    console.print(summary_table)

    # Persistência do Relatório
    output_dir = Path("data/benchmarks")
    output_dir.mkdir(parents=True, exist_ok=True)
    report_file = output_dir / "rag_triad_report.json"

    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report.model_dump_json(indent=2))

    console.print(f"\n[green]✓ Relatório salvo com sucesso em:[/green] [bold]{report_file}[/bold]\n")


if __name__ == "__main__":
    run_benchmark()