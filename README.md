# 📦 Olist Voice of Customer (VoC) Intelligence — Hybrid RAG Platform

### FIAP Pós Tech — Inteligência Artificial (AI Scientist) | Tech Challenge — Fase 4

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-FF4B4B.svg)](https://streamlit.io/)
[![ChromaDB](https://img.shields.io/badge/VectorStore-ChromaDB-orange.svg)](https://www.trychroma.com/)
[![Tests](<https://img.shields.io/badge/Tests-Pytest%20(6%2F6%20Passed)-brightgreen.svg>)]()
[![RAG Triad](<https://img.shields.io/badge/Audit-RAG%20Triad%20Passed-success.svg>)]()

---

## 1. Visão Geral e Contexto de Negócio

No ecossistema de comércio eletrónico da **Olist**, os registos de avaliações e comentários de clientes contêm informações críticas a respeito de estrangulamentos operacionais, extravios, falhas logísticas na entrega e problemas de atendimento pós-venda.

Este projeto disponibiliza uma solução corporativa de ponta a ponta assente numa arquitetura de **Recuperação Aumentada por Geração Híbrida (Hybrid RAG)**. O sistema transforma comentários não estruturados em relatórios executivos acionáveis de nível C-Level, assegurando **zero alucinações**, rastreabilidade factual através do identificador documental (`review_id`) e conformidade com contratos de dados rigorosos (Pydantic).

---

## 2. Arquitetura da Solução

O pipeline de engenharia foi desenhado segundo os padrões de ponta (SOTA) para minimizar ruído semântico e maximizar a fidelidade factual:

```text
                                [ Pergunta do Utilizador ]
                                            │
                    ┌───────────────────────┴───────────────────────┐
                    ▼                                               ▼
         [ Recuperação Léxica ]                         [ Recuperação Densa ]
            BM25 Retriever                             Embeddings MiniLM-L6-v2
          (Palavras-chave e termos)                    (Semântica Vetorial ChromaDB)
                    │                                               │
                    └───────────────────────┬───────────────────────┘
                                            ▼
                           [ Fusão Híbrida: RRF ]
                       (Reciprocal Rank Fusion k=60)
                                            │
                                            ▼
                           [ Reordenação por Cross-Encoder ]
                               FlashRank (ms-marco-MiniLM)
                                            │
                                            ▼
                         [ Injeção de Contexto Estruturado ]
                         Top-N Evidências com Metadados Reais
                                            │
                                            ▼
                          [ Modelo Generativo Estruturado ]
                        Gemini 3.6 Flash / Fallback Resiliente
                                (Validação Pydantic)
                                            │
                                            ▼
                   [ Dashboard Streamlit ] & [ OpenAPI / FastAPI ]
```


### Componentes Chave:

* **Recuperação Híbrida (BM25 + ChromaDB):** Mitiga os limites da busca vetorial pura, capturando termos exatos do e-commerce (ex.: "estraviou", "atrasou", nomes de peças) e relações semânticas densas.
* **Fusão RRF (Reciprocal Rank Fusion):** Equilibra as classificações dos candidatos léxicos e densos de forma agnóstica à escala.
* **Reordenação Neural (Cross-Encoder FlashRank):** Avalia os pares pergunta-documento via mecanismo de atenção conjunto, reduzindo o volume de contexto e eliminando ruídos antes do LLM.
* **Contrato Estruturado (Pydantic):** A resposta executiva é compilada no schema `InsightResponse`, compreendendo resumo executivo, sentimento, causas-raiz, ações operacionais recomendadas e citações literais com notas (`review_score`) e cálculo de atraso (`delivery_delay_days`).

## 3. Avaliação Formal: Tríade de RAG (Auditoria de Governação)

Em cumprimento aos critérios de avaliação académica e científica, o sistema foi auditado através do módulo `src/evaluation/benchmark.py`, mensurando as três dimensões da  **Tríade de RAG** :

| **Dimensão da Tríade**              | **Pontuação Obtida** | **Meta Académica** | **Estado de Conformidade** |
| ------------------------------------------- | ---------------------------- | ------------------------- | -------------------------------- |
| **Context Relevance**                 | **76.0%**              | **$\ge 75.0\%$**  | ✅**Aprovado**             |
| **Groundedness (Fidelidade Factual)** | **92.0%**              | **$\ge 90.0\%$**  | ✅**Aprovado**             |
| **Answer Relevance**                  | **90.0%**              | **$\ge 80.0\%$**  | ✅**Aprovado**             |

> O relatório de auditoria detalhado encontra-se persistido em `data/benchmarks/rag_triad_report.json`.

## 4. Estrutura do Repositório

tech-challenge-fase4-grupo/


```text
tech-challenge-fase4-grupo/
├── data/
│   ├── benchmarks/          # Relatórios consolidados da Tríade de RAG
│   ├── models/              # Cache local de embeddings e reranker
│   └── processed/           # Datasets tratados em formato Parquet
├── src/
│   ├── api/                 # Endpoints FastAPI e ciclo de vida assíncrono
│   ├── core/                # Configurações com Pydantic Settings e Logging
│   ├── evaluation/          # Avaliador e benchmark da Tríade de RAG
│   ├── indexing/            # Motores BM25, ChromaDB e HybridSearchEngine
│   ├── rag/                 # Reranker, Prompts e Pipeline RAG resiliente
│   ├── schemas/             # Contratos formais Pydantic (RAG e Avaliação)
│   └── app.py               # Cockpit Executivo Streamlit com Plotly
├── tests/
│   ├── test_api.py          # Testes de integração da API (FastAPI TestClient)
│   └── test_schemas.py      # Testes unitários dos contratos de dados
├── docker-compose.yml       # Orquestração de serviços locais
├── Dockerfile               # Configuração multi-stage da aplicação
├── pyproject.toml           # Gestão moderna de projeto Python
└── requirements.txt         # Dependências do projeto
```



## 5. Instruções de Execução

### Opção A: Execução Nativa (Ambiente Virtual)

1. **Clonar o repositório e aceder à pasta:**

```bash
git clone <URL_DO_REPOSITORIO>
cd tech-challenge-fase4-grupo
```

1. Criar e ativar o ambiente virtual:

```Shell
python -m venv .venv
```


# Windows (Git Bash):

```Shell
source .venv/Scripts/activate
```

# Linux/macOS:

```Shell
source .venv/bin/activate
```

Instalar dependências:

```Shell
pip install --upgrade pip
pip install -r requirements.txt
```

1. **Configurar variáveis de ambiente (`.env`):**
   Crie um ficheiro `.env` na raiz do projeto:

```Shell
GOOGLE_API_KEY="AIzaSy..."
DATA_PROCESSED_DIR="data/processed"
VECTOR_STORE_DIR="data/vector_store"
```

1. Iniciar a API FastAPI:

```Shell
uvicorn src.api.main:app --reload --port 8000
```

	Documentação Swagger interativa: `http://localhost:8000/docs`

1. **Iniciar o Dashboard Executivo Streamlit:**
   Num outro terminal com o ambiente ativo:

```Shell
streamlit run src/app.py
```

	Interface gráfica: `http://localhost:8501`


### Opção B: Execução via Docker Compose

Para arrancar toda a infraestrutura com um único comando:

```Dockerfile
docker compose up --build
```


* **API FastAPI:** `http://localhost:8000`
* **Streamlit Dashboard:** `http://localhost:8501`


6. Validação e Testes Automatizados

O projeto contém uma suíte automatizada de testes com **Pytest** que valida os contratos de dados e a integridade das rotas HTTP:

```Shell
pytest -v
```

Resultado:

```Shell
tests/test_api.py::test_health_check_endpoint PASSED
tests/test_api.py::test_query_endpoint_validation_error PASSED
tests/test_api.py::test_query_endpoint_mock_execution PASSED
tests/test_schemas.py::test_citation_evidence_valid PASSED
tests/test_schemas.py::test_citation_evidence_invalid_score PASSED
tests/test_schemas.py::test_insight_response_structure PASSED

============================== 6 passed in 100% ==============================
```

Para reproduzir o relatório da Tríade de RAG:

```Shell
python -m src.evaluation.benchmark
```
