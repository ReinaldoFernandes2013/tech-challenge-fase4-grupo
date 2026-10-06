.PHONY: help install index test benchmark run-api run-app docker-up docker-down clean

PYTHON ?= python

help:
	@echo "Comandos disponíveis:"
	@echo "  make install      Instala todas as dependências do projeto"
	@echo "  make index        Executa o pipeline de indexação (BM25 + ChromaDB)"
	@echo "  make test         Executa a suíte de testes com pytest"
	@echo "  make benchmark    Roda o benchmark formal da Tríade de RAG"
	@echo "  make run-api      Inicia o servidor backend FastAPI"
	@echo "  make run-app      Inicia a interface do Streamlit"
	@echo "  make docker-up    Constrói e sobe os contêineres via Docker Compose"
	@echo "  make docker-down  Derruba os contêineres e limpa a rede"
	@echo "  make clean        Remove caches e arquivos temporários"

install:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt

index:
	$(PYTHON) scripts/index_data.py
	$(PYTHON) scripts/verify_counts.py

test:
	pytest -v

benchmark:
	$(PYTHON) -m src.evaluation.benchmark

run-api:
	uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

run-app:
	streamlit run src/app.py --server.port 8501

docker-up:
	docker compose up --build

docker-down:
	docker compose down

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +