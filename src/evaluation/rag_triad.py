from langchain_core.prompts import ChatPromptTemplate
from src.core.logging import logger
from src.schemas.eval_schema import RAGTriadMetric
from src.rag.pipeline import get_chat_model

class RAGTriadEvaluator:
    """Avaliador da Tríade de RAG utilizando LLM-as-a-Judge real."""

    def __init__(self, judge_llm=None):
        self.llm = judge_llm or get_chat_model()
        self.structured_llm = self.llm.with_structured_output(RAGTriadMetric)
        
        self.eval_prompt = ChatPromptTemplate.from_messages([
            ("system", """Você é um juiz imparcial avaliando a qualidade de um sistema RAG (Retrieval-Augmented Generation).
Você deve avaliar a Tríade de RAG dando notas rigorosas e fracionadas (ex: 0.1, 0.5, 0.8) de 0.0 a 1.0 para os 3 critérios abaixo, e fornecer uma justificativa clara.

Critérios:
1. Context Relevance: O contexto fornecido contém informações úteis para responder à pergunta?
   (0.0 = totalmente inútil, 1.0 = contém a resposta exata e sem ruído)
2. Groundedness (Fidelidade Factual): Todas as afirmações da resposta (números, motivos, conclusões) estão ESTRITAMENTE baseadas no contexto?
   (0.0 = alucinação total ou invenção de dados, 1.0 = 100% ancorado no contexto sem adições não comprovadas)
3. Answer Relevance: A resposta fornecida endereça a pergunta original do usuário sem divagar?
   (0.0 = evasiva ou off-topic, 1.0 = responde perfeitamente e vai direto ao ponto)
"""),
            ("human", """Por favor, avalie o seguinte cenário:

PERGUNTA DO USUÁRIO:
{query}

CONTEXTO RECUPERADO:
{context}

RESPOSTA GERADA PELO SISTEMA:
{answer}

Gere o resultado da avaliação com notas precisas e justas de 0.0 a 1.0.""")
        ])
        
        self.chain = self.eval_prompt | self.structured_llm

    def evaluate(self, query: str, context: str, answer: str) -> RAGTriadMetric:
        """Calcula Context Relevance, Groundedness e Answer Relevance usando LLM-as-a-Judge."""
        try:
            metric = self.chain.invoke({
                "query": query,
                "context": context,
                "answer": answer
            })
            return metric
        except Exception as e:
            logger.error(f"Falha no LLM-as-a-Judge: {e}")
            return RAGTriadMetric(
                context_relevance=0.0,
                groundedness=0.0,
                answer_relevance=0.0,
                justification=f"Erro de avaliação: {e}"
            )