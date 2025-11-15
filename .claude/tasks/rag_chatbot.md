# Task Template: RAG Chatbot

Template para chatbots com RAG.

## Contexto
- **Domínio**: [suporte, vendas, docs]
- **Fontes**: [PDFs, docs, FAQs]
- **Volume**: [X documentos, Y queries/mês]
- **Features**: [citations, streaming, memory]

## Agentes
- @ai_architect (primário)
- @data_engineer (ingestão)
- @devops_engineer (deploy)

## Exemplo de Uso

```
@orchestrator

Chatbot RAG para suporte técnico:

Knowledge Base:
- 200 PDFs (manuais)
- 50 FAQs
- 1000 tickets resolvidos

Requisitos:
- Respostas com citations
- Histórico (5 mensagens)
- Latência <3s p95
- Fallback para humano

Stack:
- Embeddings: OpenAI
- Vector DB: Pinecone
- LLM: Claude Sonnet
- Framework: LangChain

Coordene @ai_architect, @data_engineer e @devops_engineer.
```

## Output Esperado
- [ ] RAG pipeline code
- [ ] Prompt templates
- [ ] API (FastAPI)
- [ ] Docker + K8s manifests
- [ ] Evaluation suite
- [ ] Documentação
