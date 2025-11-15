# 🎭 Claude Code Orchestrator

Sistema de orquestração multi-agent para projetos de IA, ML, Data e DevOps.

## 🚀 Quick Start

### No VS Code

1. Abra o chat do Claude Code (`Cmd/Ctrl + Shift + I`)

2. Use o orchestrator:
```
@orchestrator Preciso criar um pipeline de dados para análise de vendas
```

3. O orchestrator vai automaticamente:
   - Analisar sua solicitação
   - Chamar sub-agents especializados
   - Consolidar resultados
   - Entregar código completo

## 🤖 Sub-Agents Disponíveis

- `@data_engineer`: Pipelines, ETL, modelagem
- `@ml_engineer`: Modelos, treinamento, deployment
- `@ai_architect`: Sistemas IA, RAG, LLMs
- `@devops_engineer`: Infraestrutura, CI/CD
- `@qa_analyst`: Testes, validação, qualidade

## 📚 Templates Prontos

Use templates para começar rapidamente:

```
@orchestrator usar template data_pipeline
@orchestrator usar template rag_chatbot
@orchestrator usar template mlops_pipeline
```

## 💡 Exemplos

### Tarefa Simples
```
@data_engineer Otimize esta query SQL: [query]
```

### Tarefa Complexa
```
@orchestrator Sistema completo de recomendação ML:
- Pipeline de dados (histórico de compras)
- Modelo de ML (collaborative filtering)
- API REST (<100ms)
- Deploy em Kubernetes
```

## 🎯 Workflow Típico

1. Você descreve o que precisa
2. Orchestrator analisa e delega
3. Sub-agents executam suas partes
4. Orchestrator consolida
5. Você recebe código completo + docs + testes

## 📖 Documentação

- `prompts/`: Definições dos agentes
- `tasks/`: Templates de tarefas comuns
- `examples/`: Exemplos práticos
- `config.json`: Configuração do projeto

## 🔥 Shortcuts

- `Cmd/Ctrl + Shift + I`: Abrir Claude Code
- `@nome_agent`: Mencionar sub-agent
- `/task`: Criar task de template

## 💰 Vantagem vs Python

✅ Sem gerenciamento de API key
✅ Incluído na assinatura Claude
✅ Integração total com VS Code
✅ Context automático (seus arquivos)
✅ Iteração natural via chat

Happy Orchestrating! 🚀
