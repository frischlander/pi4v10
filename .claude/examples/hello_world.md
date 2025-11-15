# Hello World - Primeiro Uso

## Exemplo 1: Tarefa Simples

No chat do VS Code, digite:

```
@data_engineer Crie um schema SQL para um blog com:
- Tabela de posts (id, title, content, author_id, created_at)
- Tabela de users (id, name, email)
- Tabela de comments (id, post_id, user_id, content)
- Relacionamentos apropriados
```

## Exemplo 2: Uso do Orchestrator

```
@orchestrator Criar API REST para gerenciar posts de blog:

Backend:
- FastAPI
- PostgreSQL
- Autenticação JWT

Requisitos:
- CRUD de posts
- Sistema de comentários
- Rate limiting
- Documentação Swagger

Deploy:
- Docker
- Docker Compose para dev
- Pronto para produção

Coordene @data_engineer, @devops_engineer e @qa_analyst.
```

## Exemplo 3: Template

```
@orchestrator usar template data_pipeline

[Preencher os campos do template quando solicitado]
```

## Dica

Após receber o código, você pode iterar:

```
@orchestrator Adicione caching Redis na API
@qa_analyst Crie testes automatizados para todos os endpoints
@devops_engineer Adicione CI/CD com GitHub Actions
```

Claude Code mantém o contexto! 🚀
