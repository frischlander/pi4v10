# Task Template: Data Pipeline

Template para criar pipelines de dados robustos.

## Contexto
- **Volume**: [ex: 1M records/dia]
- **Frequência**: [batch/streaming]
- **Fontes**: [APIs, CSV, SQL]
- **Destino**: [DW, S3]
- **SLA**: [ex: dados até 8h]

## Agentes
- @data_engineer (primário)
- @devops_engineer (deploy)
- @qa_analyst (validação)

## Exemplo de Uso

```
@orchestrator

Pipeline ETL para análise de vendas:

Fonte: API REST (1M transações/dia)
Transformação: Agregações por produto
Destino: Snowflake (star schema)
Orquestração: Airflow (diário 2h)
Validação: Great Expectations
SLA: Dados prontos até 6h

Coordene @data_engineer, @devops_engineer e @qa_analyst.
```

## Output Esperado
- [ ] Código ETL (Python/Spark)
- [ ] Airflow DAG
- [ ] Great Expectations suite
- [ ] Terraform infra
- [ ] Testes (pytest)
- [ ] Documentação
