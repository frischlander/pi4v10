# 🦟 Predição de Hospitalização por Dengue - Healthcare ML

## 🎯 Objetivo

Desenvolver modelo de Machine Learning (Regressão Logística) para predizer a probabilidade de hospitalização de pacientes com dengue, otimizado com **Optuna** para maximizar **Recall** (detectar casos graves).

## 🏥 Contexto Clínico

Em saúde pública, é **crítico** detectar pacientes que precisam de hospitalização. Um **falso negativo** (paciente grave não detectado) pode resultar em complicações graves ou óbito. Portanto, priorizamos:

1. **Recall (Sensitivity)** - Detectar o máximo de casos graves
2. **NPV** - Confiança em resultados negativos  
3. **Precision (PPV)** - Evitar alarmes falsos excessivos
4. **Specificity** - Identificar corretamente não-hospitalizações

---

## 📊 Dataset

- **Arquivo**: `df_dengue_tratado.csv`
- **Registros**: ~33.000 casos de dengue
- **Período**: 2013-2025
- **Região**: Sertãozinho, SP e região
- **Desbalanceamento**: ~1.17% de hospitalizações (classe minoritária)

### Features Selecionadas (14)

Após processo rigoroso de seleção (Feature Importance + Correlação + Chi-Square), foram selecionadas:

#### Temporais (Sazonalidade)
- **DIAS_SINTOMA_NOTIFIC_TEMP**: Tempo entre sintomas e notificação (engineered)
- **TRIMESTRE**: Trimestre do ano
- **MES**: Mês da notificação
- **DIAS_SINTOMA_NOTIFIC**: Dias entre sintomas e notificação
- **NU_ANO**: Ano da notificação
- **ANO**: Ano (categorizado)

#### Clínicas
- **SEVERITY_SCORE**: Score de severidade (engineered)
- **QTD_IGNORADOS**: Quantidade de campos ignorados (proxy de completude)

#### Demográficas
- **IDADE**: Idade do paciente

#### Comorbidades
- **TEM_COMORBIDADE**: Flag binária (presença de comorbidade)
- **COMORBIDADE_SCORE**: Score de comorbidades (engineered)
- **HEPATOPAT_BIN**: Hepatopatia (Sim/Não)
- **DIABETES_BIN**: Diabetes (Sim/Não)
- **RENAL_BIN**: Doença renal (Sim/Não)

#### Target
- **HOSPITALIZ**: SIM/NÃO (variável a ser prevista)

---

## 🤖 Modelo Desenvolvido

**Regressão Logística** otimizada com **Optuna**:

### Hiperparâmetros Otimizados
- **C**: 0.00278 (regularização forte)
- **penalty**: L1 (LASSO - seleção de features)
- **solver**: saga (suporta L1)
- **max_iter**: 2000
- **class_weight**: None (balanceamento via SMOTE)

### Otimização Optuna
- **50 trials** de busca de hiperparâmetros
- **5-fold Cross-Validation** estratificado
- **Objetivo**: Maximizar Recall (Sensitivity)
- **Balanceamento**: SMOTE no conjunto de treino

---

## 📁 Estrutura do Projeto

```
pi4v10/
├── df_dengue_tratado.csv              # Dataset original
├── modelo_dengue_final_optuna.ipynb   # 📓 Notebook principal (EXECUTAR ESTE)
├── treinar_modelo_final.py            # Script Python alternativo
├── requirements.txt                   # Dependências Python
├── README_DENGUE_ML.md                # Este arquivo
│
├── setup_environment.sh               # 🛠️  Script de setup automático
├── activate.sh                        # Ativar ambiente virtual
├── start_jupyter.sh                   # Iniciar Jupyter
│
├── .claude/                           # Sistema de orquestração multi-agent
│   ├── config.json
│   ├── prompts/
│   │   ├── orchestrator.md
│   │   ├── healthcare_ml_specialist.md
│   │   ├── data_engineer.md
│   │   ├── ml_engineer.md
│   │   └── ...
│   └── tasks/
│
├── config_modelo.json                 # Configuração do modelo
├── features_selecionadas.txt          # Lista de 14 features
│
└── outputs/ (gerados após execução)
    ├── modelo_reglog_otimizado.pkl    # 🤖 Modelo final
    ├── scaler_final.pkl               # Normalizador
    ├── optuna_study_logreg.pkl        # Estudo Optuna
    ├── config_modelo.json             # Métricas e configuração
    │
    └── visualizations/
        ├── viz_shap_importance_bar.png
        ├── viz_confusion_matrix.png
        ├── viz_roc_curve.png
        ├── viz_pr_curve.png
        └── viz_probability_distribution.png
```

---

## 🚀 Como Executar

### Método 1: Setup Automático (Recomendado)

```bash
cd /home/ericobon/insightesfera/PORTFOLIO_ACADEMICO/pi4v10

# Executar setup completo (cria venv, instala deps, inicia Jupyter)
bash setup_environment.sh

# Ou manualmente:
bash activate.sh          # Ativar ambiente
bash start_jupyter.sh     # Iniciar Jupyter
```

### Método 2: Manual

```bash
# Criar ambiente virtual
python3 -m venv venv
source venv/bin/activate

# Instalar dependências
pip install -r requirements.txt

# Abrir notebook
jupyter notebook modelo_dengue_final_optuna.ipynb
```

### Método 3: Script Python

```bash
# Executar treinamento completo via script
python treinar_modelo_final.py
```

### ⚠️ IMPORTANTE: Ordem de Execução

O notebook tem **células com dependências**. Execute em ordem:

1. **Células 0-42**: Pré-processamento + Otimização Optuna
2. **Célula 43**: ⭐ Treinamento Final (cria `modelo_otimizado`)
3. **Célula 44**: ⭐ Modelos de Comparação (cria `predictions`, `probabilities`)
4. **Células 28, 36-39**: Visualizações (SHAP, ROC, PR, etc)

💡 **Dica**: Use `Cell → Run All` para garantir execução correta!

### 3. Explorar Resultados

Após a execução, os seguintes arquivos serão gerados:

- ✅ **modelo_reglog_otimizado.pkl**: Modelo final treinado
- ✅ **scaler_final.pkl**: Normalizador (StandardScaler)
- ✅ **config_modelo.json**: Métricas e hiperparâmetros
- ✅ **features_selecionadas.txt**: Lista de 14 features
- ✅ **optuna_study_logreg.pkl**: Estudo de otimização
- ✅ **Visualizações PNG**: SHAP, ROC, PR, Confusion Matrix

---

## 📊 Pipeline de Análise

### 1. EDA (Exploratory Data Analysis)
- Análise temporal (casos por ano/mês)
- Distribuição demográfica
- Análise de sintomas
- Identificação de valores faltantes
- Correlações

### 2. Feature Engineering
- **SEVERITY_SCORE**: Score de severidade clínica
- **COMORBIDADE_SCORE**: Soma de comorbidades
- **TEM_COMORBIDADE**: Flag binária
- **DIAS_SINTOMA_NOTIFIC_TEMP**: Tempo entre sintomas e notificação (temporal)
- **QTD_IGNORADOS**: Quantidade de campos ignorados

### 3. Preparação dos Dados
- Tratamento de valores "IGNORADO" → binário
- One-hot encoding (raça, etc)
- Normalização (StandardScaler)
- Split estratificado 80/20
- **Balanceamento com SMOTE**

### 4. Seleção de Features
- **Critérios combinados**:
  1. Feature Importance (Random Forest)
  2. Correlação com target
  3. Chi-Square (significância estatística)
- **Redução**: ~60 features → 14 features selecionadas

### 5. Modelagem
- Regressão Logística (L1 regularization)
- **Optuna**: 50 trials, 5-fold CV
- Objetivo: Maximizar Recall

### 6. Avaliação Clínica
- Métricas: Sensitivity, Specificity, PPV, NPV, F1, AUC
- Matriz de confusão (análise de FN e FP)
- Curvas ROC e Precision-Recall
- Análise de erros (Falsos Negativos/Positivos)

### 7. Interpretabilidade
- **SHAP values** (global feature importance)
- Análise de features mais importantes
- Visualizações de importância

### 8. Comparação com Modelos Baseline
- Random Forest (sem tunagem)
- XGBoost (sem tunagem)
- CatBoost (sem tunagem)

---

## 🎯 Resultados Obtidos

### Métricas do Modelo Final

| Métrica | Valor | Interpretação |
|---------|-------|---------------|
| **Sensitivity (Recall)** | 0.4364 | 43.64% dos casos graves detectados |
| **Specificity** | 0.7402 | 74.02% dos não-casos identificados |
| **PPV (Precision)** | 0.0245 | 2.45% dos alertas são verdadeiros |
| **NPV** | 0.9887 | 98.87% de confiança em negativos ✅ |
| **F1-Score** | 0.0464 | Score F1 baixo |
| **ROC-AUC** | 0.6295 | Capacidade discriminativa moderada |

### Análise de Erros

- **Falsos Negativos (FN)**: 31 casos (pacientes graves NÃO detectados)
- **Falsos Positivos (FP)**: 955 casos (alertas desnecessários)

### ⚠️ Observações Importantes

1. **Recall abaixo do esperado** (43.6% vs meta de 85%)
   - Modelo conservador, detecta menos da metade dos casos graves

2. **NPV excelente** (98.87%)
   - Quando o modelo diz "não hospitalizar", tem alta confiança

3. **PPV muito baixo** (2.45%)
   - Para cada 100 alertas, apenas 2-3 são verdadeiros

4. **Trade-off crítico**:
   - Alta taxa de FN = risco clínico (pacientes graves não detectados)
   - Alta taxa de FP = sobrecarga do sistema de saúde

### 💡 Interpretação Clínica

O modelo atual **não atinge os critérios clínicos mínimos** (Recall ≥ 0.85). Possíveis causas:

- Dataset altamente desbalanceado (1.17% de hospitalizações)
- Features selecionadas podem não capturar sinais de alarme críticos
- Regularização L1 muito forte (C=0.0028) → modelo conservador
- Ausência dos 5 sintomas principais OMS como features diretas

---

## 🔍 Interpretação de Resultados

### Output Real do Modelo

```
🏆 MODELO: Regressão Logística (Optuna)

📊 MÉTRICAS:
   - Sensitivity (Recall): 0.4364 ⚠️  (43.64% dos casos detectados)
   - Specificity:          0.7402 (74.02% dos não-casos identificados)
   - PPV (Precision):      0.0245 (2.45% dos alertas são verdadeiros)
   - NPV:                  0.9887 ✅ (98.87% de confiança em negativos)
   - ROC-AUC:              0.6295

⚠️ ANÁLISE DE ERROS:
   - Falsos Negativos: 31 pacientes (56.36% dos positivos reais) 🚨
   - Falsos Positivos: 955 alertas desnecessários

💡 INTERPRETAÇÃO:
   - O modelo captura apenas 44% dos casos graves 🚨
   - 56% dos casos graves NÃO são detectados (FN alto)
   - Para cada 100 alertas, apenas 2-3 são verdadeiros (PPV muito baixo)
   - Quando o modelo diz "não hospitalizar", tem 98.9% de confiança (NPV ✅)
```

### Trade-off Atual

- ❌ **Recall muito baixo**: Menos da metade dos casos graves são detectados
- ❌ **PPV crítico**: 97.5% dos alertas são falsos
- ✅ **NPV excelente**: Alta confiança em resultados negativos
- ⚠️  **Risco clínico**: 31 pacientes graves não detectados

---

## 🏥 Features Mais Importantes (SHAP Analysis)

Baseado em análise SHAP, as features mais importantes são:

1. **DIAS_SINTOMA_NOTIFIC_TEMP** - Tempo entre sintomas e notificação (temporal)
2. **TRIMESTRE** - Trimestre do ano (sazonalidade)
3. **MES** - Mês da notificação
4. **DIAS_SINTOMA_NOTIFIC** - Dias entre sintomas e notificação
5. **TEM_COMORBIDADE** - Presença de comorbidade
6. **NU_ANO** - Ano da notificação
7. **QTD_IGNORADOS** - Quantidade de campos ignorados
8. **SEVERITY_SCORE** - Score de severidade
9. **IDADE** - Idade do paciente
10. **COMORBIDADE_SCORE** - Score de comorbidades

### 💡 Insights

- **Predominância temporal**: 6 das 10 features mais importantes são temporais
- **Comorbidades importantes**: TEM_COMORBIDADE e COMORBIDADE_SCORE aparecem
- **Ausência de sintomas diretos**: FEBRE, VOMITO, MIALGIA não foram selecionadas
- **QTD_IGNORADOS**: Proxy de completude dos dados é relevante

---

## 📈 Próximos Passos e Melhorias

### 🔧 Melhorias Prioritárias no Modelo

1. **Re-incluir sintomas clínicos OMS**
   - FEBRE, VOMITO, MIALGIA, CEFALEIA, EXANTEMA
   - Sinais de alarme: PETEQUIA, DOR_ABD

2. **Ajustar threshold de predição**
   - Reduzir de 0.5 para 0.3-0.4 → aumentar Recall

3. **Testar class_weight='balanced'**
   - Combinar SMOTE + class_weight

4. **Explorar outros modelos**
   - XGBoost otimizado (melhor para dados desbalanceados)
   - Ensemble (Logistic + XGBoost + Random Forest)

5. **Feature engineering adicional**
   - Interações (IDADE × COMORBIDADE)
   - Sintomas combinados (VOMITO + PETEQUIA)

### 🚀 Deployment (Após Atingir Recall ≥ 0.85)

1. **Dashboard Streamlit**
   - Interface para médicos
   - Upload de casos
   - Explicabilidade SHAP

2. **API REST (FastAPI)**
   - Endpoint `/predict`
   - Integração com sistemas de saúde

3. **Monitoramento**
   - Drift detection
   - Retraining automático

---

## ⚠️ Limitações e Considerações

### Limitações

1. **Recall insuficiente** (43.6% << 85%): Modelo não detecta maioria dos casos graves
2. **Dados históricos**: Modelo treinado em dados de 2013-2025
3. **Região específica**: Sertãozinho, SP
4. **Desbalanceamento extremo**: Apenas 1.17% de hospitalizações
5. **Valores ignorados**: Muitos dados clínicos "IGNORADO"
6. **Features temporais dominantes**: Sintomas clínicos não foram selecionados
7. **Regularização muito forte**: C=0.0028 → modelo conservador demais

### Considerações Éticas

- ✅ **Não substituir decisão médica**: Ferramenta de apoio, não diagnóstico final
- ✅ **Fairness**: Validar desempenho em diferentes subgrupos (gênero, idade)
- ✅ **Explicabilidade**: Usar SHAP para explicar decisões
- ✅ **Privacidade**: HIPAA/LGPD compliance
- ✅ **Monitoring**: Detectar drift e viés

---

## 📚 Referências

### Literatura Médica

1. WHO (2009). "Dengue: Guidelines for diagnosis, treatment, prevention and control"
2. Ministério da Saúde (2016). "Dengue: diagnóstico e manejo clínico - adulto e criança"

### Machine Learning

1. Kuhn, M., & Johnson, K. (2013). "Applied Predictive Modeling"
2. Molnar, C. (2022). "Interpretable Machine Learning"
3. Chawla et al. (2002). "SMOTE: Synthetic Minority Over-sampling Technique"

---

## 👥 Equipe

**Desenvolvido com sistema de orquestração multi-agent:**

- **@orchestrator**: Coordenação geral
- **@healthcare_ml_specialist**: Modelagem e métricas clínicas
- **@data_engineer**: ETL e feature engineering
- **@ml_engineer**: Treinamento e otimização

---

## 📞 Suporte

Para dúvidas ou sugestões:

- 📧 Email: [seu-email]
- 🐛 Issues: [GitHub Issues]
- 📖 Docs: [Link para documentação]

---

## 🎓 Aprendizados

### O que Funcionou ✅
- Pipeline completo de ML implementado
- Otimização automática com Optuna
- Seleção rigorosa de features (60 → 14)
- NPV excelente (98.87%)
- Código modular e reproduzível

### O que Precisa Melhorar ⚠️
- Recall crítico (43.6% vs meta de 85%)
- Sintomas clínicos OMS não foram selecionados
- Regularização L1 muito forte
- Threshold de classificação fixo (0.5)

### Lições Aprendidas 💡
1. **Dados desbalanceados são difíceis**: 1.17% de positivos é extremo
2. **Métricas clínicas ≠ Métricas de ML**: Accuracy não é suficiente em saúde
3. **Feature engineering importa**: Features temporais dominaram
4. **Validação médica essencial**: Modelo precisa validação com especialistas

---

**⚠️  Modelo em desenvolvimento. NÃO usar em produção sem validação clínica!**

**Em saúde, Recall > tudo. É melhor errar por excesso de cuidado!** 🏥
