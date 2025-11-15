# 🩺 PI4v10: Modelo Preditivo de Hospitalização por Dengue (Sertãozinho-SP)

Este repositório contém o projeto final do **Projeto Integrador IV (PI4)** da UNIVESP, focado na análise da série histórica de notificações de Dengue em Sertãozinho-SP e no desenvolvimento de um modelo preditivo para a probabilidade de hospitalização.

O projeto utiliza uma arquitetura web baseada em **Flask** para servir tanto a análise exploratória de dados (EDA) quanto o modelo de Machine Learning.

## 🎯 Objetivo Principal

Desenvolver um modelo de Machine Learning robusto e bem calibrado para prever a **probabilidade de hospitalização** de pacientes com dengue, utilizando dados do SINAN DENGUE (2000-2025) de Sertãozinho-SP.

## 🧠 O Modelo Preditivo (v11 - Retreinado e Calibrado)

Após uma revisão completa, o modelo foi retreinado e calibrado para garantir a validade das probabilidades de predição (entre 0% e 100%).

| Característica | Detalhe |
| :--- | :--- |
| **Algoritmo** | Regressão Logística Calibrada (Platt Scaling) |
| **Features de Input** | 5 features (FEBRE, MIALGIA, CEFALEIA, VOMITO, EXANTEMA) |
| **Features Removidas** | IDADE e CS_SEXO (Identificadas como de relevância negligenciável) |
| **Performance (Teste)** | ROC-AUC: 0.9774 |
| **Calibração** | Brier Score: 0.0193 (Muito bem calibrado) |
| **Probabilidades** | Estritamente entre 0% e 100% |

A escolha das 5 features foi baseada em uma análise rigorosa de *Feature Importance*, onde os sintomas primários (Febre, Mialgia, Cefaleia) demonstraram a maior correlação com a hospitalização.

## 🛠️ Estrutura do Repositório

| Arquivo/Diretório | Descrição |
| :--- | :--- |
| `app.py` | Aplicação Flask principal, contendo as rotas para o front-end e o endpoint `/api/predict`. |
| `wsgi.py` | Ponto de entrada para o servidor Gunicorn (deploy). |
| `requirements.txt` | Dependências Python necessárias (Flask, Pandas, Scikit-learn, etc.). |
| `modelo_reglog_pi4_retrained.pkl` | O modelo de Regressão Logística Calibrada treinado com as 5 features. |
| `templates/` | Contém os arquivos HTML (`index.html`, `dashboard.html`). |
| `static/js/main.js` | Lógica JavaScript para interações do front-end, incluindo o formulário de predição. |
| `REVISAO_MODELO_RELATORIO.txt` | Relatório detalhado da análise de features e retreinamento do modelo. |
| `CORRECTION_SUMMARY.md` | Sumário das correções de sincronização Front-end/Back-end. |

## ⚙️ Correções Recentes (Sincronização Front-end/Back-end)

A versão atual (`v11`) foi submetida a uma correção crítica para sincronizar a interface do usuário com o modelo preditivo, que utiliza apenas 5 features.

| Arquivo | Correção Realizada |
| :--- | :--- |
| `templates/index.html` | **Remoção dos campos Idade e Sexo** do formulário de predição. Atualização do texto da metodologia para **5 features**. |
| `static/js/main.js` | Ajuste na função `initializePredictionForm` para coletar e enviar ao `/api/predict` **apenas** os 5 valores de sintomas. |
| `app.py` | Correção de um erro de indentação que impedia o deploy correto. |

Com estas correções, a aplicação garante que a interface do usuário reflita de forma transparente e precisa os dados de entrada esperados pelo modelo de Machine Learning.

## 🚀 Como Executar Localmente

1.  **Clonar o Repositório:**
    ```bash
    git clone https://github.com/frischlander/pi4v10
    cd pi4v10
    ```
2.  **Criar e Ativar Ambiente Virtual:**
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```
3.  **Instalar Dependências:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Executar a Aplicação Flask:**
    ```bash
    python app.py
    ```
    A aplicação estará disponível em `http://127.0.0.1:5000`.

---
*Desenvolvido para o Projeto Integrador IV - UNIVESP.*
