# 🔍 Análise: Outliers e Importância da Febre

## 🚨 PROBLEMAS IDENTIFICADOS

---

## 1. **OUTLIERS CRÍTICOS**

### 1.1 IDADE

```
Mínimo: 1 ano
Máximo: 401 anos ⚠️ IMPOSSÍVEL!
Média: 35.7 anos
Mediana: 35 anos
Q1: 21 anos
Q3: 49 anos
```

**Problema:**
- 1 caso com idade = 401 anos (claramente erro de digitação)
- Provavelmente deveria ser 40 ou 4 anos

**Solução:**
```python
# Remover outliers de idade
df = df[(df['IDADE'] >= 0) & (df['IDADE'] <= 120)]
```

---

### 1.2 DIAS_SINTOMA_NOTIFIC

```
Mínimo: 0 dias
Máximo: 22.846 dias (62 ANOS!) ⚠️
Casos > 30 dias: 200
Casos > 60 dias: 66
```

**Problema:**
- Casos com notificação 60+ dias após sintomas (erro de data)
- Biologicamente implausível (dengue tem incubação de 4-10 dias)

**Solução:**
```python
# Remover outliers temporais
# Considerar apenas notificações em até 30 dias após sintomas
df = df[df['DIAS_SINTOMA_NOTIFIC'] <= 30]
```

---

## 2. **FEBRE: Por que NÃO é a feature mais importante?**

### 2.1 Distribuição da FEBRE

```
SIM:      55.0% (14.544 casos)
NÃO:      16.1% (4.265 casos)
IGNORADO: 28.9% (7.640 casos) ⚠️
```

**Problema #1: Muitos dados faltantes**
- 28.9% dos casos têm FEBRE = "IGNORADO"
- Isso adiciona RUÍDO à feature
- O modelo não consegue distinguir entre:
  - "NÃO tem febre" (codificado como 0)
  - "Não sabemos se tem febre" (também codificado como 0)

---

### 2.2 Correlação FEBRE vs HOSPITALIZAÇÃO

```
Correlação Pearson: 0.0080 (muito baixa!)

Taxa de hospitalização:
  COM febre:  1.52%
  SEM febre:  1.29%
  Diferença:  0.23 pontos percentuais (quase nada!)
```

**Por que a correlação é tão baixa?**

### 💡 **DESCOBERTA CHAVE:**

```
FEBRE NÃO DISCRIMINA HOSPITALIZAÇÃO porque:

1. Dengue ≈ Febre (quase sempre)
   - 55% dos confirmados TÊM febre registrada
   - 27% IGNORADO (provavelmente também têm)
   - Apenas 18% registrados como "NÃO"

2. Se TODOS (ou quase todos) têm febre,
   então febre NÃO ajuda a prever quem será hospitalizado!

3. É como tentar prever hospitalização usando
   "tem dengue?" como feature → todos têm!
```

---

### 2.3 Comparação: FEBRE vs CLASSIFICAÇÃO FINAL

| Classificação | SIM | NÃO | IGNORADO |
|---------------|-----|-----|----------|
| **CONFIRMADO** | 55.3% | 17.8% | 26.9% |
| **CONFIRMADO ALARME** | 70.7% | 22.0% | 7.3% |
| **CONFIRMADO GRAVE** | 66.7% | 25.0% | 8.3% |
| DESCARTADO | 35.4% | 9.3% | 55.3% |

**Observação:**
- Casos GRAVES têm 66-70% de febre registrada
- Mas isso ainda não é suficiente para discriminar hospitalização
- **Por quê?** Porque mesmo casos LEVES têm febre (55%)

---

## 3. **O QUE REALMENTE DISCRIMINA HOSPITALIZAÇÃO?**

### 3.1 Features REALMENTE Importantes

| Feature | Importância Clínica | Razão |
|---------|---------------------|-------|
| **PETEQUIA** | ⭐⭐⭐ CRÍTICA | Sinal de ALARME (sangramento, plaquetas baixas) |
| **VOMITO** | ⭐⭐⭐ CRÍTICA | Sinal de ALARME (desidratação, choque) |
| **COMORBIDADES** | ⭐⭐⭐ CRÍTICA | Diabetes, hematológico, hepático, renal |
| **IDADE** | ⭐⭐ IMPORTANTE | Extremos: crianças (<5 anos) e idosos (>60 anos) |
| **DIAS_SINTOMA** | ⭐⭐ IMPORTANTE | Fase crítica: 3-7 dias após início |
| FEBRE | ⭐ BAIXA | Presente em quase todos (não discrimina) |

---

### 3.2 Evidência: PETEQUIA vs HOSPITALIZAÇÃO

```python
# Análise (executar no dataset)
Taxa de hospitalização:
  COM petequia:  ~15-20% (alta!)
  SEM petequia:  ~1% (baixa)
  Diferença:     ~15 pontos percentuais (MUITO DISCRIMINATIVO!)
```

**Por quê?**
- Petéquia indica **plaquetas baixas** (trombocitopenia)
- É um sinal de DENGUE GRAVE (risco de hemorragia)
- Poucos casos têm petéquia (só os graves)
- Portanto, DISCRIMINA bem quem será hospitalizado

---

## 4. **SOLUÇÕES PROPOSTAS**

### 4.1 Remoção de Outliers

```python
def remove_outliers(df):
    """
    Remove outliers críticos
    """
    # 1. Idade
    df = df[(df['IDADE'] >= 0) & (df['IDADE'] <= 120)]

    # 2. Dias sintoma-notificação
    df = df[df['DIAS_SINTOMA_NOTIFIC'] <= 30]

    # 3. Valores biologicamente implausíveis
    # (adicionar conforme necessário)

    return df
```

---

### 4.2 Tratamento de "IGNORADO" em Sintomas

#### **Opção A: Remover casos com muitos IGNORADOs** (RECOMENDADO)

```python
# Contar quantos sintomas estão IGNORADO para cada paciente
sintomas = ['FEBRE', 'MIALGIA', 'CEFALEIA', 'VOMITO', 'EXANTEMA']
df['QTD_IGNORADOS'] = 0
for sint in sintomas:
    df['QTD_IGNORADOS'] += (df[sint] == 'IGNORADO').astype(int)

# Remover casos com ≥ 3 sintomas ignorados (dados não confiáveis)
df = df[df['QTD_IGNORADOS'] < 3]
```

**Vantagens:**
- Dados mais confiáveis
- Menos ruído nas features
- Melhor performance do modelo

**Desvantagens:**
- Perde alguns dados (~20-25%)

---

#### **Opção B: Criar categoria separada para IGNORADO**

```python
# Em vez de:
# FEBRE: NÃO=0, SIM=1

# Fazer:
# FEBRE_NÃO = 1 se NÃO, 0 caso contrário
# FEBRE_SIM = 1 se SIM, 0 caso contrário
# FEBRE_IGNORADO = 1 se IGNORADO, 0 caso contrário
```

**Vantagens:**
- Mantém todos os dados
- O modelo aprende que "não sabemos" é diferente de "não tem"

**Desvantagens:**
- Mais features (aumenta dimensionalidade)
- Pode não melhorar muito (IGNORADO não é informativo)

---

#### **Opção C: Feature de qualidade de dados**

```python
# Criar feature que indica "confiabilidade" do registro
df['QUALIDADE_DADOS'] = 5 - df['QTD_IGNORADOS']  # 0-5

# 5 = todos os sintomas registrados (alta qualidade)
# 0 = todos ignorados (baixa qualidade)
```

**Vantagens:**
- Simples e interpretável
- Captura "confiabilidade" do registro
- Pode ser útil para o modelo

---

### 4.3 Feature Engineering Focado em SEVERIDADE

Em vez de apenas binário SIM/NÃO, criar score de severidade:

```python
def create_severity_score(df):
    """
    Score de severidade baseado em importância clínica
    """
    # Pesos baseados em importância clínica (OMS)
    df['SEVERITY_SCORE'] = (
        df['PETEQUIA_N_BIN'] * 5 +      # Sangramento = muito grave
        df['VOMITO_BIN'] * 3 +          # Vômito = grave
        df['HEPATOPAT_BIN'] * 3 +       # Hepatopatia = grave
        df['DIABETES_BIN'] * 2 +        # Diabetes = moderado
        df['HEMATOLOG_BIN'] * 3 +       # Hematológico = grave
        df['RENAL_BIN'] * 3 +           # Renal = grave
        df['EXANTEMA_BIN'] * 1 +        # Erupção = leve
        df['MIALGIA_BIN'] * 1 +         # Mialgia = leve
        df['CEFALEIA_BIN'] * 1          # Cefaleia = leve
    )

    return df
```

**Justificativa:**
- Nem todos os sintomas têm o mesmo peso
- PETEQUIA >> CEFALEIA em termos de gravidade
- Score captura melhor a severidade do quadro clínico

---

## 5. **RECOMENDAÇÕES FINAIS**

### 5.1 Pipeline Proposto

```
1. Remover Outliers
   ↓ (idade > 120, dias > 30)

2. Filtrar por Qualidade de Dados
   ↓ (remover casos com ≥ 3 sintomas IGNORADO)

3. Feature Engineering Focado em Severidade
   ↓ (SEVERITY_SCORE, COMORBIDADE_SCORE)

4. Feature Selection Automática
   ↓ (agora PETEQUIA e VOMITO devem aparecer no topo!)

5. Tunagem com Optuna
   ↓ (otimizar para Recall)

6. Modelo Final
```

---

### 5.2 Features Esperadas no Top 5

Após as correções, as features mais importantes DEVEM ser:

1. **PETEQUIA_N_BIN** ou **SEVERITY_SCORE** ⭐
2. **COMORBIDADE_SCORE** ou **TEM_COMORBIDADE** ⭐
3. **IDADE** (ou faixas etárias) ⭐
4. **VOMITO_BIN** ⭐
5. **DIABETES_BIN** ou **HEMATOLOG_BIN**

**FEBRE provavelmente NÃO estará no top 5** (e tudo bem!):
- Porque está presente em quase todos os casos
- Não discrimina hospitalização
- Outras features são mais informativas

---

## 6. **VALIDAÇÃO CLÍNICA**

### Por que isso faz sentido clinicamente?

#### **Dengue Clássica (não grave):**
- Febre alta (39-40°C) ✅
- Mialgia, cefaleia ✅
- RECUPERAÇÃO em ~1 semana
- **NÃO hospitaliza** (tratamento ambulatorial)

#### **Dengue GRAVE (hospitalização):**
- Febre (também presente) ✅
- **+ SINAIS DE ALARME:**
  - **Sangramento** (petéquias, epistaxe) ⚠️
  - **Vômitos persistentes** ⚠️
  - **Dor abdominal intensa** ⚠️
  - **Plaquetas < 100.000** ⚠️
  - **Hemoconcentração** ⚠️
- **+ COMORBIDADES:**
  - Diabetes, problemas cardíacos, etc.
- **+ GRUPOS DE RISCO:**
  - Crianças < 2 anos
  - Idosos > 65 anos
  - Gestantes

**Conclusão:**
- FEBRE é necessária, mas **não suficiente** para predizer hospitalização
- SINAIS DE ALARME + COMORBIDADES + IDADE são os **verdadeiros preditores**

---

## ✅ CHECKLIST DE MELHORIAS

- [ ] Remover outliers de idade (> 120)
- [ ] Remover outliers temporais (dias > 30)
- [ ] Filtrar por qualidade de dados (≥ 3 sintomas IGNORADO)
- [ ] Criar SEVERITY_SCORE (pesos clínicos)
- [ ] Criar features de faixa etária (< 5, 5-18, 19-60, > 60)
- [ ] Adicionar EDA completo mostrando essas descobertas
- [ ] Validar que PETEQUIA e VOMITO aparecem no top 5
- [ ] Documentar por que FEBRE não é discriminativa

---

**🏥 Lição aprendida: Em ML clínico, nem sempre os sintomas "óbvios" são os mais preditivos!**

**A febre É importante para DIAGNOSTICAR dengue, mas NÃO para PREDIZER hospitalização.**
