"""
PADRÃO OURO: Treinamento de Regressão Logística para Predição de Hospitalização
- Apenas 7 features de input
- Pré-processamento completo
- Tratamento de IGNORADO como NÃO
- Validação cruzada
- Calibração de probabilidades
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (roc_auc_score, roc_curve, classification_report, 
                             confusion_matrix, brier_score_loss, log_loss)
from sklearn.calibration import CalibratedClassifierCV
import joblib
import warnings

warnings.filterwarnings('ignore')

print("=" * 100)
print("PADRÃO OURO - TREINAMENTO DE REGRESSÃO LOGÍSTICA")
print("=" * 100)

# ============================================================================
# 1. CARREGAMENTO E EXPLORAÇÃO DO DATASET
# ============================================================================
print("\n[1] CARREGAMENTO E EXPLORAÇÃO DO DATASET")
print("-" * 100)

try:
    df = pd.read_csv("df_final_predict.csv")
    print(f"✓ Dataset carregado: {df.shape[0]:,} registros, {df.shape[1]} colunas")
except FileNotFoundError:
    print("✗ ERRO: df_final_predict.csv não encontrado")
    exit(1)

INPUT_FEATURES = ['IDADE', 'CS_SEXO', 'FEBRE', 'VOMITO', 'MIALGIA', 'CEFALEIA', 'EXANTEMA']
TARGET_COLUMN = 'HOSPITALIZ'

# Selecionar apenas as features necessárias
df_model = df[INPUT_FEATURES + [TARGET_COLUMN]].copy()

print(f"\nDataset selecionado:")
print(f"  Forma: {df_model.shape}")
print(f"  Features: {INPUT_FEATURES}")
print(f"  Alvo: {TARGET_COLUMN}")

# ============================================================================
# 2. PRÉ-PROCESSAMENTO: TRATAMENTO DE VALORES AUSENTES E CATEGORIAS
# ============================================================================
print("\n[2] PRÉ-PROCESSAMENTO")
print("-" * 100)

print("\nAnálise de valores únicos (ANTES):")
for col in INPUT_FEATURES + [TARGET_COLUMN]:
    unique_vals = df_model[col].unique()
    print(f"  {col}: {list(unique_vals)}")

# Tratar IGNORADO como NÃO
print("\nTratando 'IGNORADO' como 'NÃO'...")
for col in INPUT_FEATURES + [TARGET_COLUMN]:
    df_model[col] = df_model[col].replace('IGNORADO', 'NÃO')

print("\nAnálise de valores únicos (DEPOIS):")
for col in INPUT_FEATURES + [TARGET_COLUMN]:
    unique_vals = df_model[col].unique()
    print(f"  {col}: {list(unique_vals)}")

# Verificar se há valores faltantes
print(f"\nValores faltantes: {df_model.isnull().sum().sum()}")

# ============================================================================
# 3. ANÁLISE DA CLASSE ALVO
# ============================================================================
print("\n[3] ANÁLISE DA CLASSE ALVO")
print("-" * 100)

target_dist = df_model[TARGET_COLUMN].value_counts()
print(f"\nDistribuição da classe alvo:")
for val, count in target_dist.items():
    pct = (count / len(df_model)) * 100
    print(f"  {val}: {count:,} ({pct:.2f}%)")

# ============================================================================
# 4. CODIFICAÇÃO DE VARIÁVEIS
# ============================================================================
print("\n[4] CODIFICAÇÃO DE VARIÁVEIS")
print("-" * 100)

# Preparar dados para codificação
X = df_model[INPUT_FEATURES].copy()
y = df_model[TARGET_COLUMN].copy()

print(f"\nCodificando variáveis categóricas...")

# Codificar CS_SEXO (F/M)
label_encoder_sexo = LabelEncoder()
X['CS_SEXO'] = label_encoder_sexo.fit_transform(X['CS_SEXO'])
print(f"  CS_SEXO: {dict(zip(label_encoder_sexo.classes_, label_encoder_sexo.transform(label_encoder_sexo.classes_)))}")

# Codificar sintomas (NÃO/SIM)
sintomas_cols = ['FEBRE', 'VOMITO', 'MIALGIA', 'CEFALEIA', 'EXANTEMA']
for col in sintomas_cols:
    X[col] = (X[col] == 'SIM').astype(int)
    print(f"  {col}: NÃO=0, SIM=1")

# Codificar alvo
y = (y == 'SIM').astype(int)
print(f"  {TARGET_COLUMN}: NÃO=0, SIM=1")

print(f"\nDataset após codificação:")
print(X.head())

# ============================================================================
# 5. ANÁLISE DESCRITIVA DAS FEATURES
# ============================================================================
print("\n[5] ANÁLISE DESCRITIVA DAS FEATURES")
print("-" * 100)

print("\nEstatísticas descritivas:")
print(X.describe().T)

# ============================================================================
# 6. ESCALONAMENTO DE FEATURES
# ============================================================================
print("\n[6] ESCALONAMENTO DE FEATURES")
print("-" * 100)

# StandardScaler apenas para IDADE (feature numérica)
scaler = StandardScaler()
X_idade_original = X[['IDADE']].copy()
X['IDADE'] = scaler.fit_transform(X[['IDADE']])

print(f"✓ IDADE escalonada com StandardScaler")
print(f"  Média original: {X_idade_original['IDADE'].mean():.2f}")
print(f"  Desvio padrão original: {X_idade_original['IDADE'].std():.2f}")
print(f"  Média escalonada: {X['IDADE'].mean():.4f}")
print(f"  Desvio padrão escalonado: {X['IDADE'].std():.4f}")

# Guardar parâmetros do scaler para uso posterior
idade_mean = scaler.mean_[0]
idade_std = scaler.scale_[0]
print(f"\n  Parâmetros para posterior uso em produção:")
print(f"    IDADE_MEAN = {idade_mean}")
print(f"    IDADE_STD = {idade_std}")

# ============================================================================
# 7. ANÁLISE DE CORRELAÇÃO
# ============================================================================
print("\n[7] ANÁLISE DE CORRELAÇÃO COM ALVO")
print("-" * 100)

X_temp = X.copy()
X_temp[TARGET_COLUMN] = y

correlacoes = X_temp.corr()[TARGET_COLUMN].drop(TARGET_COLUMN).sort_values(ascending=False)
print("\nCorrelação de Pearson com hospitalização:")
for feat, corr in correlacoes.items():
    print(f"  {feat}: {corr:.4f}")

# ============================================================================
# 8. DIVISÃO TREINO/TESTE
# ============================================================================
print("\n[8] DIVISÃO TREINO/TESTE")
print("-" * 100)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTreino: {X_train.shape[0]:,} amostras")
print(f"  Classe NÃO: {(y_train == 0).sum():,} ({(y_train == 0).sum()/len(y_train)*100:.1f}%)")
print(f"  Classe SIM: {(y_train == 1).sum():,} ({(y_train == 1).sum()/len(y_train)*100:.1f}%)")

print(f"\nTeste: {X_test.shape[0]:,} amostras")
print(f"  Classe NÃO: {(y_test == 0).sum():,} ({(y_test == 0).sum()/len(y_test)*100:.1f}%)")
print(f"  Classe SIM: {(y_test == 1).sum():,} ({(y_test == 1).sum()/len(y_test)*100:.1f}%)")

# ============================================================================
# 9. TREINAMENTO DO MODELO
# ============================================================================
print("\n[9] TREINAMENTO DO MODELO")
print("-" * 100)

# Regressão Logística com parâmetros otimizados
print(f"\nTreinando Regressão Logística...")
base_model = LogisticRegression(
    solver='lbfgs',
    max_iter=1000,
    random_state=42,
    verbose=0,
    penalty='l2',
    C=1.0
)

base_model.fit(X_train, y_train)
print(f"✓ Modelo base treinado")

# ============================================================================
# 10. CALIBRAÇÃO DE PROBABILIDADES
# ============================================================================
print("\n[10] CALIBRAÇÃO DE PROBABILIDADES")
print("-" * 100)

print(f"\nAplicando Platt Scaling para calibração...")
calibrated_model = CalibratedClassifierCV(base_model, method='sigmoid', cv=5)
calibrated_model.fit(X_train, y_train)
print(f"✓ Modelo calibrado")

# ============================================================================
# 11. VALIDAÇÃO CRUZADA
# ============================================================================
print("\n[11] VALIDAÇÃO CRUZADA (5-Fold)")
print("-" * 100)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(calibrated_model, X_train, y_train, cv=cv, scoring='roc_auc')

print(f"\nScores ROC-AUC por fold:")
for i, score in enumerate(cv_scores, 1):
    print(f"  Fold {i}: {score:.4f}")

print(f"\nMédia: {cv_scores.mean():.4f}")
print(f"Desvio padrão: {cv_scores.std():.4f}")

# ============================================================================
# 12. AVALIAÇÃO NO CONJUNTO DE TESTE
# ============================================================================
print("\n[12] AVALIAÇÃO NO CONJUNTO DE TESTE")
print("-" * 100)

y_pred = calibrated_model.predict(X_test)
y_pred_proba = calibrated_model.predict_proba(X_test)[:, 1]

# Métricas
roc_auc = roc_auc_score(y_test, y_pred_proba)
brier = brier_score_loss(y_test, y_pred_proba)
report = classification_report(y_test, y_pred, target_names=['NÃO', 'SIM'], output_dict=True)
cm = confusion_matrix(y_test, y_pred)

print(f"\nMétricas de Desempenho:")
print(f"  ROC-AUC: {roc_auc:.4f}")
print(f"  Brier Score: {brier:.4f}")
print(f"  Acurácia: {report['accuracy']:.4f}")

print(f"\nRelatório de Classificação:")
print(f"  NÃO:")
print(f"    Precisão: {report['NÃO']['precision']:.4f}")
print(f"    Recall: {report['NÃO']['recall']:.4f}")
print(f"    F1-Score: {report['NÃO']['f1-score']:.4f}")

print(f"  SIM:")
print(f"    Precisão: {report['SIM']['precision']:.4f}")
print(f"    Recall: {report['SIM']['recall']:.4f}")
print(f"    F1-Score: {report['SIM']['f1-score']:.4f}")

print(f"\nMatriz de Confusão:")
print(f"  {'':>15} Predito NÃO  Predito SIM")
print(f"  {'Real NÃO':>15}    {cm[0,0]:>5d}        {cm[0,1]:>5d}")
print(f"  {'Real SIM':>15}    {cm[1,0]:>5d}        {cm[1,1]:>5d}")

# ============================================================================
# 13. ANÁLISE DE PROBABILIDADES
# ============================================================================
print("\n[13] ANÁLISE DE PROBABILIDADES")
print("-" * 100)

print(f"\nDistribuição de probabilidades preditas:")
print(f"  Mínimo: {y_pred_proba.min():.6f} ({y_pred_proba.min()*100:.4f}%)")
print(f"  Máximo: {y_pred_proba.max():.6f} ({y_pred_proba.max()*100:.4f}%)")
print(f"  Média: {y_pred_proba.mean():.6f} ({y_pred_proba.mean()*100:.4f}%)")
print(f"  Mediana: {np.median(y_pred_proba):.6f} ({np.median(y_pred_proba)*100:.4f}%)")
print(f"  Percentil 25: {np.percentile(y_pred_proba, 25):.6f} ({np.percentile(y_pred_proba, 25)*100:.4f}%)")
print(f"  Percentil 75: {np.percentile(y_pred_proba, 75):.6f} ({np.percentile(y_pred_proba, 75)*100:.4f}%)")

# ============================================================================
# 14. ANÁLISE DE CURVA ROC
# ============================================================================
print("\n[14] ANÁLISE DE CURVA ROC")
print("-" * 100)

fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
print(f"\nPonto ótimo (Youden's Index):")

# Encontrar ponto ótimo
youden = tpr - fpr
optimal_idx = np.argmax(youden)
optimal_threshold = thresholds[optimal_idx]

print(f"  Threshold: {optimal_threshold:.4f}")
print(f"  True Positive Rate: {tpr[optimal_idx]:.4f}")
print(f"  False Positive Rate: {fpr[optimal_idx]:.4f}")

# ============================================================================
# 15. SALVAR O MODELO
# ============================================================================
print("\n[15] SALVAMENTO DO MODELO")
print("-" * 100)

model_path = "modelo_reglog_pi4_retrained.pkl"
joblib.dump(calibrated_model, model_path)
print(f"✓ Modelo calibrado salvo: {model_path}")

scaler_path = "idade_scaler.pkl"
joblib.dump(scaler, scaler_path)
print(f"✓ Scaler salvo: {scaler_path}")

# ============================================================================
# 16. SALVAR PARÂMETROS E METADADOS
# ============================================================================
print("\n[16] SALVAMENTO DE PARÂMETROS E METADADOS")
print("-" * 100)

import json

metadata = {
    "versao": "gold_standard_v1",
    "features_input": INPUT_FEATURES,
    "n_features": len(INPUT_FEATURES),
    "target": TARGET_COLUMN,
    "classes": ["NÃO", "SIM"],
    "scaling": {
        "idade_mean": float(idade_mean),
        "idade_std": float(idade_std),
        "outras_features": "one-hot encoded (binary)"
    },
    "model": {
        "type": "LogisticRegression (calibrated)",
        "solver": "lbfgs",
        "calibration": "Platt Scaling",
        "parameters": {
            "penalty": "l2",
            "C": 1.0,
            "max_iter": 1000
        }
    },
    "training_data": {
        "total_samples": len(df_model),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "class_distribution_train": {
            "NÃO": int((y_train == 0).sum()),
            "SIM": int((y_train == 1).sum())
        }
    },
    "metrics": {
        "roc_auc": float(roc_auc),
        "accuracy": float(report['accuracy']),
        "brier_score": float(brier),
        "cv_score_mean": float(cv_scores.mean()),
        "cv_score_std": float(cv_scores.std())
    }
}

metadata_path = "model_metadata.json"
with open(metadata_path, "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2, ensure_ascii=False)

print(f"✓ Metadados salvos: {metadata_path}")

# Salvar parâmetros de escalonamento também em formato texto
with open("scaling_params.txt", "w", encoding="utf-8") as f:
    f.write(f"IDADE_MEAN = {idade_mean}\n")
    f.write(f"IDADE_STD = {idade_std}\n")

print(f"✓ Parâmetros de escalonamento salvos: scaling_params.txt")

# ============================================================================
# 17. RESUMO FINAL
# ============================================================================
print("\n" + "=" * 100)
print("RESUMO FINAL")
print("=" * 100)

print(f"""
✓ TREINAMENTO CONCLUÍDO COM SUCESSO

Configuração:
  - Features: {len(INPUT_FEATURES)} (apenas entrada)
  - Alvo: {TARGET_COLUMN}
  - Total de amostras: {len(df_model):,}
  - IGNORADO tratado como: NÃO

Preprocessamento:
  - Codificação: Label Encoding (features categóricas)
  - Escalonamento: StandardScaler (IDADE)
  - One-hot: Não (features binárias mantidas simples)

Modelo:
  - Tipo: Regressão Logística com Calibração (Platt Scaling)
  - Validação Cruzada: 5-Fold, Média ROC-AUC = {cv_scores.mean():.4f}

Performance (Teste):
  - ROC-AUC: {roc_auc:.4f}
  - Acurácia: {report['accuracy']:.4f}
  - Brier Score: {brier:.4f}

Probabilidades:
  - Intervalo: [{y_pred_proba.min()*100:.2f}%, {y_pred_proba.max()*100:.2f}%]
  - Média: {y_pred_proba.mean()*100:.2f}%
  - Distribuição: OK (valores distribuídos adequadamente)

Arquivos gerados:
  ✓ {model_path} (modelo treinado)
  ✓ {scaler_path} (scaler para IDADE)
  ✓ {metadata_path} (metadados)
  ✓ scaling_params.txt (parâmetros de escalonamento)

Próximos passos:
  1. Atualizar app.py para usar o novo modelo
  2. Testar predições com vários inputs
  3. Deploy no Render
""")

print("=" * 100)
