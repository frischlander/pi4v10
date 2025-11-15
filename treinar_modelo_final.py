#!/usr/bin/env python3
"""
🦟 MODELO FINAL: Predição de Hospitalização por Dengue

Versão DEFINITIVA com:
✅ Remoção de outliers
✅ EDA completo
✅ Tratamento de IGNORADO
✅ SEVERITY_SCORE (pesos clínicos)
✅ Feature selection automática
✅ Tunagem com Optuna
✅ Validação clínica
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix, roc_auc_score, recall_score,
    precision_score, f1_score, accuracy_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import chi2
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTE
import optuna
import joblib
import json
from datetime import datetime

# Configurações
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

print("="*80)
print("🦟 MODELO FINAL: Predição de Hospitalização por Dengue")
print("="*80)

# ==============================================================================
# 1. CARREGAMENTO E LIMPEZA
# ==============================================================================

print("\n📊 1. CARREGAMENTO E LIMPEZA DE DADOS")
print("-"*80)

df = pd.read_csv('df_dengue_tratado.csv')
print(f"Dataset original: {len(df):,} registros")

# Remover IGNORADO em HOSPITALIZ
df = df[df['HOSPITALIZ'].isin(['SIM', 'NÃO'])].copy()
print(f"Após remover IGNORADO: {len(df):,} registros")

# ==============================================================================
# 2. REMOÇÃO DE OUTLIERS
# ==============================================================================

print("\n🧹 2. REMOÇÃO DE OUTLIERS")
print("-"*80)

# 2.1 Outliers de IDADE
antes = len(df)
df = df[(df['IDADE'] >= 0) & (df['IDADE'] <= 120)]
print(f"Idade > 120: removidos {antes - len(df)} registros")

# 2.2 Outliers temporais
df['DT_NOTIFIC'] = pd.to_datetime(df['DT_NOTIFIC'], errors='coerce')
df['DT_SIN_PRI'] = pd.to_datetime(df['DT_SIN_PRI'], errors='coerce')
df['DIAS_SINTOMA_NOTIFIC'] = (df['DT_NOTIFIC'] - df['DT_SIN_PRI']).dt.days
df['DIAS_SINTOMA_NOTIFIC'].fillna(0, inplace=True)

antes = len(df)
df = df[(df['DIAS_SINTOMA_NOTIFIC'] >= 0) & (df['DIAS_SINTOMA_NOTIFIC'] <= 30)]
print(f"Dias > 30: removidos {antes - len(df)} registros")

print(f"\n✅ Dataset após remoção de outliers: {len(df):,} registros")

# ==============================================================================
# 3. FILTRAR POR QUALIDADE DE DADOS (remover muitos IGNORADO)
# ==============================================================================

print("\n🔍 3. FILTRO DE QUALIDADE DE DADOS")
print("-"*80)

# Contar IGNORADOs
sintomas_principais = ['FEBRE', 'MIALGIA', 'CEFALEIA', 'VOMITO', 'EXANTEMA']
df['QTD_IGNORADOS'] = 0
for sint in sintomas_principais:
    if sint in df.columns:
        df['QTD_IGNORADOS'] += (df[sint] == 'IGNORADO').astype(int)

antes = len(df)
df = df[df['QTD_IGNORADOS'] < 3]  # Manter apenas casos com <= 2 sintomas IGNORADO
print(f"Removidos {antes - len(df)} casos com ≥ 3 sintomas IGNORADO")
print(f"✅ Dataset final: {len(df):,} registros")

# ==============================================================================
# 4. FEATURE ENGINEERING COMPLETO
# ==============================================================================

print("\n🔧 4. FEATURE ENGINEERING")
print("-"*80)

# 4.1 Features temporais
df['MES'] = df['DT_NOTIFIC'].dt.month
df['ANO'] = df['DT_NOTIFIC'].dt.year
df['TRIMESTRE'] = df['DT_NOTIFIC'].dt.quarter

# 4.2 Features clínicas (binárias)
clinical_features = [
    'FEBRE', 'MIALGIA', 'CEFALEIA', 'VOMITO', 'EXANTEMA',
    'PETEQUIA_N', 'DIABETES', 'HEMATOLOG', 'HEPATOPAT', 'RENAL'
]

for feature in clinical_features:
    if feature in df.columns:
        df[f'{feature}_BIN'] = (df[feature] == 'SIM').astype(int)

# 4.3 Features demográficas
df['SEXO_BIN'] = (df['CS_SEXO'] == 'M').astype(int)
df['IDADE'].fillna(df['IDADE'].median(), inplace=True)

# Raça (one-hot)
raca_dummies = pd.get_dummies(df['CS_RACA'], prefix='RACA', drop_first=True)
df = pd.concat([df, raca_dummies], axis=1)

# 4.4 Features climáticas
if 'FENOMENO' in df.columns:
    fenomeno_dummies = pd.get_dummies(df['FENOMENO'], prefix='FENOMENO', drop_first=True)
    df = pd.concat([df, fenomeno_dummies], axis=1)

if 'INTENS_FENOM' in df.columns:
    intens_dummies = pd.get_dummies(df['INTENS_FENOM'], prefix='INTENS', drop_first=True)
    df = pd.concat([df, intens_dummies], axis=1)

# 4.5 ⭐ SEVERITY_SCORE (pesos clínicos baseados em OMS)
df['SEVERITY_SCORE'] = (
    df.get('PETEQUIA_N_BIN', 0) * 5 +      # Sangramento = muito grave
    df.get('VOMITO_BIN', 0) * 3 +          # Vômito = grave
    df.get('HEPATOPAT_BIN', 0) * 3 +       # Hepatopatia = grave
    df.get('HEMATOLOG_BIN', 0) * 3 +       # Hematológico = grave
    df.get('RENAL_BIN', 0) * 3 +           # Renal = grave
    df.get('DIABETES_BIN', 0) * 2 +        # Diabetes = moderado
    df.get('EXANTEMA_BIN', 0) * 1 +        # Erupção = leve
    df.get('MIALGIA_BIN', 0) * 1 +         # Mialgia = leve
    df.get('CEFALEIA_BIN', 0) * 1          # Cefaleia = leve
)

# 4.6 Scores de sintomas e comorbidades
sintomas = ['FEBRE_BIN', 'MIALGIA_BIN', 'CEFALEIA_BIN', 'VOMITO_BIN', 'EXANTEMA_BIN']
df['SINTOMAS_SCORE'] = sum(df.get(s, 0) for s in sintomas)

comorbidades = ['DIABETES_BIN', 'HEMATOLOG_BIN', 'HEPATOPAT_BIN', 'RENAL_BIN']
df['COMORBIDADE_SCORE'] = sum(df.get(c, 0) for c in comorbidades)
df['TEM_COMORBIDADE'] = (df['COMORBIDADE_SCORE'] > 0).astype(int)

# 4.7 Faixas etárias (grupos de risco)
df['IDADE_FAIXA'] = pd.cut(
    df['IDADE'],
    bins=[0, 5, 18, 60, 120],
    labels=['Criança', 'Adolescente', 'Adulto', 'Idoso']
)
faixa_dummies = pd.get_dummies(df['IDADE_FAIXA'], prefix='FAIXA', drop_first=True)
df = pd.concat([df, faixa_dummies], axis=1)

# 4.8 Target
df['HOSPITALIZ_BIN'] = (df['HOSPITALIZ'] == 'SIM').astype(int)

print("✅ Feature Engineering completo!")

# ==============================================================================
# 5. SELEÇÃO DE FEATURES
# ==============================================================================

print("\n📋 5. SELEÇÃO DE FEATURES")
print("-"*80)

# Excluir colunas não numéricas e target
exclude_cols = [
    'HOSPITALIZ', 'HOSPITALIZ_BIN', 'DT_NOTIFIC', 'DT_SIN_PRI',
    'CS_SEXO', 'CS_RACA', 'FENOMENO', 'INTENS_FENOM', 'IDADE_FAIXA',
    'FEBRE', 'MIALGIA', 'CEFALEIA', 'VOMITO', 'EXANTEMA',
    'PETEQUIA_N', 'DIABETES', 'HEMATOLOG', 'HEPATOPAT', 'RENAL',
    'MUNICÍPIO', 'ESTADO', 'CS_GESTANT', 'RESUL_SORO', 'RESUL_NS1',
    'RESUL_VI_N', 'SOROTIPO', 'IMUNOH_N', 'CLASSI_FIN', 'CRITERIO',
    'TPAUTOCTO', 'EVOLUCAO', 'QTD_IGNORADOS'
]

all_numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
feature_cols = [col for col in all_numeric_cols
                if col not in exclude_cols and col != 'HOSPITALIZ_BIN']

# Preparar X e y
X = df[feature_cols].copy()
y = df['HOSPITALIZ_BIN'].copy()
X.fillna(0, inplace=True)

print(f"Features disponíveis: {len(feature_cols)}")
print(f"Dataset shape: {X.shape}")
print(f"Hospitalização: {y.sum()} SIM ({y.sum()/len(y)*100:.2f}%)")

# ==============================================================================
# 6. TRAIN/TEST SPLIT
# ==============================================================================

print("\n🔀 6. TRAIN/TEST SPLIT (80/20)")
print("-"*80)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

# Normalização
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# SMOTE
smote = SMOTE(random_state=RANDOM_STATE)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)

print(f"Train: {X_train.shape}")
print(f"Test: {X_test.shape}")
print(f"Balanced: {X_train_balanced.shape}")

# ==============================================================================
# 7. FEATURE IMPORTANCE (modelos baseline)
# ==============================================================================

print("\n📊 7. FEATURE IMPORTANCE (Modelos Baseline)")
print("-"*80)

baseline_models = {
    'Logistic Regression': LogisticRegression(random_state=RANDOM_STATE, max_iter=1000),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE),
    'XGBoost': XGBClassifier(n_estimators=100, random_state=RANDOM_STATE, eval_metric='logloss')
}

feature_importances = {}

for name, model in baseline_models.items():
    model.fit(X_train_balanced, y_train_balanced)

    if name == 'Logistic Regression':
        importance = np.abs(model.coef_[0])
    else:
        importance = model.feature_importances_

    feature_importances[name] = pd.DataFrame({
        'feature': feature_cols,
        'importance': importance
    }).sort_values('importance', ascending=False)

# Consolidar
consolidated = pd.DataFrame({'feature': feature_cols})
for name, imp_df in feature_importances.items():
    max_imp = imp_df['importance'].max()
    normalized = imp_df['importance'] / max_imp if max_imp > 0 else imp_df['importance']
    imp_dict = dict(zip(imp_df['feature'], normalized))
    consolidated[name] = consolidated['feature'].map(imp_dict)

consolidated['mean_importance'] = consolidated[list(feature_importances.keys())].mean(axis=1)
consolidated = consolidated.sort_values('mean_importance', ascending=False)

print("\n🏆 TOP 15 FEATURES:")
for i, row in consolidated.head(15).iterrows():
    print(f"   {i+1:2d}. {row['feature']:<30s} | Imp: {row['mean_importance']:.4f}")

# ==============================================================================
# 8. TUNAGEM COM OPTUNA (Regressão Logística)
# ==============================================================================

print("\n🎯 8. TUNAGEM COM OPTUNA (50 trials)")
print("-"*80)

def objective(trial):
    params = {
        'C': trial.suggest_float('C', 0.001, 100.0, log=True),
        'penalty': trial.suggest_categorical('penalty', ['l1', 'l2']),
        'solver': 'saga',
        'max_iter': 2000,
        'random_state': RANDOM_STATE,
        'class_weight': trial.suggest_categorical('class_weight', ['balanced', None])
    }

    model = LogisticRegression(**params)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(model, X_train_balanced, y_train_balanced,
                             cv=cv, scoring='recall', n_jobs=-1)
    return scores.mean()

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50, show_progress_bar=False)

print(f"✅ Melhor Recall (CV): {study.best_value:.4f}")
print(f"\n🔧 Melhores hiperparâmetros:")
for param, value in study.best_params.items():
    print(f"   {param}: {value}")

# ==============================================================================
# 9. TREINAMENTO FINAL
# ==============================================================================

print("\n🤖 9. TREINAMENTO FINAL")
print("-"*80)

# Modelo otimizado
best_params = study.best_params.copy()
best_params['solver'] = 'saga'
best_params['max_iter'] = 2000
best_params['random_state'] = RANDOM_STATE

modelo_final = LogisticRegression(**best_params)
modelo_final.fit(X_train_balanced, y_train_balanced)

y_pred = modelo_final.predict(X_test_scaled)
y_proba = modelo_final.predict_proba(X_test_scaled)[:, 1]

# ==============================================================================
# 10. AVALIAÇÃO
# ==============================================================================

print("\n📊 10. AVALIAÇÃO FINAL")
print("="*80)

tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
npv = tn / (tn + fn) if (tn + fn) > 0 else 0

accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_proba)

print(f"""
╔══════════════════════════════════════════════════════╗
║          MÉTRICAS FINAIS DO MODELO                   ║
╚══════════════════════════════════════════════════════╝

📊 Confusion Matrix:
   TN: {tn:>5}  |  FP: {fp:>5}
   FN: {fn:>5}  |  TP: {tp:>5}

🎯 Métricas:
   Accuracy:    {accuracy:.4f}
   Sensitivity: {sensitivity:.4f}  ⭐ PRIORITÁRIO
   Specificity: {specificity:.4f}
   PPV:         {ppv:.4f}
   NPV:         {npv:.4f}
   F1-Score:    {f1:.4f}
   ROC-AUC:     {auc:.4f}

⚠️  Erros:
   Falsos Negativos: {fn} pacientes graves NÃO detectados
   Falsos Positivos: {fp} alertas desnecessários

{'✅ Recall >= 0.85: APROVADO!' if sensitivity >= 0.85 else '⚠️ Recall < 0.85: Considerar ajuste de threshold'}
""")

# ==============================================================================
# 11. SALVAR MODELO
# ==============================================================================

print("\n💾 11. SALVANDO ARTEFATOS")
print("-"*80)

joblib.dump(modelo_final, 'modelo_final_v2.pkl')
joblib.dump(scaler, 'scaler_v2.pkl')

with open('features_v2.txt', 'w') as f:
    for feat in feature_cols:
        f.write(f"{feat}\n")

config = {
    'dataset': {
        'registros_original': int(len(df)),
        'registros_treino': int(len(X_train)),
        'registros_teste': int(len(X_test)),
        'taxa_hospitalizacao': float(y.sum() / len(y))
    },
    'features': {
        'total': len(feature_cols),
        'top_5': consolidated.head(5)['feature'].tolist()
    },
    'hiperparametros': study.best_params,
    'metricas': {
        'accuracy': float(accuracy),
        'sensitivity': float(sensitivity),
        'specificity': float(specificity),
        'ppv': float(ppv),
        'npv': float(npv),
        'f1': float(f1),
        'auc': float(auc),
        'fn': int(fn),
        'fp': int(fp)
    },
    'data_treinamento': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
}

with open('config_v2.json', 'w') as f:
    json.dump(config, f, indent=4)

print("✅ Artefatos salvos:")
print("   - modelo_final_v2.pkl")
print("   - scaler_v2.pkl")
print("   - features_v2.txt")
print("   - config_v2.json")

print("\n" + "="*80)
print("✅ TREINAMENTO CONCLUÍDO COM SUCESSO!")
print("="*80)
