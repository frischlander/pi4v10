"""
Retrain model using only 6 features (no age):
CS_SEXO, FEBRE, VOMITO, MIALGIA, CEFALEIA, EXANTEMA

Saves:
 - modelo_reglog_pi4_retrained.pkl
 - model_metadata.json
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score, accuracy_score, brier_score_loss
import joblib
import json

RANDOM_STATE = 42
FEATURES = ['CS_SEXO', 'FEBRE', 'VOMITO', 'MIALGIA', 'CEFALEIA', 'EXANTEMA']
MODEL_OUT = 'modelo_reglog_pi4_retrained.pkl'
METADATA_OUT = 'model_metadata.json'

print("Carregando dados de df_final_predict.csv...")
df = pd.read_csv('df_final_predict.csv')
print(f"Registros totais: {len(df):,}")

# Garantir colunas existem
missing = [c for c in FEATURES + ['HOSPITALIZ'] if c not in df.columns]
if missing:
    raise RuntimeError(f"Colunas faltando no dataset: {missing}")

# Tratar IGNORADO como NÃO para features
for col in FEATURES:
    df[col] = df[col].fillna('IGNORADO').replace('IGNORADO', 'NÃO')

# Preparar X e y
X = pd.DataFrame()
# CS_SEXO: F=0, M=1
X['CS_SEXO'] = (df['CS_SEXO'].fillna('F').str.upper() == 'M').astype(int)
# Sintomas: SIM=1, NÃO=0
for col in ['FEBRE', 'VOMITO', 'MIALGIA', 'CEFALEIA', 'EXANTEMA']:
    X[col] = (df[col].str.upper() == 'SIM').astype(int)

y = (df['HOSPITALIZ'].str.upper() == 'SIM').astype(int)

print('Distribuição de classes:')
print(y.value_counts(normalize=True) * 100)

# Split treino/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE)
print(f"Treino: {len(X_train):,}, Teste: {len(X_test):,}")

# Modelo base
lr = LogisticRegression(solver='lbfgs', max_iter=1000)
clf = CalibratedClassifierCV(estimator=lr, cv=5, method='sigmoid')

print('Treinando modelo (CalibratedClassifierCV com Platt)...')
clf.fit(X_train, y_train)

# Avaliar
y_pred_proba = clf.predict_proba(X_test)[:,1]
y_pred = (y_pred_proba >= 0.5).astype(int)
roc = roc_auc_score(y_test, y_pred_proba)
acc = accuracy_score(y_test, y_pred)
brier = brier_score_loss(y_test, y_pred_proba)

print(f"ROC-AUC (teste): {roc:.4f}")
print(f"Accuracy (teste): {acc:.4f}")
print(f"Brier score (teste): {brier:.4f}")

# Faixa de probabilidades
print(f"Probabilidade - min: {y_pred_proba.min()*100:.2f}%, max: {y_pred_proba.max()*100:.2f}%")

# Cross-val ROC
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
cv_scores = cross_val_score(clf, X, y, cv=cv, scoring='roc_auc')
print(f"5-Fold CV ROC-AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# Salvar modelo
joblib.dump(clf, MODEL_OUT)
print(f"Modelo salvo em: {MODEL_OUT}")

# Salvar metadata
metadata = {
    'features': FEATURES,
    'n_samples': int(len(df)),
    'train_size': int(len(X_train)),
    'test_size': int(len(X_test)),
    'roc_auc_test': float(roc),
    'accuracy_test': float(acc),
    'brier_test': float(brier),
    'cv_roc_mean': float(cv_scores.mean()),
    'cv_roc_std': float(cv_scores.std())
}
with open(METADATA_OUT, 'w', encoding='utf-8') as f:
    json.dump(metadata, f, indent=2, ensure_ascii=False)
print(f"Metadata salvo em: {METADATA_OUT}")

print('Treinamento concluído.')
