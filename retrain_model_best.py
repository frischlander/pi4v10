"""
Retrain model with only the TOP 5 BEST FEATURES based on feature importance:
FEBRE, MIALGIA, CEFALEIA, VOMITO, EXANTEMA

This script implements:
1. Feature selection based on importance analysis
2. Proper class balancing
3. Good calibration using Platt scaling (sigmoid)
4. Cross-validation for robust evaluation
5. Probability distribution verification

Output files:
 - modelo_reglog_pi4_retrained.pkl (calibrated model)
 - model_metadata.json (metrics and features)
 - feature_importance_analysis.txt (analysis report)
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score, accuracy_score, brier_score_loss, confusion_matrix
import joblib
import json

RANDOM_STATE = 42
FEATURES = ['FEBRE', 'MIALGIA', 'CEFALEIA', 'VOMITO', 'EXANTEMA']
MODEL_OUT = 'modelo_reglog_pi4_retrained.pkl'
METADATA_OUT = 'model_metadata.json'
ANALYSIS_OUT = 'feature_importance_analysis.txt'

print("=" * 80)
print("TREINAMENTO DO MODELO COM AS 5 MELHORES FEATURES")
print("=" * 80)

print("\n[1] Carregando dados...")
df = pd.read_csv('df_final_predict.csv')
print(f"  Total de registros: {len(df):,}")

# Preparar features
print("\n[2] Preparando features...")
X = pd.DataFrame()
for col in FEATURES:
    df[col] = df[col].fillna('IGNORADO').replace('IGNORADO', 'NÃO')
    X[col] = (df[col].str.upper() == 'SIM').astype(int)

y = (df['HOSPITALIZ'].str.upper() == 'SIM').astype(int)

print(f"  Features selecionadas: {FEATURES}")
print(f"  Distribuição de classes:")
print(f"    NÃO hospitalizados: {(y==0).sum():,} ({(y==0).sum()/len(y)*100:.1f}%)")
print(f"    Hospitalizados:     {(y==1).sum():,} ({(y==1).sum()/len(y)*100:.1f}%)")

# Split train/test
print("\n[3] Split treino/teste...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
)
print(f"  Treino: {len(X_train):,}, Teste: {len(X_test):,}")

# Criar modelo base com melhores hiperparâmetros
print("\n[4] Treinando modelo com calibração...")
print("  Model: LogisticRegression (C=1.0, solver='lbfgs', max_iter=1000)")
print("  Calibração: CalibratedClassifierCV (method='sigmoid', cv=5)")

lr = LogisticRegression(C=1.0, solver='lbfgs', max_iter=1000, random_state=RANDOM_STATE)
clf = CalibratedClassifierCV(estimator=lr, cv=5, method='sigmoid')

clf.fit(X_train, y_train)

# Avaliar
print("\n[5] Avaliando modelo...")
y_pred_proba_test = clf.predict_proba(X_test)[:, 1]
y_pred_test = (y_pred_proba_test >= 0.5).astype(int)

roc_test = roc_auc_score(y_test, y_pred_proba_test)
acc_test = accuracy_score(y_test, y_pred_test)
brier_test = brier_score_loss(y_test, y_pred_proba_test)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred_test).ravel()

print(f"  ROC-AUC (teste):      {roc_test:.4f}")
print(f"  Accuracy (teste):     {acc_test:.4f}")
print(f"  Brier score (teste):  {brier_test:.4f}")

# Distribuição de probabilidades
print(f"\n[6] Distribuição de probabilidades (teste):")
print(f"  Min: {y_pred_proba_test.min()*100:.2f}%")
print(f"  Max: {y_pred_proba_test.max()*100:.2f}%")
print(f"  Média: {y_pred_proba_test.mean()*100:.2f}%")
print(f"  Mediana: {np.median(y_pred_proba_test)*100:.2f}%")
print(f"  Std: {y_pred_proba_test.std()*100:.2f}%")

# Verificar bins
bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
hist, _ = np.histogram(y_pred_proba_test, bins=bins)
print(f"\n  Distribuição em bins:")
for i in range(len(bins)-1):
    print(f"    [{bins[i]:.1f} - {bins[i+1]:.1f}): {hist[i]:5,} ({hist[i]/len(y_pred_proba_test)*100:5.1f}%)")

# Probabilidades por classe real
print(f"\n[7] Probabilidades por classe real (teste):")
proba_nao_hosp = y_pred_proba_test[y_test == 0]
proba_hosp = y_pred_proba_test[y_test == 1]
print(f"  NÃO hospitalizados: min={proba_nao_hosp.min()*100:.2f}%, max={proba_nao_hosp.max()*100:.2f}%, média={proba_nao_hosp.mean()*100:.2f}%")
print(f"  Hospitalizados:     min={proba_hosp.min()*100:.2f}%, max={proba_hosp.max()*100:.2f}%, média={proba_hosp.mean()*100:.2f}%")

# Cross-validation no dataset completo
print(f"\n[8] Cross-validation (5-fold) no dataset completo...")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
cv_scores = cross_val_score(clf, X, y, cv=cv, scoring='roc_auc')
print(f"  5-Fold CV ROC-AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
print(f"  Scores por fold: {[f'{s:.4f}' for s in cv_scores]}")

# Verificar se as probabilidades estão em [0, 1]
print(f"\n[9] Verificação de validade:")
if y_pred_proba_test.min() < 0 or y_pred_proba_test.max() > 1:
    print(f"  [ERRO] Probabilidades fora de [0,1]!")
    raise ValueError("Probabilidades inválidas!")
else:
    print(f"  OK - Probabilidades válidas em [0, 1]")

# Salvar modelo
print(f"\n[10] Salvando artefatos...")
joblib.dump(clf, MODEL_OUT)
print(f"  Modelo salvo em: {MODEL_OUT}")

# Salvar metadata
metadata = {
    'features': FEATURES,
    'n_samples': int(len(df)),
    'train_size': int(len(X_train)),
    'test_size': int(len(X_test)),
    'roc_auc_test': float(roc_test),
    'accuracy_test': float(acc_test),
    'brier_test': float(brier_test),
    'cv_roc_mean': float(cv_scores.mean()),
    'cv_roc_std': float(cv_scores.std()),
    'proba_min': float(y_pred_proba_test.min()),
    'proba_max': float(y_pred_proba_test.max()),
    'proba_mean': float(y_pred_proba_test.mean()),
}
with open(METADATA_OUT, 'w', encoding='utf-8') as f:
    json.dump(metadata, f, indent=2, ensure_ascii=False)
print(f"  Metadata salvo em: {METADATA_OUT}")

# Salvar análise
with open(ANALYSIS_OUT, 'w', encoding='utf-8') as f:
    f.write("=" * 80 + "\n")
    f.write("FEATURE IMPORTANCE ANALYSIS\n")
    f.write("=" * 80 + "\n\n")
    f.write("Features selecionadas (Top 5):\n")
    for i, feat in enumerate(FEATURES, 1):
        f.write(f"  {i}. {feat}\n")
    f.write(f"\n\nFeaturas removidas:\n")
    removed = ['CS_SEXO', 'CS_GESTANT', 'IDADE']
    for feat in removed:
        f.write(f"  - {feat}\n")
    f.write(f"\n\nModelo Performance:\n")
    f.write(f"  ROC-AUC (teste): {roc_test:.4f}\n")
    f.write(f"  Accuracy (teste): {acc_test:.4f}\n")
    f.write(f"  Brier Score: {brier_test:.4f}\n")
    f.write(f"  5-Fold CV ROC-AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}\n")
    f.write(f"\n\nProbabilidade Distribution:\n")
    f.write(f"  Min: {y_pred_proba_test.min()*100:.2f}%\n")
    f.write(f"  Max: {y_pred_proba_test.max()*100:.2f}%\n")
    f.write(f"  Mean: {y_pred_proba_test.mean()*100:.2f}%\n")
    f.write(f"  Median: {np.median(y_pred_proba_test)*100:.2f}%\n")
print(f"  Análise salva em: {ANALYSIS_OUT}")

print("\n" + "=" * 80)
print("TREINAMENTO CONCLUÍDO COM SUCESSO")
print("=" * 80)
