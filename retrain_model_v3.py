"""
Script para retreinar o modelo com drop_first=True
Isso remove uma categoria por coluna para evitar multicolinearidade
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
import joblib
import numpy as np

# Definir as 7 features de input
INPUT_FEATURES = ['IDADE', 'CS_SEXO', 'FEBRE', 'VOMITO', 'MIALGIA', 'CEFALEIA', 'EXANTEMA']
TARGET_COLUMN = 'HOSPITALIZ'

print("=" * 80)
print("RETREINAMENTO DO MODELO - COM drop_first=True")
print("=" * 80)

# 1. Carregar o dataset
try:
    df = pd.read_csv("df_final_predict.csv")
    print(f"\n✓ Dataset carregado: {df.shape[0]} registros, {df.shape[1]} colunas")
except FileNotFoundError:
    print("ERRO: df_final_predict.csv não encontrado.")
    exit(1)

# 2. Selecionar apenas as features de input e a coluna alvo
df_model = df[INPUT_FEATURES + [TARGET_COLUMN]].copy()

# 3. Remover APENAS valores 'IGNORADO' no alvo (HOSPITALIZ)
print(f"\nAntes da filtragem: {len(df_model)} registros")

df_model = df_model[df_model[TARGET_COLUMN] != 'IGNORADO']

print(f"Após remover 'IGNORADO' no alvo: {len(df_model)} registros")

print(f"\nDistribuição de HOSPITALIZ após filtragem:")
print(df_model[TARGET_COLUMN].value_counts())

# 4. Aplicar One-Hot Encoding com drop_first=True
categorical_cols = ['CS_SEXO', 'FEBRE', 'VOMITO', 'MIALGIA', 'CEFALEIA', 'EXANTEMA']
df_encoded = pd.get_dummies(df_model, columns=categorical_cols, drop_first=True)

print(f"\nApós one-hot encoding (drop_first=True): {df_encoded.shape[1]} colunas")
print(f"Colunas após encoding:")
for i, col in enumerate(df_encoded.columns, 1):
    print(f"  {i:2d}. {col}")

# 5. Criar a coluna alvo binária
df_encoded['HOSPITALIZ_SIM'] = (df_encoded[TARGET_COLUMN] == 'SIM').astype(int)

print(f"\nDistribuição da classe alvo:")
print(f"  NÃO: {(df_encoded['HOSPITALIZ_SIM'] == 0).sum()}")
print(f"  SIM: {(df_encoded['HOSPITALIZ_SIM'] == 1).sum()}")

# 6. Remover a coluna alvo original
X = df_encoded.drop(columns=[TARGET_COLUMN, 'HOSPITALIZ_SIM'])
y = df_encoded['HOSPITALIZ_SIM']

print(f"\nFeatures finais: {X.shape[1]}")
print(f"Amostras: {X.shape[0]}")

# 7. Calcular estatísticas da IDADE para escalonamento
print(f"\nEstatísticas de IDADE:")
print(f"  Média: {X['IDADE'].mean():.2f}")
print(f"  Desvio padrão: {X['IDADE'].std():.2f}")

# 8. Escalar IDADE
scaler = StandardScaler()
X['IDADE'] = scaler.fit_transform(X[['IDADE']])

# 9. Dividir em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"\nTreino/Teste split:")
print(f"  Treino: {X_train.shape[0]} amostras")
print(f"  Teste: {X_test.shape[0]} amostras")

# 10. Treinar o Modelo com class_weight='balanced'
print(f"\nTreinando modelo Regressão Logística com class_weight='balanced'...")
model = LogisticRegression(solver='lbfgs', max_iter=1000, random_state=42, class_weight='balanced')
model.fit(X_train, y_train)

print(f"✓ Modelo treinado")

# 11. Avaliação
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

roc_auc = roc_auc_score(y_test, y_proba)
report = classification_report(y_test, y_pred, target_names=['NÃO', 'SIM'], output_dict=True)
cm = confusion_matrix(y_test, y_pred)

print(f"\n" + "=" * 80)
print("RESULTADOS DO MODELO")
print("=" * 80)
print(f"\nROC AUC: {roc_auc:.4f}")
print(f"Acurácia: {report['accuracy']:.4f}")

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
print(f"  {'Real NÃO':>15}    {cm[0,0]:>4d}        {cm[0,1]:>4d}")
print(f"  {'Real SIM':>15}    {cm[1,0]:>4d}        {cm[1,1]:>4d}")

# 12. Analisar distribuição de probabilidades
print(f"\nDistribuição de probabilidades preditas:")
print(f"  Mínimo: {y_proba.min():.6f}")
print(f"  Máximo: {y_proba.max():.6f}")
print(f"  Média: {y_proba.mean():.6f}")
print(f"  Mediana: {np.median(y_proba):.6f}")
print(f"  Percentil 25: {np.percentile(y_proba, 25):.6f}")
print(f"  Percentil 75: {np.percentile(y_proba, 75):.6f}")

# 13. Salvar o modelo
new_model_path = "modelo_reglog_pi4_retrained.pkl"
joblib.dump(model, new_model_path)
print(f"\n✓ Novo modelo salvo em: {new_model_path}")

# 14. Salvar os parâmetros de escalonamento
scaling_params = {
    "idade_mean": float(scaler.mean_[0]),
    "idade_std": float(scaler.scale_[0])
}

with open("scaling_params.txt", "w") as f:
    f.write(f"IDADE_MEAN = {scaling_params['idade_mean']}\n")
    f.write(f"IDADE_STD = {scaling_params['idade_std']}\n")

print(f"✓ Parâmetros de escalonamento salvos")

# 15. Salvar as métricas
import json
metrics = {
    "roc_auc": round(roc_auc * 100, 2),
    "accuracy": round(report['accuracy'] * 100, 2),
    "precision_nao": round(report['NÃO']['precision'] * 100, 2),
    "recall_nao": round(report['NÃO']['recall'] * 100, 2),
    "precision_sim": round(report['SIM']['precision'] * 100, 2),
    "recall_sim": round(report['SIM']['recall'] * 100, 2),
    "features": X.columns.tolist(),
    "n_features": len(X.columns),
    "scaling_params": scaling_params
}

with open("model_metrics_retrained.json", "w") as f:
    json.dump(metrics, f, indent=2)

print(f"✓ Métricas salvas em model_metrics_retrained.json")

print(f"\n" + "=" * 80)
print("RETREINAMENTO CONCLUÍDO COM SUCESSO")
print("=" * 80)
