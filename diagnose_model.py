"""
Script de diagnóstico para verificar o modelo carregado
"""
import joblib
import pandas as pd
import numpy as np

print("=" * 70)
print("DIAGNÓSTICO DO MODELO PREDITIVO")
print("=" * 70)

try:
    model = joblib.load("modelo_reglog_pi4_retrained.pkl")
    print(f"\n✓ Modelo carregado: {type(model).__name__}")
    print(f"  Número de coeficientes: {model.coef_.shape[1]}")
    
    if hasattr(model, 'feature_names_in_'):
        features = model.feature_names_in_
        print(f"  Número de features esperadas: {len(features)}")
        print(f"\n  Features do modelo:")
        for i, feat in enumerate(features, 1):
            print(f"    {i:2d}. {feat}")
    else:
        print("  AVISO: Modelo não tem feature_names_in_")
        
except Exception as e:
    print(f"\n✗ ERRO ao carregar modelo: {e}")
    exit(1)

print("\n" + "=" * 70)
print("ANÁLISE DAS 7 FEATURES DE INPUT")
print("=" * 70)

INPUT_FEATURES = ['IDADE', 'CS_SEXO', 'FEBRE', 'VOMITO', 'MIALGIA', 'CEFALEIA', 'EXANTEMA']
CATEGORICAL_COLS = ['CS_SEXO', 'FEBRE', 'VOMITO', 'MIALGIA', 'CEFALEIA', 'EXANTEMA']

# Simular entrada
input_data = {
    "IDADE": [30],
    "CS_SEXO": ["F"],
    "FEBRE": ["NÃO"],
    "VOMITO": ["NÃO"],
    "MIALGIA": ["NÃO"],
    "CEFALEIA": ["NÃO"],
    "EXANTEMA": ["NÃO"]
}

input_df = pd.DataFrame(input_data)
print(f"\nInput original:")
print(input_df)

# One-hot encode
input_encoded = pd.get_dummies(input_df, columns=CATEGORICAL_COLS, drop_first=False)
print(f"\nApós one-hot encoding ({len(input_encoded.columns)} colunas):")
print(input_encoded.columns.tolist())

# Comparar com features do modelo
print(f"\nComparação:")
print(f"  Colunas após encoding: {len(input_encoded.columns)}")
print(f"  Features esperadas pelo modelo: {len(features)}")
print(f"  Diferença: {len(features) - len(input_encoded.columns)} features faltando")

print("\n" + "=" * 70)
print("TESTE DE PREDIÇÃO")
print("=" * 70)

# Alinhar com features do modelo
input_aligned = pd.DataFrame(0, index=[0], columns=features)

for col in input_encoded.columns:
    if col in input_aligned.columns:
        input_aligned[col] = input_encoded[col].iloc[0]
    else:
        print(f"  AVISO: Coluna '{col}' não encontrada no modelo")

# Aplicar escalonamento de IDADE
IDADE_MEAN = 35.63
IDADE_STD = 19.34
if 'IDADE' in input_aligned.columns:
    idade_escalada = (30 - IDADE_MEAN) / IDADE_STD
    input_aligned['IDADE'] = idade_escalada
    print(f"\nEscalonamento de IDADE:")
    print(f"  Valor original: 30")
    print(f"  Média: {IDADE_MEAN}")
    print(f"  Desvio padrão: {IDADE_STD}")
    print(f"  Valor escalado: {idade_escalada:.4f}")

print(f"\nInput alinhado com {len(input_aligned.columns)} colunas")

# Fazer predição
try:
    prediction_proba = model.predict_proba(input_aligned)
    prob_class_0 = prediction_proba[0][0]
    prob_class_1 = prediction_proba[0][1]
    
    print(f"\nPredição:")
    print(f"  P(NÃO hospitalizado): {prob_class_0:.4f} ({prob_class_0*100:.2f}%)")
    print(f"  P(Hospitalizado):     {prob_class_1:.4f} ({prob_class_1*100:.2f}%)")
    print(f"  Soma das probabilidades: {prob_class_0 + prob_class_1:.4f}")
    
    if prob_class_1 * 100 > 100:
        print(f"\n✗ ERRO: Probabilidade acima de 100%!")
    elif 0 <= prob_class_1 * 100 <= 100:
        print(f"\n✓ OK: Probabilidade está no intervalo correto [0%, 100%]")
        
except Exception as e:
    print(f"\n✗ ERRO ao fazer predição: {e}")

print("\n" + "=" * 70)
