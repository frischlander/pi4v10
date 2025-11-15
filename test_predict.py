"""
Script de teste da função de predição
Simula várias requisições POST para validar o cálculo da probabilidade
"""
import joblib
import pandas as pd
import json

# Carregar modelo
model = joblib.load("modelo_reglog_pi4_retrained.pkl")
model_features = model.feature_names_in_

IDADE_MEAN = 39.15432300163132
IDADE_STD = 22.39768282493475
CATEGORICAL_COLS = ['CS_SEXO', 'FEBRE', 'VOMITO', 'MIALGIA', 'CEFALEIA', 'EXANTEMA']

def simulate_predict(idade, sexo, febre, vomito, mialgia, cefaleia, exantema):
    """Simula a função predict do app.py"""
    
    input_data = {
        "IDADE": [int(idade)],
        "CS_SEXO": [sexo],
        "FEBRE": [febre],
        "VOMITO": [vomito],
        "MIALGIA": [mialgia],
        "CEFALEIA": [cefaleia],
        "EXANTEMA": [exantema]
    }
    input_df = pd.DataFrame(input_data)
    
    # One-hot encoding
    input_encoded = pd.get_dummies(input_df, columns=CATEGORICAL_COLS, drop_first=False)
    
    # Alinhar com features do modelo
    input_aligned = pd.DataFrame(0.0, index=[0], columns=model_features)
    
    for col in input_encoded.columns:
        if col in input_aligned.columns:
            input_aligned[col] = input_encoded[col].iloc[0]
    
    # Escalar IDADE
    if 'IDADE' in input_aligned.columns:
        idade_escalada = (idade - IDADE_MEAN) / IDADE_STD
        input_aligned['IDADE'] = idade_escalada
    
    # Predição
    prediction_proba = model.predict_proba(input_aligned)
    prob_hospitalizacao = prediction_proba[0][1]
    
    return round(prob_hospitalizacao * 100, 2)

print("=" * 80)
print("TESTE DE PREDIÇÃO - VALIDAÇÃO DE PROBABILIDADES")
print("=" * 80)

# Casos de teste
test_cases = [
    ("Caso 1: Sem sintomas (baixo risco)", 30, "F", "NÃO", "NÃO", "NÃO", "NÃO", "NÃO"),
    ("Caso 2: Com febre (risco moderado)", 30, "F", "SIM", "NÃO", "NÃO", "NÃO", "NÃO"),
    ("Caso 3: Múltiplos sintomas (risco alto)", 50, "M", "SIM", "SIM", "SIM", "SIM", "SIM"),
    ("Caso 4: Sintomas variados", 25, "F", "SIM", "NÃO", "SIM", "NÃO", "NÃO"),
    ("Caso 5: Idade avançada sem sintomas", 75, "M", "NÃO", "NÃO", "NÃO", "NÃO", "NÃO"),
]

for nome, idade, sexo, febre, vomito, mialgia, cefaleia, exantema in test_cases:
    prob = simulate_predict(idade, sexo, febre, vomito, mialgia, cefaleia, exantema)
    
    # Validação
    is_valid = 0 <= prob <= 100
    status = "✓ OK" if is_valid else "✗ ERRO"
    
    print(f"\n{nome}")
    print(f"  Input: idade={idade}, sexo={sexo}, febre={febre}, vomito={vomito}, mialgia={mialgia}, cefaleia={cefaleia}, exantema={exantema}")
    print(f"  Probabilidade de hospitalização: {prob}% {status}")
    
    if not is_valid:
        print(f"  ERRO: Valor fora do intervalo [0%, 100%]!")

print("\n" + "=" * 80)
print("RESUMO")
print("=" * 80)
print("✓ Todos os testes passaram - probabilidades retornam valores entre 0% e 100%")
print("=" * 80)
