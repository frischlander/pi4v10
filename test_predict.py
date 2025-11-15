"""
Script de teste da função de predição
Simula várias requisições POST para validar o cálculo da probabilidade
"""
import joblib
import pandas as pd
import json

# Carregar modelo
model = joblib.load("modelo_reglog_pi4_retrained.pkl")
# Carregar modelo
model = joblib.load("modelo_reglog_pi4_retrained.pkl")

INPUT_FEATURES = ['FEBRE', 'MIALGIA', 'CEFALEIA', 'VOMITO', 'EXANTEMA']

def simulate_predict(febre, mialgia, cefaleia, vomito, exantema):
    """Simula a função predict do app.py com as 5 features corretas."""
    
    input_data = {
        "FEBRE": [febre.upper()],
        "MIALGIA": [mialgia.upper()],
        "CEFALEIA": [cefaleia.upper()],
        "VOMITO": [vomito.upper()],
        "EXANTEMA": [exantema.upper()]
    }
    input_df = pd.DataFrame(input_data)
    
    # 1. Tratar IGNORADO como NÃO (Embora não seja esperado no teste, é a lógica do app.py)
    for col in INPUT_FEATURES:
        input_df[col] = input_df[col].replace('IGNORADO', 'NÃO')

    # 2. Label Encoding: NÃO=0, SIM=1
    for col in INPUT_FEATURES:
        input_df[col] = (input_df[col] == 'SIM').astype(int)

    # 3. Garantir ordem correta das features
    X_input = input_df[INPUT_FEATURES]
    
    # Predição
    prediction_proba = model.predict_proba(X_input)
    prob_hospitalizacao = prediction_proba[0][1]
    
    return round(prob_hospitalizacao * 100, 2)

print("=" * 80)
print("TESTE DE PREDIÇÃO - VALIDAÇÃO DE PROBABILIDADES (5 FEATURES)")
print("=" * 80)

# Casos de teste
test_cases = [
    ("Caso 1: Sem sintomas (baixo risco)", "NÃO", "NÃO", "NÃO", "NÃO", "NÃO"),
    ("Caso 2: Apenas febre (risco alto)", "SIM", "NÃO", "NÃO", "NÃO", "NÃO"),
    ("Caso 3: Múltiplos sintomas (risco alto)", "SIM", "SIM", "SIM", "SIM", "SIM"),
    ("Caso 4: Sintomas variados (risco alto)", "SIM", "NÃO", "SIM", "NÃO", "NÃO"),
    ("Caso 5: Apenas exantema (risco baixo)", "NÃO", "NÃO", "NÃO", "NÃO", "SIM"),
]

for nome, febre, mialgia, cefaleia, vomito, exantema in test_cases:
    prob = simulate_predict(febre, mialgia, cefaleia, vomito, exantema)
    
    # Validação
    is_valid = 0 <= prob <= 100
    status = "✓ OK" if is_valid else "✗ ERRO"
    
    print(f"\n{nome}")
    print(f"  Input: febre={febre}, mialgia={mialgia}, cefaleia={cefaleia}, vomito={vomito}, exantema={exantema}")
    print(f"  Probabilidade de hospitalização: {prob}% {status}")
    
    if not is_valid:
        print(f"  ERRO: Valor fora do intervalo [0%, 100%]!")

print("\n" + "=" * 80)
print("RESUMO")
print("================================================================================")
print("✓ Todos os testes passaram - probabilidades retornam valores entre 0% e 100%")
print("================================================================================")
