        input_data = {
            "FEBRE": [data.get("febre", "NÃO").upper()],
            "MIALGIA": [data.get("mialgia", "NÃO").upper()],
            "CEFALEIA": [data.get("cefaleia", "NÃO").upper()],
            "VOMITO": [data.get("vomito", "NÃO").upper()],
            "EXANTEMA": [data.get("exantema", "NÃO").upper()]
        }

        input_df = pd.DataFrame(input_data)

        # 2. Tratar IGNORADO como NÃO
        for col in INPUT_FEATURES:
            input_df[col] = input_df[col].replace('IGNORADO', 'NÃO')

        # 3. Label Encoding: NÃO=0, SIM=1
        for col in INPUT_FEATURES:
            input_df[col] = (input_df[col] == 'SIM').astype(int)

        # 4. Garantir ordem correta das features
        X_input = input_df[INPUT_FEATURES]