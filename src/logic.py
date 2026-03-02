import pandas as pd
import numpy as np

def limpiar_tel(serie):
    return serie.astype(str).str.replace(r'\.0$', '', regex=True).str.strip()

def limpiar_dni(serie):
    return pd.to_numeric(serie, errors='coerce').fillna(0).astype(int)

def calcular_score_base(df_universo, modelo):
    df_universo['feature_opsitel'] = df_universo['Activo'].apply(lambda x: 1 if str(x).upper() == 'SI' else 0)
    df_universo['FECHA_ACT'] = pd.to_datetime(df_universo['FECHA_ACT'], errors='coerce')
    # Antigüedad en meses
    df_universo['feature_antiguedad_meses'] = ((2026 - df_universo['FECHA_ACT'].dt.year) * 12 + (2 - df_universo['FECHA_ACT'].dt.month)).fillna(300)

    X = df_universo[['feature_opsitel', 'feature_antiguedad_meses']]
    df_universo['total_score'] = modelo.predict_proba(X)[:, 1]

    return df_universo[['DNI', 'Telefono', 'total_score']]