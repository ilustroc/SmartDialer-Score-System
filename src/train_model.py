import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

from logic import limpiar_tel, limpiar_dni, validar_telefono_peru


NEGATIVOS_DETALLE = {
    'NRO. NO PERTENECE',
    'TELEFONO APAGADO',
    'FDS/NE',
    'FAILED',
    'IVR FALLIDA',
    'NO CONTESTAN',
    'NO ANSWER',
    'BUSY',
    'EXITWITHTIMEOUT',
    'CONGESTION',
    'ABANDON',
    'SIN CONTACTO',
    'CORTAN LA LLAMADA'
}


def normalizar_texto(serie):
    return serie.astype(str).str.strip().str.upper()


def calcular_last_date_feature(df, group_cols, mask_col, fecha_col, output_col):
    """
    Genera la última fecha histórica de un evento específico,
    usando solo información previa a la fila actual.
    """
    temp = df[fecha_col].where(df[mask_col] == 1)
    last_seen = temp.groupby([df[c] for c in group_cols]).ffill()
    df[output_col] = last_seen.groupby([df[c] for c in group_cols]).shift(1)
    return df


def agregar_features_historicas(df, col_resultado):
    """
    Crea variables históricas por (DNI, Telefono), usando solo el pasado de cada gestión.
    """
    df = df.copy()

    if 'Fecha_de_gestion' not in df.columns:
        raise ValueError("La columna 'Fecha_de_gestion' es obligatoria para crear features históricas.")

    # Orden temporal
    df = df.sort_values(['DNI', 'Telefono', 'Fecha_de_gestion']).reset_index(drop=True)

    # Señales base
    df['is_contacto_directo'] = (df[col_resultado] == 'CONTACTO DIRECTO').astype(int)
    df['is_contacto_indirecto'] = (df[col_resultado] == 'CONTACTO INDIRECTO').astype(int)
    df['is_sms'] = (df[col_resultado] == 'SMS').astype(int)
    df['is_no_contacto'] = (df[col_resultado] == 'NO CONTACTO').astype(int)

    if 'Resultado_Gestion' in df.columns:
        df['Resultado_Gestion'] = normalizar_texto(df['Resultado_Gestion'])
        df['is_fallo_detalle'] = df['Resultado_Gestion'].isin(NEGATIVOS_DETALLE).astype(int)
    else:
        df['is_fallo_detalle'] = 0

    group_cols = ['DNI', 'Telefono']

    # Conteos históricos previos
    grp = df.groupby(group_cols, sort=False)

    df['feature_total_gestiones_previas'] = grp.cumcount()

    df['feature_contactos_directos_previos'] = (
        grp['is_contacto_directo'].cumsum() - df['is_contacto_directo']
    )

    df['feature_contactos_indirectos_previos'] = (
        grp['is_contacto_indirecto'].cumsum() - df['is_contacto_indirecto']
    )

    df['feature_sms_previos'] = (
        grp['is_sms'].cumsum() - df['is_sms']
    )

    df['feature_no_contactos_previos'] = (
        grp['is_no_contacto'].cumsum() - df['is_no_contacto']
    )

    df['feature_fallos_previos'] = (
        grp['is_fallo_detalle'].cumsum() - df['is_fallo_detalle']
    )

    # Últimas fechas históricas
    df = calcular_last_date_feature(
        df, group_cols, 'is_contacto_directo', 'Fecha_de_gestion', 'ultima_fecha_contacto_directo_previo'
    )
    df = calcular_last_date_feature(
        df, group_cols, 'is_contacto_indirecto', 'Fecha_de_gestion', 'ultima_fecha_contacto_indirecto_previo'
    )
    df = calcular_last_date_feature(
        df, group_cols, 'is_fallo_detalle', 'Fecha_de_gestion', 'ultima_fecha_fallo_previo'
    )
    df = calcular_last_date_feature(
        df, group_cols, 'is_sms', 'Fecha_de_gestion', 'ultima_fecha_sms_previo'
    )
    df = calcular_last_date_feature(
        df, group_cols, 'is_no_contacto', 'Fecha_de_gestion', 'ultima_fecha_no_contacto_previo'
    )

    # Recencia en días
    fecha_actual = df['Fecha_de_gestion']

    df['feature_dias_desde_ultimo_cd'] = (
        fecha_actual - df['ultima_fecha_contacto_directo_previo']
    ).dt.days.fillna(9999)

    df['feature_dias_desde_ultimo_ci'] = (
        fecha_actual - df['ultima_fecha_contacto_indirecto_previo']
    ).dt.days.fillna(9999)

    df['feature_dias_desde_ultimo_fallo'] = (
        fecha_actual - df['ultima_fecha_fallo_previo']
    ).dt.days.fillna(9999)

    df['feature_dias_desde_ultimo_sms'] = (
        fecha_actual - df['ultima_fecha_sms_previo']
    ).dt.days.fillna(9999)

    df['feature_dias_desde_ultimo_no_contacto'] = (
        fecha_actual - df['ultima_fecha_no_contacto_previo']
    ).dt.days.fillna(9999)

    # Ratios históricos
    denom = df['feature_total_gestiones_previas'].replace(0, np.nan)

    df['feature_ratio_cd_previo'] = (
        df['feature_contactos_directos_previos'] / denom
    ).fillna(0)

    df['feature_ratio_ci_previo'] = (
        df['feature_contactos_indirectos_previos'] / denom
    ).fillna(0)

    df['feature_ratio_fallos_previo'] = (
        df['feature_fallos_previos'] / denom
    ).fillna(0)

    return df


def entrenar_modelo():
    print(">>> Iniciando preparación de datos para ML...")

    path_gestiones = 'data/raw/gestiones_muestra.csv'
    path_opsitel = 'data/raw/base_opsitel.xlsx'
    path_act = 'data/raw/fecha_activacion.xlsx'
    path_blacklist = 'data/raw/blacklist_telefonos.xlsx'
    output_model = 'data/output/modelo_contactabilidad.pkl'

    os.makedirs(os.path.dirname(output_model), exist_ok=True)

    # 1. Cargar archivos
    df_gestiones = pd.read_csv(path_gestiones, dtype={'Telefono': str, 'DNI': str})
    df_opsitel = pd.read_excel(path_opsitel)
    df_act = pd.read_excel(path_act)

    df_black = None
    if os.path.exists(path_blacklist):
        df_black = pd.read_excel(path_blacklist)

    # 2. Limpieza base
    for df in [df_gestiones, df_opsitel, df_act]:
        if 'Telefono' in df.columns:
            df['Telefono'] = limpiar_tel(df['Telefono'])
        if 'DNI' in df.columns:
            df['DNI'] = limpiar_dni(df['DNI'])

    if df_black is not None:
        if 'Telefono' in df_black.columns:
            df_black['Telefono'] = limpiar_tel(df_black['Telefono'])
        if 'DNI' in df_black.columns:
            df_black['DNI'] = limpiar_dni(df_black['DNI'])

    # 3. Normalizar fecha
    if 'Fecha_de_gestion' in df_gestiones.columns:
        df_gestiones['Fecha_de_gestion'] = pd.to_datetime(
            df_gestiones['Fecha_de_gestion'],
            errors='coerce'
        )
    else:
        raise ValueError("El archivo de gestiones debe tener la columna 'Fecha_de_gestion'.")

    # 4. Detectar columna correcta para target
    if 'resultado' in df_gestiones.columns:
        col_resultado = 'resultado'
    elif 'Resultado_Gestion' in df_gestiones.columns:
        col_resultado = 'Resultado_Gestion'
    else:
        raise ValueError(
            "No se encontró la columna de resultado. Se esperaba 'resultado' o 'Resultado_Gestion'."
        )

    print(f">>> Columna detectada para target: {col_resultado}")

    # 5. Construir dataset base
    cols_opsitel = [c for c in ['DNI', 'Telefono', 'Activo'] if c in df_opsitel.columns]
    cols_act = [c for c in ['DNI', 'Telefono', 'FECHA_ACT'] if c in df_act.columns]

    df = df_gestiones.merge(df_opsitel[cols_opsitel], on=['DNI', 'Telefono'], how='left')
    df = df.merge(df_act[cols_act], on=['DNI', 'Telefono'], how='left')

    print(f">>> Registros iniciales luego del merge: {len(df)}")

    # 6. Aplicar filtros del main
    df = df.drop_duplicates().copy()
    print(f">>> Luego de drop_duplicates: {len(df)}")

    df = df[df['Telefono'].astype(str) != df['DNI'].astype(str)].copy()
    print(f">>> Luego de excluir Telefono == DNI: {len(df)}")

    print(">>> Validando estructura de números y prefijos regionales en entrenamiento...")
    mask_validos = df.apply(validar_telefono_peru, axis=1)
    df = df[mask_validos].copy()
    print(f">>> Luego de validar telefono_peru: {len(df)}")

    if df_black is not None and {'DNI', 'Telefono', 'Motivo'}.issubset(df_black.columns):
        df = pd.merge(
            df,
            df_black[['DNI', 'Telefono', 'Motivo']],
            on=['DNI', 'Telefono'],
            how='left'
        )
        df = df[df['Motivo'].isna()].copy()
        print(f">>> Luego de filtrar blacklist: {len(df)}")

    # 7. Ingeniería de variables base
    if 'Activo' not in df.columns:
        df['Activo'] = None

    df['feature_opsitel'] = df['Activo'].apply(
        lambda x: 1 if str(x).upper().strip() == 'SI' else 0
    )

    if 'FECHA_ACT' not in df.columns:
        df['FECHA_ACT'] = pd.NaT

    df['FECHA_ACT'] = pd.to_datetime(df['FECHA_ACT'], errors='coerce')

    df['feature_antiguedad_meses'] = (
        (2026 - df['FECHA_ACT'].dt.year) * 12 +
        (3 - df['FECHA_ACT'].dt.month)
    ).fillna(300)

    # 8. Normalizar resultado y crear features históricas
    df[col_resultado] = normalizar_texto(df[col_resultado])

    print(">>> Valores únicos normalizados de resultado:")
    print(df[col_resultado].value_counts(dropna=False).head(20))

    df = agregar_features_historicas(df, col_resultado)

    # 9. Crear target
    df['target'] = (df[col_resultado] == 'CONTACTO DIRECTO').astype(int)

    print(">>> CONTACTO DIRECTO después de filtros:", (df[col_resultado] == 'CONTACTO DIRECTO').sum())
    print(">>> Distribución del target:")
    print(df['target'].value_counts(dropna=False))

    # 10. Dataset final
    feature_cols = [
        'feature_opsitel',
        'feature_antiguedad_meses',
        'feature_total_gestiones_previas',
        'feature_contactos_directos_previos',
        'feature_contactos_indirectos_previos',
        'feature_sms_previos',
        'feature_no_contactos_previos',
        'feature_fallos_previos',
        'feature_dias_desde_ultimo_cd',
        'feature_dias_desde_ultimo_ci',
        'feature_dias_desde_ultimo_fallo',
        'feature_dias_desde_ultimo_sms',
        'feature_dias_desde_ultimo_no_contacto',
        'feature_ratio_cd_previo',
        'feature_ratio_ci_previo',
        'feature_ratio_fallos_previo',
    ]

    df_model = df[['DNI', 'Telefono'] + feature_cols + ['target']].copy()
    df_model[feature_cols] = df_model[feature_cols].fillna(0)

    if df_model.empty:
        raise ValueError("No quedaron registros válidos para entrenar el modelo.")

    print(f">>> Registros válidos para entrenamiento: {len(df_model)}")
    print(">>> Distribución final del target:")
    print(df_model['target'].value_counts(dropna=False))

    if df_model['target'].nunique() < 2:
        raise ValueError("El target solo tiene una clase después de los filtros.")

    # 11. Entrenamiento
    X = df_model[feature_cols]
    y = df_model['target']

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print(">>> Entrenando RandomForest con features históricas...")
    modelo = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight='balanced',
        max_depth=12,
        min_samples_split=8,
        min_samples_leaf=3,
        n_jobs=-1
    )
    modelo.fit(X_train, y_train)

    # 12. Evaluación
    print(">>> Evaluación del modelo:")
    y_pred = modelo.predict(X_test)
    print(classification_report(y_test, y_pred, digits=4))

    # 13. Importancia de variables
    importancia = pd.DataFrame({
        'feature': feature_cols,
        'importance': modelo.feature_importances_
    }).sort_values('importance', ascending=False)

    print(">>> Top 15 features más importantes:")
    print(importancia.head(15))

    # 14. Guardar modelo
    joblib.dump(modelo, output_model)
    print(f">>> Modelo guardado exitosamente en: {output_model}")


if __name__ == "__main__":
    entrenar_modelo()