import pandas as pd
import numpy as np


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

NEGATIVOS_CRITICOS_DESCARTE = {
    'NRO. NO PERTENECE',
    'TELEFONO APAGADO',
    'FDS/NE',
    'FAILED',
    'IVR FALLIDA',
}

RESULTADOS_CONTACTO_VALIDO = {
    'CONTACTO DIRECTO',
    'CONTACTO INDIRECTO',
}

DETALLES_PROMESA = {
    'COMPROMISO DE PAGO',
    'PROMESA DE PAGO',
    'CANCELO DEUDA',
}

FEATURE_COLS_MODELO = [
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


def limpiar_tel(serie):
    """Limpia el formato de teléfono eliminando decimales .0 y espacios."""
    return serie.astype(str).str.replace(r'\.0$', '', regex=True).str.strip()


def limpiar_dni(serie):
    """Convierte el DNI a entero, manejando errores y nulos."""
    return pd.to_numeric(serie, errors='coerce').fillna(0).astype(int)


def normalizar_texto(serie):
    """Normaliza textos para comparaciones."""
    return serie.astype(str).str.strip().str.upper()


def obtener_columna_resultado(df):
    """Devuelve la columna general de resultado priorizando 'resultado'."""
    if 'resultado' in df.columns:
        return 'resultado'
    if 'Resultado_Gestion' in df.columns:
        return 'Resultado_Gestion'
    return None


def construir_resumen_contactabilidad(df_historial):
    """
    Resume fallos, contactos y promesas por (DNI, Telefono).

    La regla de descarte usa solo fallos criticos posteriores al ultimo
    contacto/promesa para no castigar telefonos recuperados por una gestion
    positiva reciente.
    """
    cols_salida = [
        'DNI',
        'Telefono',
        'total_fallos',
        'fallos_posteriores_proteccion',
        'ultima_fecha_fallo',
        'ultima_fecha_fallo_posterior',
        'ultima_fecha_exito',
        'ultima_fecha_promesa',
        'fecha_ultima_proteccion',
        'mejor_resultado',
        'motivo_proteccion',
        'tiene_promesa',
    ]

    if df_historial is None or df_historial.empty:
        return pd.DataFrame(columns=cols_salida)

    if 'DNI' not in df_historial.columns or 'Telefono' not in df_historial.columns:
        return pd.DataFrame(columns=cols_salida)

    df = df_historial.copy()
    df['Telefono'] = limpiar_tel(df['Telefono'])
    df['DNI'] = limpiar_dni(df['DNI'])

    if 'Fecha_de_gestion' in df.columns:
        df['Fecha_de_gestion'] = pd.to_datetime(df['Fecha_de_gestion'], errors='coerce')
    else:
        df['Fecha_de_gestion'] = pd.NaT

    if 'Resultado_Gestion' in df.columns:
        df['Resultado_Gestion_norm'] = normalizar_texto(df['Resultado_Gestion'])
    else:
        df['Resultado_Gestion_norm'] = ''

    col_resultado = obtener_columna_resultado(df)
    if col_resultado is not None:
        df['resultado_norm'] = normalizar_texto(df[col_resultado])
    else:
        df['resultado_norm'] = ''

    if 'fecha_promesa' in df.columns:
        df['fecha_promesa_norm'] = pd.to_datetime(df['fecha_promesa'], errors='coerce')
    else:
        df['fecha_promesa_norm'] = pd.NaT

    if 'monto_promesa' in df.columns:
        df['monto_promesa_num'] = pd.to_numeric(df['monto_promesa'], errors='coerce').fillna(0)
    else:
        df['monto_promesa_num'] = 0

    df['es_fallo'] = df['Resultado_Gestion_norm'].isin(NEGATIVOS_CRITICOS_DESCARTE)
    df['es_promesa'] = (
        df['Resultado_Gestion_norm'].isin(DETALLES_PROMESA) |
        df['Resultado_Gestion_norm'].str.contains('COMPROMISO|PROMESA', regex=True, na=False) |
        df['fecha_promesa_norm'].notna() |
        (df['monto_promesa_num'] > 0)
    )
    df['es_exito'] = df['resultado_norm'].isin(RESULTADOS_CONTACTO_VALIDO) | df['es_promesa']

    grupos = ['DNI', 'Telefono']

    fallos = df[df['es_fallo']].copy()
    if not fallos.empty:
        stats_fallos = fallos.groupby(grupos).agg(
            total_fallos=('es_fallo', 'size'),
            ultima_fecha_fallo=('Fecha_de_gestion', 'max'),
        ).reset_index()
    else:
        stats_fallos = pd.DataFrame(columns=grupos + ['total_fallos', 'ultima_fecha_fallo'])

    exitos = df[df['es_exito']].copy()
    if not exitos.empty:
        exitos['fecha_exito_evento'] = exitos['Fecha_de_gestion']
        exitos.loc[~exitos['es_exito'], 'fecha_exito_evento'] = pd.NaT
        exitos['fecha_promesa_evento'] = exitos['fecha_promesa_norm'].where(exitos['es_promesa'])
        exitos['fecha_proteccion_evento'] = exitos[
            ['fecha_exito_evento', 'fecha_promesa_evento']
        ].max(axis=1)

        stats_exitos = exitos.groupby(grupos).agg(
            ultima_fecha_exito=('Fecha_de_gestion', 'max'),
            ultima_fecha_promesa=('fecha_promesa_evento', 'max'),
            fecha_ultima_proteccion=('fecha_proteccion_evento', 'max'),
            tiene_promesa=('es_promesa', 'max'),
        ).reset_index()

        ultimos = exitos.sort_values(
            grupos + ['fecha_proteccion_evento', 'Fecha_de_gestion'],
            ascending=[True, True, False, False],
        ).drop_duplicates(grupos)

        ultimos = ultimos[grupos + ['resultado_norm', 'Resultado_Gestion_norm']].rename(
            columns={
                'resultado_norm': 'mejor_resultado',
                'Resultado_Gestion_norm': 'motivo_proteccion',
            }
        )

        stats_exitos = pd.merge(stats_exitos, ultimos, on=grupos, how='left')
    else:
        stats_exitos = pd.DataFrame(
            columns=grupos + [
                'ultima_fecha_exito',
                'ultima_fecha_promesa',
                'fecha_ultima_proteccion',
                'tiene_promesa',
                'mejor_resultado',
                'motivo_proteccion',
            ]
        )

    resumen = pd.merge(stats_fallos, stats_exitos, on=grupos, how='outer')

    if not fallos.empty:
        fallos = pd.merge(
            fallos,
            resumen[grupos + ['fecha_ultima_proteccion']],
            on=grupos,
            how='left',
        )

        fallos['es_fallo_posterior_proteccion'] = (
            fallos['fecha_ultima_proteccion'].isna() |
            (fallos['Fecha_de_gestion'] > fallos['fecha_ultima_proteccion'])
        )

        fallos_posteriores = fallos[fallos['es_fallo_posterior_proteccion']].groupby(grupos).agg(
            fallos_posteriores_proteccion=('es_fallo_posterior_proteccion', 'size'),
            ultima_fecha_fallo_posterior=('Fecha_de_gestion', 'max'),
        ).reset_index()
    else:
        fallos_posteriores = pd.DataFrame(
            columns=grupos + ['fallos_posteriores_proteccion', 'ultima_fecha_fallo_posterior']
        )

    resumen = pd.merge(resumen, fallos_posteriores, on=grupos, how='left')

    for col in ['total_fallos', 'fallos_posteriores_proteccion']:
        resumen[col] = resumen[col].fillna(0).astype(int)

    for col in [
        'ultima_fecha_fallo',
        'ultima_fecha_fallo_posterior',
        'ultima_fecha_exito',
        'ultima_fecha_promesa',
        'fecha_ultima_proteccion',
    ]:
        resumen[col] = pd.to_datetime(resumen[col], errors='coerce')

    resumen['tiene_promesa'] = resumen['tiene_promesa'].astype('boolean').fillna(False).astype(bool)
    resumen['mejor_resultado'] = resumen['mejor_resultado'].fillna('')
    resumen['motivo_proteccion'] = resumen['motivo_proteccion'].fillna('')

    return resumen[cols_salida].copy()


def validar_telefono_peru(row):
    """
    Reglas de validación de telefonía peruana:
    - Retorna False si el DNI es igual al Teléfono.
    - Solo 8 o 9 dígitos son válidos.
    - 9 dígitos: Debe iniciar con 9.
    - 8 dígitos: No puede iniciar con 11.
      Si inicia con 1 es válido.
      Si inicia con otro número, debe estar en prefijos regionales permitidos.
    """
    tel = str(row['Telefono']).strip()
    dni = str(row['DNI']).strip()
    largo = len(tel)

    if tel == dni:
        return False

    if largo not in [8, 9]:
        return False

    if largo == 9:
        return tel.startswith('9')

    if largo == 8:
        if tel.startswith('11'):
            return False

        if tel.startswith('1'):
            return True

        prefijos_permitidos = [
            '41', '43', '83', '54', '66', '76', '84', '67', '62', '56',
            '64', '44', '74', '65', '82', '53', '63', '73', '51', '42',
            '52', '72', '61'
        ]

        return tel[:2] in prefijos_permitidos

    return False


def construir_features_historial(df_historial, fecha_referencia=None):
    """
    Construye features históricas agregadas por (DNI, Telefono)
    para usar en scoring productivo.
    """
    if df_historial is None or df_historial.empty:
        return pd.DataFrame(columns=['DNI', 'Telefono'] + FEATURE_COLS_MODELO[2:])

    df = df_historial.copy()

    if 'Telefono' not in df.columns or 'DNI' not in df.columns:
        return pd.DataFrame(columns=['DNI', 'Telefono'] + FEATURE_COLS_MODELO[2:])

    df['Telefono'] = limpiar_tel(df['Telefono'])
    df['DNI'] = limpiar_dni(df['DNI'])

    if 'Fecha_de_gestion' in df.columns:
        df['Fecha_de_gestion'] = pd.to_datetime(df['Fecha_de_gestion'], errors='coerce')
    else:
        df['Fecha_de_gestion'] = pd.NaT

    col_resultado = obtener_columna_resultado(df)
    if col_resultado is None:
        df['resultado_norm'] = ''
    else:
        df['resultado_norm'] = normalizar_texto(df[col_resultado])

    if 'Resultado_Gestion' in df.columns:
        df['resultado_gestion_norm'] = normalizar_texto(df['Resultado_Gestion'])
    else:
        df['resultado_gestion_norm'] = ''

    df['is_cd'] = (df['resultado_norm'] == 'CONTACTO DIRECTO').astype(int)
    df['is_ci'] = (df['resultado_norm'] == 'CONTACTO INDIRECTO').astype(int)
    df['is_sms'] = (df['resultado_norm'] == 'SMS').astype(int)
    df['is_no_contacto'] = (df['resultado_norm'] == 'NO CONTACTO').astype(int)
    df['is_fallo_detalle'] = df['resultado_gestion_norm'].isin(NEGATIVOS_DETALLE).astype(int)

    if fecha_referencia is None:
        fechas_validas = df['Fecha_de_gestion'].dropna()
        if len(fechas_validas) > 0:
            fecha_referencia = fechas_validas.max().normalize()
        else:
            fecha_referencia = pd.Timestamp.today().normalize()
    else:
        fecha_referencia = pd.to_datetime(fecha_referencia, errors='coerce')
        if pd.isna(fecha_referencia):
            fecha_referencia = pd.Timestamp.today().normalize()

    agrupado = df.groupby(['DNI', 'Telefono'], dropna=False)

    resumen = agrupado.agg(
        feature_total_gestiones_previas=('Telefono', 'size'),
        feature_contactos_directos_previos=('is_cd', 'sum'),
        feature_contactos_indirectos_previos=('is_ci', 'sum'),
        feature_sms_previos=('is_sms', 'sum'),
        feature_no_contactos_previos=('is_no_contacto', 'sum'),
        feature_fallos_previos=('is_fallo_detalle', 'sum'),
        ultima_fecha_cd=('Fecha_de_gestion', lambda s: s[df.loc[s.index, 'is_cd'] == 1].max() if ((df.loc[s.index, 'is_cd'] == 1).any()) else pd.NaT),
        ultima_fecha_ci=('Fecha_de_gestion', lambda s: s[df.loc[s.index, 'is_ci'] == 1].max() if ((df.loc[s.index, 'is_ci'] == 1).any()) else pd.NaT),
        ultima_fecha_fallo=('Fecha_de_gestion', lambda s: s[df.loc[s.index, 'is_fallo_detalle'] == 1].max() if ((df.loc[s.index, 'is_fallo_detalle'] == 1).any()) else pd.NaT),
        ultima_fecha_sms=('Fecha_de_gestion', lambda s: s[df.loc[s.index, 'is_sms'] == 1].max() if ((df.loc[s.index, 'is_sms'] == 1).any()) else pd.NaT),
        ultima_fecha_no_contacto=('Fecha_de_gestion', lambda s: s[df.loc[s.index, 'is_no_contacto'] == 1].max() if ((df.loc[s.index, 'is_no_contacto'] == 1).any()) else pd.NaT),
    ).reset_index()

    resumen['feature_dias_desde_ultimo_cd'] = (fecha_referencia - resumen['ultima_fecha_cd']).dt.days.fillna(9999)
    resumen['feature_dias_desde_ultimo_ci'] = (fecha_referencia - resumen['ultima_fecha_ci']).dt.days.fillna(9999)
    resumen['feature_dias_desde_ultimo_fallo'] = (fecha_referencia - resumen['ultima_fecha_fallo']).dt.days.fillna(9999)
    resumen['feature_dias_desde_ultimo_sms'] = (fecha_referencia - resumen['ultima_fecha_sms']).dt.days.fillna(9999)
    resumen['feature_dias_desde_ultimo_no_contacto'] = (fecha_referencia - resumen['ultima_fecha_no_contacto']).dt.days.fillna(9999)

    denom = resumen['feature_total_gestiones_previas'].replace(0, np.nan)

    resumen['feature_ratio_cd_previo'] = (resumen['feature_contactos_directos_previos'] / denom).fillna(0)
    resumen['feature_ratio_ci_previo'] = (resumen['feature_contactos_indirectos_previos'] / denom).fillna(0)
    resumen['feature_ratio_fallos_previo'] = (resumen['feature_fallos_previos'] / denom).fillna(0)

    cols_finales = ['DNI', 'Telefono'] + FEATURE_COLS_MODELO[2:]
    return resumen[cols_finales].copy()


def preparar_features_modelo(df_universo, df_historial=None, fecha_referencia=None):
    """
    Prepara todas las features que el modelo espera para scoring.
    """
    df = df_universo.copy()

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

    df_hist_features = construir_features_historial(df_historial, fecha_referencia=fecha_referencia)

    df = pd.merge(
        df,
        df_hist_features,
        on=['DNI', 'Telefono'],
        how='left'
    )

    for col in FEATURE_COLS_MODELO:
        if col not in df.columns:
            df[col] = 0

    df[FEATURE_COLS_MODELO] = df[FEATURE_COLS_MODELO].fillna(0)

    return df


def calcular_score_base(df_universo, modelo, df_historial=None, fecha_referencia=None):
    """
    Calcula la probabilidad de contacto usando el modelo entrenado
    con features base + históricas.
    """
    df = preparar_features_modelo(
        df_universo=df_universo,
        df_historial=df_historial,
        fecha_referencia=fecha_referencia
    )

    X = df[FEATURE_COLS_MODELO].copy()
    df['total_score'] = modelo.predict_proba(X)[:, 1]

    return df[['DNI', 'Telefono', 'total_score'] + FEATURE_COLS_MODELO].copy()
