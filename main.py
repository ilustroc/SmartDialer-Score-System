import os
import joblib
import pandas as pd

from src.logic import (
    calcular_score_base,
    limpiar_tel,
    limpiar_dni,
    validar_telefono_peru,
    obtener_columna_resultado,
    normalizar_texto,
)

def ejecutar_ranking_ia():
    print(">>> Iniciando SmartDialer Engine (Limpieza de Negativos)...")

    temp_csv = 'data/raw/temp_base_completa.csv'
    path_blacklist = 'data/raw/blacklist_telefonos.xlsx'
    path_opsitel = 'data/raw/base_opsitel.xlsx'
    path_act = 'data/raw/fecha_activacion.xlsx'
    path_modelo = 'data/output/modelo_contactabilidad.pkl'

    # 1. Cargar maestros y modelo
    df_opsitel = pd.read_excel(path_opsitel)
    df_act = pd.read_excel(path_act)
    df_black = pd.read_excel(path_blacklist)
    modelo_ia = joblib.load(path_modelo)

    for df in [df_opsitel, df_act, df_black]:
        if 'Telefono' in df.columns:
            df['Telefono'] = limpiar_tel(df['Telefono'])
        if 'DNI' in df.columns:
            df['DNI'] = limpiar_dni(df['DNI'])

    # 2. Cargar historial
    df_historial = None
    if os.path.exists(temp_csv):
        df_historial = pd.read_csv(temp_csv, dtype={'Telefono': str, 'DNI': str})
        df_historial['Telefono'] = limpiar_tel(df_historial['Telefono'])
        df_historial['DNI'] = limpiar_dni(df_historial['DNI'])

        if 'Fecha_de_gestion' in df_historial.columns:
            df_historial['Fecha_de_gestion'] = pd.to_datetime(
                df_historial['Fecha_de_gestion'],
                errors='coerce'
            )

    # 3. Construir universo total
    df_universo = pd.merge(df_opsitel, df_act, on=['DNI', 'Telefono'], how='outer')

    if df_historial is not None and not df_historial.empty:
        df_tels_gestion = df_historial[['DNI', 'Telefono']].drop_duplicates()
        df_universo = pd.merge(
            df_universo,
            df_tels_gestion,
            on=['DNI', 'Telefono'],
            how='outer'
        )

    # 4. Filtrado maestro
    df_universo = df_universo.drop_duplicates(subset=['DNI', 'Telefono']).copy()

    df_universo = df_universo[
        df_universo['Telefono'].astype(str) != df_universo['DNI'].astype(str)
    ].copy()

    print(">>> Validando estructura de números y prefijos regionales...")
    mask_validos = df_universo.apply(validar_telefono_peru, axis=1)
    df_universo = df_universo[mask_validos].copy()

    df_universo = pd.merge(
        df_universo,
        df_black[['DNI', 'Telefono', 'Motivo']],
        on=['DNI', 'Telefono'],
        how='left'
    )
    df_universo = df_universo[df_universo['Motivo'].isna()].copy()

    # 5. Scoring base IA con nuevas features
    fecha_referencia = None
    if df_historial is not None and 'Fecha_de_gestion' in df_historial.columns:
        fechas_validas = df_historial['Fecha_de_gestion'].dropna()
        if len(fechas_validas) > 0:
            fecha_referencia = fechas_validas.max().normalize()

    df_scored = calcular_score_base(
        df_universo=df_universo,
        modelo=modelo_ia,
        df_historial=df_historial,
        fecha_referencia=fecha_referencia
    )

    # 6. Procesar fallos y éxitos históricos
    if df_historial is not None and not df_historial.empty:
        if 'Resultado_Gestion' in df_historial.columns:
            df_historial['Resultado_Gestion_norm'] = normalizar_texto(df_historial['Resultado_Gestion'])
        else:
            df_historial['Resultado_Gestion_norm'] = ''

        col_resultado = obtener_columna_resultado(df_historial)
        if col_resultado is not None:
            df_historial['resultado_norm'] = normalizar_texto(df_historial[col_resultado])
        else:
            df_historial['resultado_norm'] = ''

        negativos = ['NRO. NO PERTENECE', 'TELEFONO APAGADO', 'FDS/NE', 'FAILED', 'IVR FALLIDA']
        df_historial['es_fallo'] = df_historial['Resultado_Gestion_norm'].isin(negativos).astype(int)

        stats_fallos = df_historial.groupby(['DNI', 'Telefono']).agg(
            total_fallos=('es_fallo', 'sum'),
            ultima_fecha_fallo=('Fecha_de_gestion', 'max')
        ).reset_index()

        df_scored = pd.merge(
            df_scored,
            stats_fallos,
            on=['DNI', 'Telefono'],
            how='left'
        )

        df_scored['total_fallos'] = df_scored['total_fallos'].fillna(0)
        df_scored['ultima_fecha_fallo'] = pd.to_datetime(df_scored['ultima_fecha_fallo'], errors='coerce')

        exitosos = df_historial[
            df_historial['resultado_norm'].isin(['CONTACTO DIRECTO', 'CONTACTO INDIRECTO'])
        ].copy()

        if not exitosos.empty:
            exitosos = exitosos.sort_values('Fecha_de_gestion', ascending=False)

            ultima_gestion_exito = exitosos.groupby(['DNI', 'Telefono']).agg(
                ultima_fecha_exito=('Fecha_de_gestion', 'max'),
                mejor_resultado=('resultado_norm', 'first')
            ).reset_index()

            df_scored = pd.merge(
                df_scored,
                ultima_gestion_exito,
                on=['DNI', 'Telefono'],
                how='left'
            )

            df_scored.loc[df_scored['mejor_resultado'] == 'CONTACTO DIRECTO', 'total_score'] += 100.0
            df_scored.loc[df_scored['mejor_resultado'] == 'CONTACTO INDIRECTO', 'total_score'] += 50.0
        else:
            df_scored['ultima_fecha_exito'] = pd.NaT
            df_scored['mejor_resultado'] = None
    else:
        df_scored['total_fallos'] = 0
        df_scored['ultima_fecha_fallo'] = pd.NaT
        df_scored['ultima_fecha_exito'] = pd.NaT
        df_scored['mejor_resultado'] = None

    # 7. Identificar y exportar números descartados
    mask_descarte = (
        (df_scored['total_fallos'] >= 3) &
        (df_scored['ultima_fecha_exito'].isnull())
    )

    df_descartados = df_scored[mask_descarte].copy()
    df_descartados['MOTIVO'] = 'Exceso de gestiones negativas reincidentes'

    if not df_descartados.empty:
        df_descartados['DNI'] = df_descartados['DNI'].apply(lambda x: str(int(x)).zfill(8))
        df_descartados[['DNI', 'Telefono', 'MOTIVO']].to_excel(
            'data/output/telefonos_descartados.xlsx',
            index=False
        )
        print(f">>> {len(df_descartados)} teléfonos movidos a lista de descartados por reincidencia.")

    # 8. Filtrar salida final
    df_scored_final = df_scored[~mask_descarte].copy()

    col_fecha = 'ultima_fecha_exito' if 'ultima_fecha_exito' in df_scored_final.columns else 'total_score'

    df_sorted = df_scored_final.sort_values(
        by=['DNI', 'total_score', col_fecha],
        ascending=[True, False, False]
    )

    df_top3 = df_sorted.groupby('DNI').head(3).copy()

    # 9. Exportar horizontal
    print('>>> Generando lista_final_horizontal.csv con formato de texto y comillas...')
    with open('data/output/lista_final_horizontal.csv', 'w', encoding='utf-8-sig') as f:
        f.write('"DNI";"Telefono_1";"Telefono_2";"Telefono_3"\n')

        for dni, grupo in df_top3.groupby('DNI'):
            dni_str = str(int(dni)).zfill(8)
            tels = [str(t).strip() for t in grupo['Telefono'].tolist() if pd.notnull(t) and str(t).strip() != '']

            while len(tels) < 3:
                tels.append('')

            f.write(f'"{dni_str}";"{tels[0]}";"{tels[1]}";"{tels[2]}"\n')

    # 10. Exportar explicación
    df_sorted['DNI'] = df_sorted['DNI'].apply(lambda x: str(int(x)).zfill(8))
    df_sorted.to_csv('data/output/explicacion_score.csv', index=False, encoding='utf-8-sig')

    print(">>> Proceso terminado con éxito.")


if __name__ == "__main__":
    ejecutar_ranking_ia()