import os
import joblib
import pandas as pd

from src.logic import (
    calcular_score_base,
    limpiar_tel,
    limpiar_dni,
    validar_telefono_peru,
    construir_resumen_contactabilidad,
)


def generar_ivr_kpi():
    print(">>> Iniciando SmartDialer Engine para IVR (Filtro temporal de reincidencia x3)...")
    temp_csv = 'data/raw/temp_base_completa.csv'
    path_opsitel = 'data/raw/base_opsitel.xlsx'
    path_act = 'data/raw/fecha_activacion.xlsx'
    path_modelo = 'data/output/modelo_contactabilidad.pkl'

    # 1. Cargar maestros, modelo e historial
    df_opsitel = pd.read_excel(path_opsitel)
    df_act = pd.read_excel(path_act)
    modelo_ia = joblib.load(path_modelo)

    for df in [df_opsitel, df_act]:
        df['Telefono'] = limpiar_tel(df['Telefono'])
        df['DNI'] = limpiar_dni(df['DNI'])

    df_historial = None
    resumen_contactabilidad = pd.DataFrame()
    if os.path.exists(temp_csv):
        df_historial = pd.read_csv(temp_csv, dtype={'Telefono': str, 'DNI': str}, low_memory=False)
        df_historial['Telefono'] = limpiar_tel(df_historial['Telefono'])
        df_historial['DNI'] = limpiar_dni(df_historial['DNI'])

        if 'Fecha_de_gestion' in df_historial.columns:
            df_historial['Fecha_de_gestion'] = pd.to_datetime(
                df_historial['Fecha_de_gestion'],
                errors='coerce'
            )

        resumen_contactabilidad = construir_resumen_contactabilidad(df_historial)

    # 2. Construir universo unico
    df_universo = pd.merge(df_opsitel, df_act, on=['DNI', 'Telefono'], how='outer')

    if df_historial is not None and not df_historial.empty:
        df_tels_gestion = df_historial[['DNI', 'Telefono']].drop_duplicates()
        df_universo = pd.merge(
            df_universo,
            df_tels_gestion,
            on=['DNI', 'Telefono'],
            how='outer'
        )

    df_universo = df_universo.drop_duplicates(subset=['DNI', 'Telefono']).copy()

    # 3. Aplicar filtros de calidad
    df_universo = df_universo[
        df_universo['Telefono'].astype(str) != df_universo['DNI'].astype(str)
    ].copy()

    mask_validos = df_universo.apply(validar_telefono_peru, axis=1)
    df_universo = df_universo[mask_validos].copy()

    if not resumen_contactabilidad.empty:
        descartes = resumen_contactabilidad[
            resumen_contactabilidad['fallos_posteriores_proteccion'] >= 3
        ][['DNI', 'Telefono']].copy()

        if not descartes.empty:
            count_antes = len(df_universo)
            descartes['descartar_reincidencia'] = True
            df_universo = pd.merge(
                df_universo,
                descartes,
                on=['DNI', 'Telefono'],
                how='left'
            )
            df_universo = df_universo[df_universo['descartar_reincidencia'].isna()].copy()
            df_universo = df_universo.drop(columns=['descartar_reincidencia'])
            print(f">>> [FILTRO] {count_antes - len(df_universo)} registros eliminados por 3+ fallos posteriores al ultimo contacto/promesa.")

    # 4. Scorear con IA
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

    if not resumen_contactabilidad.empty:
        df_scored = pd.merge(
            df_scored,
            resumen_contactabilidad,
            on=['DNI', 'Telefono'],
            how='left'
        )
        df_scored['tiene_promesa'] = df_scored['tiene_promesa'].astype('boolean').fillna(False).astype(bool)
        df_scored['mejor_resultado'] = df_scored['mejor_resultado'].fillna('')

        mask_promesa = df_scored['tiene_promesa']
        mask_cd = (~mask_promesa) & (df_scored['mejor_resultado'] == 'CONTACTO DIRECTO')
        mask_ci = (~mask_promesa) & (df_scored['mejor_resultado'] == 'CONTACTO INDIRECTO')

        df_scored.loc[mask_promesa, 'total_score'] += 150.0
        df_scored.loc[mask_cd, 'total_score'] += 100.0
        df_scored.loc[mask_ci, 'total_score'] += 50.0

    # 5. Seleccion Top 2 por DNI
    col_fecha = 'fecha_ultima_proteccion' if 'fecha_ultima_proteccion' in df_scored.columns else 'total_score'
    df_final = df_scored.sort_values(by=['DNI', 'total_score', col_fecha], ascending=[True, False, False])
    df_ivr = df_final.groupby('DNI').head(2).copy()

    # 6. Exportar formato final
    output_path = 'data/output/CARGA_IVR_KPI.csv'
    with open(output_path, 'w', encoding='utf-8') as f:
        for _, row in df_ivr.iterrows():
            dni_str = str(int(row['DNI'])).zfill(8)
            linea = f"IVR_KPI,{row['Telefono']},documento={dni_str},,9999\n"
            f.write(linea)

    print(f">>> PROCESO COMPLETADO. Total IVR: {len(df_ivr)}")


if __name__ == "__main__":
    generar_ivr_kpi()
