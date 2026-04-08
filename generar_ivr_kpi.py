import pandas as pd
import joblib
import os
from src.logic import (
    calcular_score_base, 
    limpiar_tel, 
    limpiar_dni, 
    validar_telefono_peru, 
    normalizar_texto
)

def generar_ivr_kpi():
    print(">>> Iniciando SmartDialer Engine para IVR (Filtro por Reincidencia x3)...")
    temp_csv = 'data/raw/temp_base_completa.csv'
    path_opsitel = 'data/raw/base_opsitel.xlsx'
    path_act = 'data/raw/fecha_activacion.xlsx'
    path_modelo = 'data/output/modelo_contactabilidad.pkl'

    # 1. Cargar Maestros y Modelo
    df_opsitel = pd.read_excel(path_opsitel)
    df_act = pd.read_excel(path_act)
    modelo_ia = joblib.load(path_modelo)

    for df in [df_opsitel, df_act]:
        df['Telefono'] = limpiar_tel(df['Telefono'])
        df['DNI'] = limpiar_dni(df['DNI'])

    # 2. Identificar números INVÁLIDOS por REINCIDENCIA (3 o más del mismo tipo)
    lista_negra_reincidente = set()
    if os.path.exists(temp_csv):
        df_historial = pd.read_csv(temp_csv, dtype={'Telefono': str, 'DNI': str})
        df_historial['Telefono'] = limpiar_tel(df_historial['Telefono'])
        
        if 'Resultado_Gestion' in df_historial.columns:
            df_historial['res_norm'] = normalizar_texto(df_historial['Resultado_Gestion'])
            
            # Contamos cuántas veces aparece cada resultado por cada teléfono
            conteo_fallos = df_historial.groupby(['Telefono', 'res_norm']).size().reset_index(name='cantidad')
            
            # Filtramos solo los resultados negativos que se repiten 3 o más veces
            negativos_criticos = ['FAILED', 'IVR FALLIDA', 'FDS/NE', 'NRO. NO PERTENECE', 'TELEFONO APAGADO']
            reincidentes = conteo_fallos[
                (conteo_fallos['res_norm'].isin(negativos_criticos)) & 
                (conteo_fallos['cantidad'] >= 3)
            ]
            
            lista_negra_reincidente = set(reincidentes['Telefono'])
            print(f">>> [REINCIDENCIA] {len(lista_negra_reincidente)} números descartados por tener 3+ fallos del mismo tipo.")

    # 3. Construir Universo Único
    df_universo = pd.merge(df_opsitel, df_act, on=['DNI', 'Telefono'], how='outer').drop_duplicates()

    # 4. APLICAR FILTROS DE CALIDAD
    
    # A. Filtro de Identidad: Telefono == DNI
    df_universo = df_universo[df_universo['Telefono'].astype(str) != df_universo['DNI'].astype(str)].copy()

    # B. Filtro de Estructura de Red
    mask_validos = df_universo.apply(validar_telefono_peru, axis=1)
    df_universo = df_universo[mask_validos].copy()

    # C. Filtro de Reincidencia (La nueva regla)
    if lista_negra_reincidente:
        count_antes = len(df_universo)
        df_universo = df_universo[~df_universo['Telefono'].isin(lista_negra_reincidente)].copy()
        print(f">>> [FILTRO] {count_antes - len(df_universo)} registros eliminados por ser reincidentes negativos.")

    # 5. Scorear con IA
    df_scored = calcular_score_base(df_universo, modelo_ia)

    # 6. Selección Top 2 por DNI
    df_final = df_scored.sort_values(by=['DNI', 'total_score'], ascending=[True, False])
    df_ivr = df_final.groupby('DNI').head(2).copy()

    # 7. Exportar formato final
    output_path = 'data/output/CARGA_IVR_KPI.csv'
    with open(output_path, 'w', encoding='utf-8') as f:
        for _, row in df_ivr.iterrows():
            dni_str = str(int(row['DNI'])).zfill(8)
            linea = f"IVR_KPI,{row['Telefono']},documento={dni_str},,9999\n"
            f.write(linea)

    print(f">>> PROCESO COMPLETADO. Total IVR: {len(df_ivr)}")

if __name__ == "__main__":
    generar_ivr_kpi()