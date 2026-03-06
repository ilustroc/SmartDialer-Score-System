import pandas as pd
import joblib
import os
import time
from src.logic import calcular_score_base, limpiar_tel, limpiar_dni

def ejecutar_ranking_ia():
    print(">>> Iniciando SmartDialer Engine (Limpieza de Negativos)...")
    temp_csv = 'data/raw/temp_base_completa.csv'
    path_blacklist = 'data/raw/blacklist_telefonos.xlsx'
    
    # 1. Cargar Maestros y Modelo
    df_opsitel = pd.read_excel('data/raw/base_opsitel.xlsx')
    df_act = pd.read_excel('data/raw/fecha_activacion.xlsx')
    df_black = pd.read_excel(path_blacklist)
    modelo_ia = joblib.load('data/output/modelo_contactabilidad.pkl')

    for df in [df_opsitel, df_act, df_black]:
        df['Telefono'] = limpiar_tel(df['Telefono'])
        df['DNI'] = limpiar_dni(df['DNI'])

    # 2. Cargar Historial
    df_historial = None
    if os.path.exists(temp_csv):
        df_historial = pd.read_csv(temp_csv, dtype={'Telefono': str, 'DNI': str})
        df_historial['Telefono'] = limpiar_tel(df_historial['Telefono'])
        df_historial['DNI'] = limpiar_dni(df_historial['DNI'])
        df_historial['Fecha_de_gestion'] = pd.to_datetime(df_historial['Fecha_de_gestion'], errors='coerce')

    # 3. Construir Universo Total
    df_universo = pd.merge(df_opsitel, df_act, on=['DNI', 'Telefono'], how='outer')
    if df_historial is not None:
        df_tels_gestion = df_historial[['DNI', 'Telefono']].drop_duplicates()
        df_universo = pd.merge(df_universo, df_tels_gestion, on=['DNI', 'Telefono'], how='outer')

    # 4. FILTRADO: Blacklist + Teléfono igual a DNI
    df_universo = df_universo.drop_duplicates(subset=['DNI', 'Telefono'])
    df_universo = df_universo[df_universo['Telefono'].astype(str) != df_universo['DNI'].astype(str)].copy()
    df_universo = pd.merge(df_universo, df_black[['DNI', 'Telefono', 'Motivo']], on=['DNI', 'Telefono'], how='left')
    df_universo = df_universo[df_universo['Motivo'].isna()].copy()

    # 5. Calificar Score Base IA
    df_scored = calcular_score_base(df_universo, modelo_ia)

    # 6. Procesar Fallos y Éxitos
    if df_historial is not None:
        negativos = ['Nro. No pertenece', 'Telefono Apagado', 'Fds/Ne', 'FAILED', 'IVR Fallida']
        df_historial['es_fallo'] = df_historial['Resultado_Gestion'].isin(negativos).astype(int)
        
        stats_fallos = df_historial.groupby(['DNI', 'Telefono']).agg(
            total_fallos=('es_fallo', 'sum'),
            ultima_fecha_fallo=('Fecha_de_gestion', 'max')
        ).reset_index()
        
        df_scored = pd.merge(df_scored, stats_fallos, on=['DNI', 'Telefono'], how='left').fillna(0)

        # 7. RESCATE: Prioridad por Contacto Directo e Indirecto
        exitosos = df_historial[df_historial['resultado'].isin(['CONTACTO DIRECTO', 'CONTACTO INDIRECTO'])].copy()
        
        if not exitosos.empty:
            ultima_gestion_exito = exitosos.sort_values('Fecha_de_gestion', ascending=False).groupby(['DNI', 'Telefono']).agg(
                ultima_fecha_exito=('Fecha_de_gestion', 'max'),
                mejor_resultado=('resultado', 'first')
            ).reset_index()
            
            df_scored = pd.merge(df_scored, ultima_gestion_exito, on=['DNI', 'Telefono'], how='left')
            
            df_scored.loc[df_scored['mejor_resultado'] == 'CONTACTO DIRECTO', 'total_score'] += 100.0
            df_scored.loc[df_scored['mejor_resultado'] == 'CONTACTO INDIRECTO', 'total_score'] += 50.0

    # 8. Identificar y Exportar Números con 3 o más Fallos
    mask_descarte = (df_scored['total_fallos'] >= 3) & (df_scored.get('ultima_fecha_exito', pd.Series([None]*len(df_scored))).isnull())
    
    df_descartados = df_scored[mask_descarte].copy()
    df_descartados['MOTIVO'] = 'Exceso de gestiones negativas reincidentes'
    
    if not df_descartados.empty:
        # Formatear DNI para el Excel de descartados también
        df_descartados['DNI'] = df_descartados['DNI'].apply(lambda x: str(int(x)).zfill(8))
        df_descartados[['DNI', 'Telefono', 'MOTIVO']].to_excel('data/output/telefonos_descartados.xlsx', index=False)
        print(f">>> {len(df_descartados)} teléfonos movidos a lista de descartados.")

    # 9. Filtrar el universo para la salida final
    df_scored_final = df_scored[~mask_descarte].copy()

    # 10. Ordenamiento y Salida
    # CORRECCIÓN: Usamos df_scored_final para la lógica de columnas
    col_fecha = 'ultima_fecha_exito' if 'ultima_fecha_exito' in df_scored_final.columns else 'total_score'
    
    df_sorted = df_scored_final.sort_values(
        by=['DNI', 'total_score', col_fecha], 
        ascending=[True, False, False]
    )
    
    df_top3 = df_sorted.groupby('DNI').head(3).copy()

    print("Generando lista_final_horizontal.csv con formato de texto...")
    with open('data/output/lista_final_horizontal.csv', 'w') as f:
        # Encabezado con comillas y punto y coma
        f.write('"DNI";"Telefono_1";"Telefono_2";"Telefono_3"\n')
        
        for dni, grupo in df_top3.groupby('DNI'):
            dni_str = str(int(dni)).zfill(8)
            tels = [str(t).strip() for t in grupo['Telefono'].tolist() if pd.notnull(t) and str(t) != '']
            
            while len(tels) < 3:
                tels.append("")
            
            linea = f'"{dni_str}";"{tels[0]}";"{tels[1]}";"{tels[2]}"\n'
            f.write(linea)

    # Explicación con DNI formateado
    df_sorted['DNI'] = df_sorted['DNI'].apply(lambda x: str(int(x)).zfill(8))
    df_sorted.to_csv('data/output/explicacion_score.csv', index=False)
    print(">>> Proceso terminado con éxito.")

if __name__ == "__main__":
    ejecutar_ranking_ia()