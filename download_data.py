import pandas as pd
import os
from src.db_handler import get_db_connection
from sqlalchemy import text

def descargar_gestiones():
    print(">>> Iniciando descarga de gestiones a ALTA VELOCIDAD...")
    temp_csv = 'data/raw/temp_base_completa.csv'
    
    if os.path.exists(temp_csv):
        os.remove(temp_csv)

    # 1. Filtro Incremental (Solo últimos 6 meses para que sea rápido)
    # Si realmente necesitas TODO, quita el WHERE, pero tardará más.
    query = text("""
        SELECT DNI, Telefono, Resultado_Gestion, Fecha_de_gestion, resultado 
        FROM vw_gestiones_unificadas 
        WHERE Fecha_de_gestion >= DATE_SUB(NOW(), INTERVAL 6 MONTH)
    """)
    
    try:
        engine = get_db_connection()
        
        # 2. Usamos 'execution_options' con stream_results para no saturar la RAM
        # y evitar el lento OFFSET.
        with engine.connect().execution_options(stream_results=True) as conn:
            print(">>> Consultando servidor... por favor espera.")
            
            primer_bloque = True
            count = 0
            
            # 3. Leemos directamente por bloques del flujo de red
            for chunk in pd.read_sql(query, conn, chunksize=200000):
                modo = 'w' if primer_bloque else 'a'
                header = True if primer_bloque else False
                
                chunk.to_csv(temp_csv, mode=modo, index=False, header=header, encoding='utf-8')
                
                count += len(chunk)
                print(f">>> Recibidos {count} registros...")
                primer_bloque = False

        print(f">>> DESCARGA EXITOSA. Total: {count} registros en {temp_csv}")
        engine.dispose()

    except Exception as e:
        print(f"❌ Error crítico en la descarga: {e}")

if __name__ == "__main__":
    descargar_gestiones()