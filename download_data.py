import pandas as pd
import os
from src.db_handler import get_db_connection

def descargar_gestiones():
    print(">>> Iniciando descarga de gestiones desde MySQL...")
    engine = get_db_connection()
    temp_csv = 'data/raw/temp_base_completa.csv'
    
    # Eliminar archivo previo para asegurar datos frescos
    if os.path.exists(temp_csv):
        os.remove(temp_csv)

    query = "SELECT DNI, Telefono, Resultado_Gestion, Fecha_de_gestion, resultado FROM vw_gestiones_unificadas"
    
    try:
        with engine.connect() as conn:
            primer_bloque = True
            # Descarga en bloques de 150k para evitar desconexiones (Error 10054/2006)
            for chunk in pd.read_sql(query, conn, chunksize=150000):
                modo = 'w' if primer_bloque else 'a'
                header = True if primer_bloque else False
                chunk.to_csv(temp_csv, mode=modo, index=False, header=header)
                primer_bloque = False
                print(f"Descargando... bloque guardado en local.")
        
        print(f">>> Descarga finalizada. Archivo disponible en: {temp_csv}")
    except Exception as e:
        print(f"❌ Error en la descarga: {e}")

if __name__ == "__main__":
    descargar_gestiones()