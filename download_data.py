import os
import pandas as pd
from sqlalchemy import text
from src.db_handler import get_db_connection


def descargar_gestiones():
    print(">>> Iniciando descarga de gestiones a ALTA VELOCIDAD...")

    temp_csv = "data/raw/temp_base_completa.csv"
    os.makedirs(os.path.dirname(temp_csv), exist_ok=True)

    if os.path.exists(temp_csv):
        os.remove(temp_csv)

    query = text("""
        SELECT DNI, Telefono, Resultado_Gestion, Fecha_de_gestion, resultado
        FROM vw_gestiones_unificadas
        WHERE Fecha_de_gestion >= DATE_SUB(NOW(), INTERVAL 6 MONTH)
    """)

    engine = None
    total_registros = 0
    primer_bloque = True

    try:
        engine = get_db_connection()

        with engine.connect().execution_options(stream_results=True) as conn:
            print(">>> Consultando servidor... por favor espera.")

            for chunk in pd.read_sql(query, conn, chunksize=200000):
                modo = "w" if primer_bloque else "a"
                header = primer_bloque

                chunk.to_csv(
                    temp_csv,
                    mode=modo,
                    index=False,
                    header=header,
                    encoding="utf-8"
                )

                total_registros += len(chunk)
                print(f">>> Recibidos {total_registros} registros...")
                primer_bloque = False

        if total_registros == 0:
            print("⚠️ La consulta no devolvió registros.")
        else:
            print(f">>> DESCARGA EXITOSA. Total: {total_registros} registros en {temp_csv}")

    except Exception as e:
        print(f"❌ Error crítico en la descarga: {e}")

    finally:
        if engine is not None:
            engine.dispose()


if __name__ == "__main__":
    descargar_gestiones()