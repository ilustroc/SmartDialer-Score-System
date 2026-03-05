import os
import urllib.parse
from dotenv import load_dotenv
from sqlalchemy import create_engine

# Carga las variables del archivo .env
load_dotenv()

def get_db_connection():
    user = os.getenv('DB_USER')
    # CAMBIO: Usar DB_PASS para que sea igual al .env
    password = os.getenv('DB_PASS') 
    host = os.getenv('DB_HOST')
    db = os.getenv('DB_NAME')
    
    # Verificación de que la contraseña no llegue vacía
    if not password:
        raise ValueError("❌ No se encontró la contraseña. Revisa que DB_PASS esté en tu .env")
    
    # Codificar caracteres especiales como '@' y '!'
    safe_password = urllib.parse.quote_plus(str(password))
    
    # Construcción de la URL
    conn_str = f'mysql+pymysql://{user}:{safe_password}@{host}/{db}'
    return create_engine(conn_str)