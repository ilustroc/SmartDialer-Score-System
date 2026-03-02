import os
import urllib.parse
from dotenv import load_dotenv
from sqlalchemy import create_engine

# Carga las variables del archivo .env
load_dotenv()

def get_db_connection():
    user = os.getenv('DB_USER')
    password = os.getenv('DB_PASSWORD')
    host = os.getenv('DB_HOST')
    db = os.getenv('DB_NAME')
    
    # IMPORTANTE: Encode de la contraseña para manejar el '@' y el '!'
    safe_password = urllib.parse.quote_plus(password)
    
    # Construcción de la URL con la contraseña codificada
    conn_str = f'mysql+pymysql://{user}:{safe_password}@{host}/{db}'
    return create_engine(conn_str)