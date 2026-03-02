import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

def entrenar_modelo():
    print("Iniciando preparación de datos para ML...")
    
    # 1. Cargar datos
    df_gestiones = pd.read_csv('data/raw/gestiones_muestra.csv')
    df_opsitel = pd.read_excel('data/raw/base_opsitel.xlsx')
    df_act = pd.read_excel('data/raw/fecha_activacion.xlsx')

    # Limpieza de teléfonos
    for df in [df_gestiones, df_opsitel, df_act]:
        df['Telefono'] = df['Telefono'].astype(str).str.strip().str.replace('.0', '', regex=False)

    # 2. Construir el Dataset de Entrenamiento (Triple Unión)
    # Incluimos todos los teléfonos de gestiones, incluso los que no están en maestros
    df = df_gestiones.merge(df_opsitel[['Telefono', 'Activo']], on='Telefono', how='left')
    df = df.merge(df_act[['Telefono', 'FECHA_ACT']], on='Telefono', how='left')

    # --- INGENIERÍA DE VARIABLES ---
    # A. Opsitel
    df['feature_opsitel'] = df['Activo'].apply(lambda x: 1 if str(x).upper() == 'SI' else 0)

    # B. Antigüedad en meses
    df['FECHA_ACT'] = pd.to_datetime(df['FECHA_ACT'], errors='coerce')
    df['feature_antiguedad_meses'] = ((2026 - df['FECHA_ACT'].dt.year) * 12 + (2 - df['FECHA_ACT'].dt.month)).fillna(300)

    # C. NUEVA VARIABLE: Éxito Histórico (¿Tuvo contacto antes de esta gestión?)
    # Nota: En un entorno real, esto se calcula con datos previos a la fecha de la gestión actual
    # para evitar "Data Leakage". Por ahora, usaremos la columna resultado como target.
    df['target'] = (df['resultado'] == 'CONTACTO DIRECTO').astype(int)

    # 3. Entrenamiento
    # Entrenamos a la IA para que reconozca que Opsitel y Antigüedad son importantes,
    # pero el 'main.py' aplicará el bono de 100 puntos sobre este resultado.
    X = df[['feature_opsitel', 'feature_antiguedad_meses']]
    y = df['target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Entrenando RandomForest con pesos balanceados...")
    modelo = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    modelo.fit(X_train, y_train)

    # 4. Evaluación
    print("Evaluación del modelo:")
    y_pred = modelo.predict(X_test)
    print(classification_report(y_test, y_pred))

    # 5. Guardar modelo
    joblib.dump(modelo, 'data/output/modelo_contactabilidad.pkl')
    print("Modelo guardado exitosamente.")

if __name__ == "__main__":
    entrenar_modelo()