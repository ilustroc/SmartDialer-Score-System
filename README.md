SmartDialer Score System (ML Edition) 🚀
Este sistema procesa universos de datos de cobranzas (hasta 3.5M de registros) para priorizar los mejores teléfonos de contacto por cliente utilizando un modelo de Machine Learning (Random Forest).

🛠️ Estructura del Proyecto
main.py: Motor principal que coordina la carga, calificación, penalización y exportación.

src/logic.py: Contiene la lógica de limpieza de datos, ingeniería de variables y scoring base.

src/train_model.py: Script para entrenar el modelo .pkl usando muestras históricas.

download_data.py: Utilidad para descargar gestiones desde MySQL a un temporal local.

📋 Requisitos de Datos (data/raw/)
Para que el sistema funcione, debes tener los siguientes archivos en la carpeta raw:

Archivo                     Formato     Descripción
base_opsitel.xlsx           Excel       Estado de la línea (SI/NO).
fecha_activacion.xlsx       Excel       Fecha de activación (YYYY-MM).
blacklist_telefonos.xlsx    Excel       Teléfonos a omitir por DNI y Motivo.
gestiones_muestra.csv       CSV         Muestra para entrenamiento del modelo.

⚙️ Funcionamiento de la IA y Reglas
El sistema aplica un enfoque híbrido para garantizar que el Top 1 sea siempre el mejor número posible:

Ingeniería de Variables: Convierte la fecha de activación a meses de antigüedad y el estado Opsitel a valores binarios.

Blacklist: Los números en esta lista son eliminados del universo antes de calificar.

Modelo ML: Calcula la probabilidad de éxito basándose en patrones de "Contacto Directo".

Penalización Dura: Si un teléfono registra 3 o más fallos críticos (Nro. No pertenece, Apagado, etc.) en el historial, su score se reduce automáticamente a 0.0.

🚀 Guía de Uso

Paso 1: Actualizar Datos
Si la base en MySQL cambió, actualiza el archivo temporal local:

python download_data.py

Paso 2: Entrenar (Opcional - Una vez al mes)
Si tienes nuevas muestras de gestión, re-entrena el cerebro de la IA:

python src/train_model.py

Paso 3: Ejecutar Scoring
Procesa el universo completo y genera los resultados:

python main.py

📤 Salidas del Sistema (data/output/)
lista_final_horizontal.csv: Formato optimizado para marcadores (DNI, Telefono_1, Telefono_2, Telefono_3). Sin comas vacías al final.

explicacion_score.csv: Detalle completo de todos los teléfonos calificados por cliente para auditoría.

