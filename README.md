SmartDialer Score System (ML Edition)
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

Penalización temporal: Un teléfono solo se descarta si registra 3 o más fallos críticos después del último contacto válido o promesa. Si antes tuvo FAILED, Apagado u otro negativo, pero luego tuvo CONTACTO DIRECTO, CONTACTO INDIRECTO o Compromiso de pago, el conteo de fallos se reinicia desde esa fecha y el teléfono se conserva.

🔢 Orden de los Teléfonos
El archivo lista_final_horizontal.csv se organiza por DNI y muestra hasta 3 teléfonos priorizados por cliente:

Telefono_1: Es el mejor teléfono disponible para llamar.

Telefono_2: Es el segundo mejor teléfono.

Telefono_3: Es el tercer mejor teléfono.

Antes de ordenar, el sistema elimina teléfonos duplicados, teléfonos iguales al DNI, números inválidos según reglas de telefonía peruana, teléfonos incluidos en la blacklist y teléfonos con 3 o más fallos críticos posteriores al último contacto/promesa.

El orden se calcula así:

1. Primero se ordena por DNI.
2. Dentro de cada DNI, los teléfonos se ordenan por total_score de mayor a menor.
3. El total_score viene del modelo de IA y se refuerza con el historial: Compromiso de pago tiene mayor prioridad, luego CONTACTO DIRECTO y luego CONTACTO INDIRECTO.
4. Si hay teléfonos con contacto exitoso o promesa, se prioriza el evento positivo más reciente usando fecha_ultima_proteccion.
5. Finalmente se toman los primeros 3 teléfonos de cada DNI y se colocan en Telefono_1, Telefono_2 y Telefono_3.

Si un DNI tiene menos de 3 teléfonos válidos, las columnas restantes salen vacías.

Guía de Uso

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

