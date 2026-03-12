import dspy
import os
from dotenv import load_dotenv

# Cargar API Keys
env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
load_dotenv(env_path)

# -------------------------------------------------------------------------
# Definición de Signatures
# -------------------------------------------------------------------------

class Planeador(dspy.Signature):
    """
    Traduce las solicitudes categorizadas por el inferenciador
    en un conjunto estructurado de acciones sobre las herramientas disponibles del sistema.
    Su función es decidir qué herramientas utilizar, en qué orden y con qué parámetros,
    evaluando la viabilidad funcional de cada solicitud.
    """

    solicitudes_categorizadas: dict[str, dict] = dspy.InputField(
        desc=(
            "Conjunto de solicitudes previamente segmentadas y categorizadas por el inferenciador. "
            "Cada clave representa una solicitud individual identificada con un identificador '@N'. "
            "Cada valor es un diccionario con las siguientes claves:\n"
            "- 'solicitud' (string): formulación autocontenida de la intención del usuario, utilizada "
            "como base principal para la planificación de acciones.\n"
            "- 'escenario' (string): clasificación funcional inferida (por ejemplo, consumo_basico, "
            "comparacion_consumos, deteccion_anomalias). Este campo sirve como guía semántica para "
            "priorizar y seleccionar herramientas.\n"
            "- 'formato' (string): preferencia de presentación de la respuesta esperada por el usuario, "
            "puede ser ('texto', 'grafico', 'mixto' o 'no_especificado') y debe ser considerada como parte de los requisitos de la solicitud."
        )
    )

    system_summary: dict = dspy.InputField(
        desc=(
            "Resumen estructurado de las herramientas disponibles en el sistema, organizado por servidores. "
            "Incluye, para cada herramienta, su propósito, criterios de uso, esquema de entradas (input_schema) "
            "y esquema de salidas (output_schema). Esta información debe ser utilizada por el planeador para:\n"
            "- determinar qué herramientas son funcionalmente adecuadas para cada solicitud,\n"
            "- construir cadenas de acciones donde la salida de una herramienta pueda servir como entrada de otra,\n"
            "- y evaluar la viabilidad de un plan antes de su ejecución."
        )
    )
    temporal_context: dict = dspy.InputField(
    desc=(
        "Contexto temporal del sistema, utilizado como referencia para interpretar "
        "expresiones temporales relativas presentes en las solicitudes del usuario. "
        "Los rangos son disjuntos, no se solapan y no deben combinarse ni extenderse "
        "fuera de sus límites definidos."
        )
    )



    plan_acciones: list[dict] = dspy.OutputField(
        desc=(
            "Lista estructurada de acciones planificadas para resolver las solicitudes del usuario. "
            "Cada elemento de la lista representa una acción individual e incluye:\n"
            "- 'id' (string): identificador único de la acción, con el formato '@N.M', donde '@N' "
            "corresponde a la solicitud de origen y 'M' indica el orden secuencial de la acción dentro "
            "de dicha solicitud.\n"
            "- 'server_id' (string): identificador del servidor donde se encuentra la herramienta a invocar.\n"
            "- 'tool' (string): nombre de la herramienta seleccionada.\n"
            "- 'inputs' (dict): parámetros de entrada de la herramienta, que pueden incluir referencias "
            "a salidas de acciones previas mediante identificadores '@N.M'.\n"
            "- 'descripcion' (string): breve descripción semántica de la acción y su propósito dentro del plan."
        )
    )

    estado_solicitudes: dict[str, dict] = dspy.OutputField(
    desc=(
        "Estado final de cada solicitud procesada. Cada clave '@N' corresponde "
        "a cada solicitud de entrada y su valor es un diccionario con las claves:\n"
        "- 'estado':\n"
        "* 'resuelta': cuando la solicitud, criterios y preferencias se pueden satisfacer completamente.\n"
        "* 'parcial': cuando solo una parte de la solicitud puede resolverse, "
        "o alguno de los requisitos o preferencias del usuario no pueden cumplirse.\n"
        "* 'no_resuelta': cuando la solicitud no puede resolverse con las herramientas disponibles.\n"
        "- 'motivo' (obligatorio si estado != 'resuelta'): diccionario con las claves:\n"
        "- 'tipo': categoría del problema detectado. Valores permitidos:\n"
        "* 'falta_informacion_usuario': la solicitud no contiene información suficiente "
        "para construir un plan ejecutable completo, aun cuando el sistema tendría "
        "capacidad para resolverla si dicha información estuviera disponible.\n"
        "* 'parametros_incompatibles': la solicitud especifica parámetros que no pueden coexistir de forma válida " 
        "según los esquemas y restricciones de las herramientas disponibles en el sistema.\n"
        "* 'limitacion_sistema': la solicitud requiere una capacidad que el sistema no posee "
        "según la definición actual de herramientas disponibles.\n"
        "- 'detalle': descripción breve y concreta del problema, indicando qué parte "
        "de la solicitud no pudo resolverse."
    )
)
