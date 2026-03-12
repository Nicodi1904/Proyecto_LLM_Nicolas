import dspy
import os
from dotenv import load_dotenv

# Cargar variables de entorno desde archivo .env
env_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '.env'))
load_dotenv(dotenv_path=env_path)

# -------------------------------------------------------------------------
# Definición de Signature: Presentador
# -------------------------------------------------------------------------

class Presentador(dspy.Signature):
    """
    Módulo de razonamiento que genera una respuesta final detallada para el usuario.
    Analiza los resultados obtenidos de las herramientas y proporciona interpretación,
    contexto sobre las tareas realizadas y recomendaciones energéticas.
    """

    solicitudes_categorizadas: dict[str, dict] = dspy.InputField(
        desc=(
            "Solicitudes originales del usuario segmentadas y categorizadas por el interpretador. "
            "Contiene la intención, el escenario y las preferencias de formato."
        )
    )

    reporte_acciones: dict[str, list[dict]] = dspy.InputField(
        desc=(
            "Reporte consolidado de todas las acciones planificadas y ejecutadas. "
            "Incluye resultados exitosos, datos obtenidos del sistema MCP y causas de fallo si las hubo."
        )
    )

    indicaciones_manejo_informacion: str = dspy.InputField(
        desc="Instrucciones específicas sobre el tono, formato de respuesta deseado y manejo de datos sensibles."
    )

    respuesta_detallada: str = dspy.OutputField(
        desc=(
            "Texto final para el usuario. Debe incluir:\n"
            "1. Resumen de tareas realizadas y sus resultados.\n"
            "2. Interpretación semántica de los datos obtenidos.\n"
            "3. Recomendaciones sobre la gestión del consumo energético si son aplicables.\n"
            "4. Explicación amable de cualquier limitación o fallo encontrado."
        )
    )
