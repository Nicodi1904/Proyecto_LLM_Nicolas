
import dspy
import os
from dotenv import load_dotenv

# Cargar variables de entorno desde archivo .env
env_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '.env'))
load_dotenv(dotenv_path=env_path)

# -------------------------------------------------------------------------
# 3. Definición de Signature
# -------------------------------------------------------------------------

class Gerente(dspy.Signature):
    """
    El Gerente es responsable de generar la respuesta final al usuario a partir de:
    - las solicitudes originales,
    - y un reporte unificado de acciones planificadas y ejecutadas.

    Su función es interpretar los resultados disponibles y comunicarlos de forma clara,
    coherente y alineada con las preferencias del usuario.
    """

    solicitudes_categorizadas: dict[str, dict] = dspy.InputField(
        desc=(
            "Solicitudes del usuario previamente segmentadas y categorizadas. "
            "Cada clave '@N' identifica una solicitud individual e incluye su formulación original, "
            "escenario funcional y preferencias de formato."
        )
    )

    reporte_acciones: dict[str, list[dict]] = dspy.InputField(
        desc=(
            "Reporte unificado de acciones asociadas a cada solicitud '@N'. "
            "Incluye tanto acciones ejecutadas como acciones no ejecutadas. "
            "Cada acción contiene:\n"
            "- su identificador,\n"
            "- la herramienta asociada,\n"
            "- una descripción semántica,\n"
            "- el resultado producido cuando existe,\n"
            "- y la causa documentada cuando la acción no pudo ejecutarse."
        )
    )

    respuesta_usuario: str = dspy.OutputField(
        desc=(
            "Respuesta final presentada al usuario, organizada por solicitud '@N'. "
            "Debe sintetizar la información contenida en el reporte de acciones, "
            "explicando de forma clara los resultados obtenidos y el alcance efectivo "
            "de cada solicitud."
        )
    )
