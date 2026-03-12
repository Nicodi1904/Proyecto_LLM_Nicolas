import dspy
import os
from dotenv import load_dotenv
# -------------------------------------------------------------------------
# Entradas del módulo interpretador
# -------------------------------------------------------------------------
mensaje_usuario:str = ""
escenarios_disponibles:dict = {
    "consumo_basico": {
        "descripcion": (
            "Solicitudes cuya intención principal es consultar valores de consumo energético "
            "ya registrados para uno o varios dispositivos, zonas del hogar o el hogar completo, "
            "dentro de un intervalo temporal explícito o claramente inferible. "
            "La intención es acceder a datos puntuales o históricos sin realizar "
            "comparaciones analíticas, análisis de patrones, inferencias ni estimaciones."
        ),
        "usar_si": [
            "El usuario solicita valores de consumo energético históricos",
            "La consulta incluye un periodo de tiempo explícito o claramente inferible",
            "La respuesta esperada es un valor cuantitativo único o una serie temporal"
        ]
    },

    "comparacion_consumos": {
        "descripcion": (
            "Solicitudes cuya intención principal es comparar consumos energéticos "
            "entre dispositivos, zonas del hogar o periodos de tiempo distintos, "
            "con el fin de identificar diferencias relativas o determinar cuál "
            "consume más o menos energía."
        ),
        "usar_si": [
            "El usuario pide comparar dos o más consumos",
            "La consulta utiliza términos comparativos explícitos",
            "Se requiere analizar diferencias absolutas o relativas"
        ]
    },

    "deteccion_anomalias_tendencias": {
        "descripcion": (
            "Solicitudes orientadas a identificar comportamientos anómalos, "
            "picos de consumo o tendencias sostenidas en el consumo energético, "
            "a partir del análisis de datos históricos y patrones temporales."
        ),
        "usar_si": [
            "El usuario menciona picos, anomalías o consumos irregulares",
            "La consulta sugiere aumentos o disminuciones sostenidas",
            "Se busca detectar fallos o hábitos de uso ineficientes"
        ]
    },

    "prediccion_estimacion": {
        "descripcion": (
            "Solicitudes que buscan estimar o predecir el consumo energético futuro "
            "o evaluar escenarios hipotéticos basados en cambios de hábitos, "
            "dispositivos o condiciones especiales."
        ),
        "usar_si": [
            "El usuario solicita estimaciones futuras",
            "La consulta plantea escenarios hipotéticos",
            "La respuesta requiere inferencia o proyección"
        ]
    },

    "entrada_inadmisible": {
        "descripcion": (
            "Solicitudes que expresan una intención explícita pero que se encuentran "
            "fuera del dominio del consumo energético del hogar, o que no corresponden "
            "a ninguno de los escenarios de entrada admitidos por el sistema. "
            "Estas solicitudes son identificables semánticamente, pero no representan "
            "acciones que el sistema esté diseñado para interpretar o procesar."
        ),
        "usar_si": [
            "La consulta no está relacionada con consumo energético",
            "La consulta es ambigua o carece de contexto",
            "No se puede mapear la intención a los otros escenarios"
        ]
    }

}
# -------------------------------------------------------------------------
# Definición de la signature
# -------------------------------------------------------------------------
class Interpretador(dspy.Signature):
    "Identifica las solicitudes realizadas por el usuario y las categoriza."

    prompt_usuario: str = dspy.InputField(
        desc="prompt del usuario en lenguaje natural."
    )


    solicitudes_categorizadas: dict[str, dict] = dspy.OutputField(
    desc=(
        "Solicitudes segmentadas y categorizadas por el sistema. "
        "El resultado debe ser un único diccionario JSON, donde cada clave "
        "tiene el formato '@N' (N es un entero positivo consecutivo comenzando en 1), "
        "y cada valor es un diccionario con las siguientes claves:\n"
        "'solicitud' (string): solicitud específica y detallada, completamente autocontenida, no debe depender de otras solicitudes.\n"
        "'escenario' (string): escenario de entrada admitido por el sistema."
        )
    )

    notas: str = dspy.OutputField(
    desc=(
        "razonamiento que llevó a elegir el escenario para cada solicitud"
        )
    )


# Cargar API Keys
# Se ajusta la ruta del .env para que apunte al directorio raíz del paquete (../.env)
env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
load_dotenv(env_path)


# -------------------------------------------------------------------------
# Datos de Entrada
# -------------------------------------------------------------------------

escenarios_entrada = {
    "consumo_basico": {
        "descripcion": (
            "Solicitudes orientadas a consultar o recuperar valores cuantitativos "
            "de consumo energético ya registrados para uno o varios dispositivos, "
            "zonas del hogar o el hogar completo, dentro de un periodo de tiempo "
            "explícitamente definido. "
            "La intención es acceder a datos puntuales o históricos sin realizar "
            "comparaciones, análisis de patrones, inferencias, proyecciones "
            "ni definir formatos de visualización o presentación de resultados."
        ),
        "usar_si": [
            "El usuario solicita valores de consumo energético existentes",
            "La consulta incluye un periodo de tiempo explícito o claramente inferible",
            "La respuesta esperada es un valor cuantitativo único o agregado simple"
        ]
    },

    "comparacion_consumos": {
        "descripcion": (
            "Solicitudes cuya intención principal es comparar consumos energéticos "
            "entre dispositivos, zonas del hogar o periodos de tiempo distintos, "
            "con el fin de identificar diferencias relativas o determinar cuál "
            "consume más o menos energía."
        ),
        "usar_si": [
            "El usuario pide comparar dos o más consumos",
            "La consulta utiliza términos comparativos explícitos",
            "Se requiere analizar diferencias absolutas o relativas"
        ]
    },

    "deteccion_anomalias_tendencias": {
        "descripcion": (
            "Solicitudes orientadas a identificar comportamientos anómalos, "
            "picos de consumo o tendencias sostenidas en el consumo energético, "
            "a partir del análisis de datos históricos y patrones temporales."
        ),
        "usar_si": [
            "El usuario menciona picos, anomalías o consumos irregulares",
            "La consulta sugiere aumentos o disminuciones sostenidas",
            "Se busca detectar fallos o hábitos de uso ineficientes"
        ]
    },

    "prediccion_estimacion": {
        "descripcion": (
            "Solicitudes que buscan estimar o predecir el consumo energético futuro "
            "o evaluar escenarios hipotéticos basados en cambios de hábitos, "
            "dispositivos o condiciones especiales."
        ),
        "usar_si": [
            "El usuario solicita estimaciones futuras",
            "La consulta plantea escenarios hipotéticos",
            "La respuesta requiere inferencia o proyección"
        ]
    },

    "entrada_inadmisible": {
        "descripcion": (
            "Solicitudes que expresan una intención explícita pero que se encuentran "
            "fuera del dominio del consumo energético del hogar, o que no corresponden "
            "a ninguno de los escenarios de entrada admitidos por el sistema. "
            "Estas solicitudes son identificables semánticamente, pero no representan "
            "acciones que el sistema esté diseñado para interpretar o procesar."
        ),
        "usar_si": [
            "La consulta no está relacionada con consumo energético",
            "La consulta es ambigua o carece de contexto",
            "No se puede mapear la intención a los otros escenarios"
        ]
    }
}