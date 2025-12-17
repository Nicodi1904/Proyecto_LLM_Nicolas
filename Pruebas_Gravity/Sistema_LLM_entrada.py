import dspy
import os
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

# Importar summary del script hermano
try:
    from MCP_C_obtener_summary import system_summary
except ImportError:
    # Fallback si se ejecuta desde otro directorio
    import sys
    sys.path.append(os.path.dirname(__file__))
    from MCP_C_obtener_summary import system_summary

# Cargar API Keys
# Cargar API Keys
env_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(env_path)

APIKEY_GOOGLE = os.getenv("apikey_google_ai_studio")
APIKEY_OPENROUTER = os.getenv("apikey_openrouter")
# Add other keys if needed

# -------------------------------------------------------------------------
# Configuración de LLMs
# -------------------------------------------------------------------------

# LLMs Locales con Ollama (Sin API Key según instrucciones)

mistral = dspy.LM('ollama_chat/mistral', api_base='http://localhost:11434', api_key='')
qwen3_4b = dspy.LM('ollama_chat/qwen3:4b', api_base='http://localhost:11434', api_key='')
llama_31_8b = dspy.LM('ollama_chat/llama3.1:latest', api_base='http://localhost:11434', api_key='')
tinyllama = dspy.LM('ollama_chat/tinyllama:latest', api_base='http://localhost:11434', api_key='')
deepseek_r1_8b = dspy.LM('ollama_chat/deepseek-r1:8b', api_base='http://localhost:11434', api_key='')


#LLMs gratuitos de más parámetros con OpenRouter (https://openrouter.ai)
openrouter_gemini2flash = dspy.LM(model="openrouter/google/gemini-2.0-flash-exp:free",
                            api_base="https://openrouter.ai/api/v1",
                            api_key=APIKEY_OPENROUTER)

openrouter_llama33_70b = dspy.LM(model="openrouter/meta-llama/llama-3.3-70b-instruct:free",
                            api_base="https://openrouter.ai/api/v1",
                            api_key=APIKEY_OPENROUTER)


#LLMs gratuitos de más parámetros con LiteLLM (https://www.litellm.ai)




print("Sistema de entrada inicializado")

# -------------------------------------------------------------------------
# Definición de Signatures
# -------------------------------------------------------------------------

class Interpretador(dspy.Signature):
    "El interpretador se encarga de identificar y categorizar las solicitudes e indicaciones adicionales realizadas por el usuario."

    prompt_usuario: str = dspy.InputField(
        desc="prompt del usuario en lenguaje natural."
    )

    escenarios_entrada: dict = dspy.InputField(
        desc=(
            "Escenarios estructurados admitidos por el sistema."
        )
    )   

    peticiones_categorizadas: dict = dspy.OutputField(
    desc=(
        "Solicitudes segmentadas y categorizadas por el sistema. "
        "El resultado debe ser un único diccionario JSON, donde cada clave tiene el formato '@N'"
        "(N es un entero positivo consecutivo comenzando en 1), y cada valor es un diccionario que incluye únicamente las claves: "
        "1) 'solicitud': solicitud específica y detallada (string), "
        "2) 'escenario': escenario de entrada admitido por el sistema asociado a la solicitud (string)."
        )
    )

    notas: str = dspy.OutputField(
        desc="Notas e indicaciones adicionales identificadas en el prompt del usuario."
    )





################################################################
escenarios_entrada = {
    "consumo_basico": {
        "descripcion": (
            "Solicitudes que piden valores cuantitativos de consumo energético "
            "para uno o varios dispositivos, zonas del hogar o el hogar completo, "
            "dentro de un periodo de tiempo claramente definido. "
            "La intención es obtener información puntual o histórica "
            "sin realizar comparaciones, análisis de patrones ni proyecciones."
        ),
        "usar_si": [
            "El usuario solicita consumo energético de uno o varios dispositivos",
            "La consulta incluye un periodo de tiempo explícito",
            "La respuesta esperada es un valor numérico o agregado simple"
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
            "Solicitudes que no pueden ser procesadas por el sistema por estar "
            "fuera del dominio energético, ser ambiguas o no contener "
            "información suficiente para identificar una intención válida."
        ),
        "usar_si": [
            "La consulta no está relacionada con consumo energético",
            "La consulta es ambigua o carece de contexto",
            "No se puede mapear la intención a los otros escenarios"
        ]
    }
}


prompt_usuario=("Necesito saber cuánto consumió mi nevera ayer por la noche, "
    "y también cuánto consumió mi lavadora el sábado pasado en la mañana. "
    "Además quiero que me digas si entre esos dos días cuál gastó más energía. "
    "Ah, y por cierto, mientras miraba esos consumos se me descargó el celular "
    "y me dio mucha pereza pararme a buscar el cargador, pero igual quiero la comparación."
    "Papá también pidió que le dijeras cuánto fue el consumo de todos los dispositivos en el año 2024, quiero ver gráficas de todo lo que se pueda")

################################################################

#Inicialización de predictores (Ejemplo)

dspy.configure(lm=llama_31_8b)

interpretador_llama31 = dspy.Predict(Interpretador)
resultado_llama31 = interpretador_llama31(
    prompt_usuario = prompt_usuario,
    escenarios_entrada=escenarios_entrada,
)

print("\n\nInterpretador Llama 3.1 8b")
print(resultado_llama31.peticiones_categorizadas)
print("\nNotas Llama 3.1 8b")
print(resultado_llama31.notas)

###
dspy.configure(lm=openrouter_llama33_70b)

interpretador_openrouter = dspy.Predict(Interpretador)
resultado_openrouter = interpretador_openrouter(
    prompt_usuario = prompt_usuario,
    escenarios_entrada=escenarios_entrada,
)

print("\n\nInterpretador Llama 3.3 70b")
print(resultado_openrouter.peticiones_categorizadas)
print("\nNotas Llama 3.3 70b")
print(resultado_openrouter.notas) 

###
dspy.configure(lm=openrouter_gemini2flash)

interpretador_gemini2flash = dspy.Predict(Interpretador)
resultado_gemini2flash = interpretador_gemini2flash(
    prompt_usuario = prompt_usuario,
    escenarios_entrada=escenarios_entrada,
)

print("\n\nInterpretador Gemini 2.0 Flash")
print(resultado_gemini2flash.peticiones_categorizadas)
print("\nNotas Gemini 2.0 Flash")
print(resultado_gemini2flash.notas)

