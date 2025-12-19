import dspy
import os
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

# Importar summary del script hermano if needed
try:
    from MCP_C_obtener_summary import system_summary
except ImportError:
    # Fallback si se ejecuta desde otro directorio
    import sys
    sys.path.append(os.path.dirname(__file__))
    try:
        from MCP_C_obtener_summary import system_summary
    except ImportError:
        # Dummy summary for testing if file missing
        system_summary = {
            "herramientas": {
                "obtener_consumo": {
                    "descripcion": "Calcula el consumo energético.",
                    "argumentos": ["dispositivos", "fecha_inicio", "fecha_fin", "granularidad"]
                },
                "analizar_comparacion": {
                    "descripcion": "Compara consumos entre contextos.",
                    "argumentos": ["objetivo_a", "objetivo_b"]
                }
            }
        }

# Cargar API Keys
env_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(env_path)

APIKEY_GOOGLE = os.getenv("apikey_google_ai_studio")
APIKEY_OPENROUTER = os.getenv("apikey_openrouter")

# -------------------------------------------------------------------------
# Configuración de LLMs
# -------------------------------------------------------------------------

# LLMs Locales con Ollama
llama_31_8b = dspy.LM('ollama_chat/llama3.1:latest', api_base='http://localhost:11434', api_key='')
deepseek_r1_8b = dspy.LM('ollama_chat/deepseek-r1:8b', api_base='http://localhost:11434', api_key='')
gemma_7b = dspy.LM('ollama_chat/gemma:latest', api_base='http://localhost:11434', api_key='')
mistral_7b = dspy.LM('ollama_chat/mistral', api_base='http://localhost:11434', api_key='')
qwen3_4b = dspy.LM('ollama_chat/qwen3:4b', api_base='http://localhost:11434', api_key='')

# LLMs OpenRouter
openrouter_gemini2flash = dspy.LM(model="openrouter/google/gemini-2.0-flash-exp:free",
                            api_base="https://openrouter.ai/api/v1",
                            api_key=APIKEY_OPENROUTER)

openrouter_llama33_70b = dspy.LM(model="openrouter/meta-llama/llama-3.3-70b-instruct:free",
                            api_base="https://openrouter.ai/api/v1",
                            api_key=APIKEY_OPENROUTER)

print("Sistema de pruebas 'Evaluador' inicializado")

# -------------------------------------------------------------------------
# Definición de Signatures
# -------------------------------------------------------------------------

class Evaluador(dspy.Signature):
    """
    El Evaluador determina si cada solicitud identificada por el Interpretador es completamente realizable 
    utilizando exclusivamente las herramientas disponibles en el sistema.
    """

    peticiones_categorizadas: dict = dspy.InputField(
        desc=(
            "Diccionario de solicitudes generado por el interpretador. "
            "Cada clave sigue el formato '@N' (entero positivo consecutivo) "
            "y cada valor contiene la solicitud específica y su escenario asociado."
        )
    )

    system_summary: dict = dspy.InputField(
        desc=(
            "Resumen estructurado de las herramientas disponibles en el sistema, "
            "incluyendo sus capacidades, parámetros requeridos y restricciones de uso."
        )
    )

    factibilidad: dict = dspy.OutputField(
        desc=(
            "Diccionario JSON que evalúa la factibilidad de cada solicitud. "
            "Las claves deben coincidir exactamente con las claves '@N' de "
            "peticiones_categorizadas. "
            "Cada valor es un indicador binario: "
            "1 si la solicitud es completamente realizable con las herramientas actuales, "
            "0 si no lo es."
        )
    )

    evaluacion_detallada: dict = dspy.OutputField(
        desc=(
            "Diccionario JSON alineado uno a uno con 'factibilidad' y "
            "con las mismas claves '@N'."
            "Si la solicitud es realizable (factibilidad = 1), el valor debe ser "
            "una lista explícita de las herramientas que deben ejecutarse. "
            "Si no es realizable (factibilidad = 0), el valor debe ser una "
            "justificación clara y concreta de la causa de inadmisibilidad, "
            "sin proponer correcciones, alternativas ni suposiciones adicionales."
        )
    )

# -------------------------------------------------------------------------
# Datos de Prueba
# -------------------------------------------------------------------------

# Simulación de salida del Interpretador
peticiones_categorizadas = {
    "@1": {
        "solicitud": "Quiero saber el consumo de la nevera ayer por la noche",
        "escenario": "consumo_basico"
    },
    "@2": {
        "solicitud": "Comparar consumo de nevera vs lavadora",
        "escenario": "comparacion_consumos"
    },
    "@3": {
        "solicitud": "Predecir el precio del Bitcoin mañana",
        "escenario": "entrada_inadmisible"
    }
}

print("\n###############################################")
print("Peticiones de Entrada (Mock):", peticiones_categorizadas)

# -------------------------------------------------------------------------
# Ejecución de Modelos (Comentar/Descomentar según necesidad)
# -------------------------------------------------------------------------

"""
# --- Llama 3.1 8b ---
dspy.configure(lm=llama_31_8b)

evaluador_llama31 = dspy.Predict(Evaluador)
resultado_llama31 = evaluador_llama31(
    peticiones_categorizadas = peticiones_categorizadas,
    system_summary = system_summary
)

print("\\nEvaluador Llama 3.1 8b")
print("Factibilidad:", resultado_llama31.factibilidad)
print("Evaluación:", resultado_llama31.evaluacion_detallada)
print("\\n###############################################")
"""

"""
# --- DeepSeek R1 8b ---
dspy.configure(lm=deepseek_r1_8b)

evaluador_deepseek_r1_8b = dspy.Predict(Evaluador)
resultado_deepseek_r1_8b = evaluador_deepseek_r1_8b(
    peticiones_categorizadas = peticiones_categorizadas,
    system_summary = system_summary
)

print("\\n###############################################\\nEvaluador DeepSeek R1 8b")
print("Factibilidad:", resultado_deepseek_r1_8b.factibilidad)
print("Evaluación:", resultado_deepseek_r1_8b.evaluacion_detallada)
print("\\n###############################################")
"""

"""
# --- Gemma 7b ---
dspy.configure(lm=gemma_7b)

evaluador_gemma_7b = dspy.Predict(Evaluador)
resultado_gemma_7b = evaluador_gemma_7b(
    peticiones_categorizadas = peticiones_categorizadas,
    system_summary = system_summary
)

print("\\nEvaluador Gemma 7b")
print("Factibilidad:", resultado_gemma_7b.factibilidad)
print("Evaluación:", resultado_gemma_7b.evaluacion_detallada)
print("\\n###############################################")
"""

"""
# --- Mistral 7b ---
dspy.configure(lm=mistral_7b)

evaluador_mistral_7b = dspy.Predict(Evaluador)
resultado_mistral_7b = evaluador_mistral_7b(
    peticiones_categorizadas = peticiones_categorizadas,
    system_summary = system_summary
)

print("\\nEvaluador Mistral 7b")
print("Factibilidad:", resultado_mistral_7b.factibilidad)
print("Evaluación:", resultado_mistral_7b.evaluacion_detallada)
print("\\n###############################################")
"""

"""
# --- Qwen 3 4b ---
dspy.configure(lm=qwen3_4b)

evaluador_qwen3_4b = dspy.Predict(Evaluador)
resultado_qwen3_4b = evaluador_qwen3_4b(
    peticiones_categorizadas = peticiones_categorizadas,
    system_summary = system_summary
)

print("\\nEvaluador Qwen 3 4b")
print("Factibilidad:", resultado_qwen3_4b.factibilidad)
print("Evaluación:", resultado_qwen3_4b.evaluacion_detallada)
print("\\n###############################################")
"""

"""
# --- OpenRouter Llama 3.3 70b ---
dspy.configure(lm=openrouter_llama33_70b)

evaluador_openrouter = dspy.Predict(Evaluador)
resultado_openrouter = evaluador_openrouter(
    peticiones_categorizadas = peticiones_categorizadas,
    system_summary = system_summary
)

print("\\nEvaluador Llama 3.3 70b")
print("Factibilidad:", resultado_openrouter.factibilidad)
print("Evaluación:", resultado_openrouter.evaluacion_detallada)
print("\\n###############################################")
"""

"""
# --- OpenRouter Gemini 2.0 Flash ---
dspy.configure(lm=openrouter_gemini2flash)

evaluador_gemini2flash = dspy.Predict(Evaluador)
resultado_gemini2flash = evaluador_gemini2flash(
    peticiones_categorizadas = peticiones_categorizadas,
    system_summary = system_summary
)

print("\\nEvaluador Gemini 2.0 Flash")
print("Factibilidad:", resultado_gemini2flash.factibilidad)
print("Evaluación:", resultado_gemini2flash.evaluacion_detallada)
print("\\n###############################################")
"""
