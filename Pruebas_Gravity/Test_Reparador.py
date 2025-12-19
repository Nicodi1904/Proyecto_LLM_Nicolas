import dspy
import os
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

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

print("Sistema de pruebas 'Reparador' inicializado")

# -------------------------------------------------------------------------
# Definición de Signatures
# -------------------------------------------------------------------------

class Reparador(dspy.Signature):
    """
    Signature dummy para el Reparador. 
    (Relleno temporal para evitar errores, pendiente de definición real por el usuario).
    """

    dato_entrada: str = dspy.InputField(
        desc="Dato de entrada de prueba para el reparador."
    )

    dato_salida: str = dspy.OutputField(
        desc="Resultado de la reparación o transformación."
    )

# -------------------------------------------------------------------------
# Datos de Prueba
# -------------------------------------------------------------------------

input_dummy = "Este es un dato de entrada corrupto o que necesita reparación."

print("\n###############################################")
print("Entrada:", input_dummy)

# -------------------------------------------------------------------------
# Ejecución de Modelos (Comentar/Descomentar según necesidad)
# -------------------------------------------------------------------------

"""
# --- Llama 3.1 8b ---
dspy.configure(lm=llama_31_8b)

reparador_llama31 = dspy.Predict(Reparador)
resultado_llama31 = reparador_llama31(
    dato_entrada = input_dummy
)

print("\\nReparador Llama 3.1 8b")
print("Salida:", resultado_llama31.dato_salida)
print("\\n###############################################")
"""

"""
# --- DeepSeek R1 8b ---
dspy.configure(lm=deepseek_r1_8b)

reparador_deepseek_r1_8b = dspy.Predict(Reparador)
resultado_deepseek_r1_8b = reparador_deepseek_r1_8b(
    dato_entrada = input_dummy
)

print("\\n###############################################\\nReparador DeepSeek R1 8b")
print("Salida:", resultado_deepseek_r1_8b.dato_salida)
print("\\n###############################################")
"""

"""
# --- Gemma 7b ---
dspy.configure(lm=gemma_7b)

reparador_gemma_7b = dspy.Predict(Reparador)
resultado_gemma_7b = reparador_gemma_7b(
    dato_entrada = input_dummy
)

print("\\nReparador Gemma 7b")
print("Salida:", resultado_gemma_7b.dato_salida)
print("\\n###############################################")
"""

"""
# --- Mistral 7b ---
dspy.configure(lm=mistral_7b)

reparador_mistral_7b = dspy.Predict(Reparador)
resultado_mistral_7b = reparador_mistral_7b(
    dato_entrada = input_dummy
)

print("\\nReparador Mistral 7b")
print("Salida:", resultado_mistral_7b.dato_salida)
print("\\n###############################################")
"""

"""
# --- Qwen 3 4b ---
dspy.configure(lm=qwen3_4b)

reparador_qwen3_4b = dspy.Predict(Reparador)
resultado_qwen3_4b = reparador_qwen3_4b(
    dato_entrada = input_dummy
)

print("\\nReparador Qwen 3 4b")
print("Salida:", resultado_qwen3_4b.dato_salida)
print("\\n###############################################")
"""

"""
# --- OpenRouter Llama 3.3 70b ---
dspy.configure(lm=openrouter_llama33_70b)

reparador_openrouter = dspy.Predict(Reparador)
resultado_openrouter = reparador_openrouter(
    dato_entrada = input_dummy
)

print("\\nReparador Llama 3.3 70b")
print("Salida:", resultado_openrouter.dato_salida)
print("\\n###############################################")
"""

"""
# --- OpenRouter Gemini 2.0 Flash ---
dspy.configure(lm=openrouter_gemini2flash)

reparador_gemini2flash = dspy.Predict(Reparador)
resultado_gemini2flash = reparador_gemini2flash(
    dato_entrada = input_dummy
)

print("\\nReparador Gemini 2.0 Flash")
print("Salida:", resultado_gemini2flash.dato_salida)
print("\\n###############################################")
"""
