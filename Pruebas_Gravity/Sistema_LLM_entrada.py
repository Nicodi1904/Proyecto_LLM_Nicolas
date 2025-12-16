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
llama_31 = dspy.LM('ollama_chat/llama3.1:latest', api_base='http://localhost:11434', api_key='')
mistral = dspy.LM('ollama_chat/mistral', api_base='http://localhost:11434', api_key='')



#Adaptar modelos de google para poder inicializarse con DSPY
openrouter_model = dspy.LM(model="openrouter/google/gemini-2.0-flash-exp:free",api_base="https://openrouter.ai/api/v1",api_key=APIKEY_OPENROUTER)





# -------------------------------------------------------------------------
# Definición de Signatures
# -------------------------------------------------------------------------

class Toolformer(dspy.Signature):
    pregunta: str = dspy.InputField(
        desc="Mensaje original del usuario."
    )
    system_summary:Dict[str, Any] = dspy.InputField(
        desc=(
        "Resumen de herramientas disponibles y sus parámetros. "
        "Incluye cómo deben llamarse y en qué casos se usan."
        )
    )
    tiempo_actual:str=dspy.InputField(
        desc=("Tiempo actual en formato 'YYYY-MM-DD HH:MM'")
    )
    Toolformer:str=dspy.OutputField(
        desc=("Mensaje del usuario original con llamadas a herramientas incluidas, las llamadas a herramientas se deben incluir delimitando con asteriscos: Texto*@nombre_herramienta,@...*Texto ")
    )
    razonamiento:str=dspy.OutputField(
        desc=("Explicación de cada llamado a herramienta")
    )

print("Sistema de entrada inicializado. Modelos disponibles: Llama 3.1, Mistral, OpenRouter")


################################################################
pregunta=("Necesito saber cuánto consumió mi nevera ayer por la noche, "
    "y también cuánto consumió mi lavadora el sábado pasado en la mañana. "
    "Además quiero que me digas si entre esos dos días cuál gastó más energía. "
    "Ah, y por cierto, mientras miraba esos consumos se me descargó el celular "
    "y me dio mucha pereza pararme a buscar el cargador, pero igual quiero la comparación."
    "Papá también pidió que le dijeras cuánto fue el consumo de todos los dispositivos en el año 2024, quiero ver gráficas de todo lo que se pueda")

################################################################

#Inicialización de predictores (Ejemplo)

dspy.configure(lm=llama_31)
Toolformer_llama31 = dspy.Predict(Toolformer)
resultado_llama31 = Toolformer_llama31(
    pregunta = pregunta,
    tiempo_actual="2025-5-15 16:30",
    system_summary=system_summary,
)

print("\n\nToolformer Llama 3.1")
print(resultado_llama31.Toolformer)
print("\nRazonamiento Llama 3.1")
print(resultado_llama31.razonamiento)

dspy.configure(lm=openrouter_model)
Toolformer_openrouter = dspy.Predict(Toolformer)
resultado_openrouter = Toolformer_openrouter(
    pregunta = pregunta,
    tiempo_actual="2025-5-15 16:30",
    system_summary=system_summary,
)

print("\n\nToolformer OpenRouter")
print(resultado_openrouter.Toolformer)
print("\nRazonamiento OpenRouter")
print(resultado_openrouter.razonamiento)