import dspy
import os
from dotenv import load_dotenv

# Cargar API Keys (reutilizamos logic de entrada o cargamos directo)
# Cargar API Keys (reutilizamos logic de entrada o cargamos directo)
env_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(env_path)
APIKEY_GOOGLE = os.getenv("apikey_google_ai_studio")

# -------------------------------------------------------------------------
# Configuración de LLMs (Idéntica a Sistema_LLM_entrada)
# -------------------------------------------------------------------------
llama_31 = dspy.LM('ollama_chat/llama3.1:latest', api_base='http://localhost:11434', api_key='')
mistral = dspy.LM('ollama_chat/mistral', api_base='http://localhost:11434', api_key='')
# Google config (Placeholder/Conditional)
if APIKEY_GOOGLE:
     try:
        google_model = dspy.Google("models/gemini-1.5-flash", api_key=APIKEY_GOOGLE)
     except:
        google_model = dspy.LM('google/gemini-1.5-flash', api_key=APIKEY_GOOGLE)
else:
    google_model = None

# -------------------------------------------------------------------------
# Signature Lector
# -------------------------------------------------------------------------

class Lector(dspy.Signature):
    """
    El Analista (Lector) recibe el plan ejecutado y sus resultados técnicos para generar una respuesta en lenguaje natural para el usuario.
    Debe sintetizar la información cuantitativa y cualitativa.
    """
    mensaje_usuario = dspy.InputField(desc="El mensaje original del usuario preguntando algo.")
    plan_estructurado = dspy.InputField(desc="La lista de pasos o herramientas que se planificaron.")
    respuestas_mcp = dspy.InputField(desc="Diccionario con los resultados obtenidos de la ejecución de las herramientas (Salida de MCP_C).")
    
    lectura_usuario = dspy.OutputField(desc="Respuesta final explicativa para el usuario, integrando los datos obtenidos.")

# Inicialización de predictores (Puede ser llamado desde un main o importado)
# dspy.configure(lm=llama_31)
# lector_agent = dspy.Predict(Lector)
