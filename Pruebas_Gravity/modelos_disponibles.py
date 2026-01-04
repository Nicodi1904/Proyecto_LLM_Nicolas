import dspy
import os
from dotenv import load_dotenv

# Cargar API Keys si es necesario
env_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(env_path)

def get_model(model_name: str):
    """
    Inicializa y devuelve una instancia de modelo dspy.LM basada en el nombre.
    """
    # Configuración por defecto (Ollama local)
    base_url = "http://localhost:11434"
    
    if "llama3.1" in model_name:
        # Ejemplo: 'llama3.1:latest' o 'llama3.1:8b'
        return dspy.LM(f'ollama_chat/{model_name}', api_base=base_url, api_key='')
    
    elif "gemma2" in model_name:
        return dspy.LM(f'ollama_chat/{model_name}', api_base=base_url, api_key='')
    
    elif "phi3" in model_name:
        return dspy.LM(f'ollama_chat/{model_name}', api_base=base_url, api_key='')
    
    elif "gemini-2.5-flash" in model_name:
        api_key = os.getenv("APIKEY_GEMINI_JUANC")
        return dspy.LM(model="openai/gemini-2.5-flash", 
                        api_base="https://generativelanguage.googleapis.com/v1beta/openai/",
                        api_key=api_key)
        
    else:
        # Fallback a llama3.1 si el modelo no está mapeado explícitamente
        return dspy.LM('ollama_chat/llama3.1:latest', api_base=base_url, api_key='')

def listar_modelos_configurados():
    """
    Devuelve la lista de modelos que queremos mostrar en el dropdown.
    """
    return [
        {"label": "Llama 3.1 8B (vía Ollama)", "value": "llama3.1:latest"},
        {"label": "Gemma 2 9B (vía Ollama)", "value": "gemma2:9b"},
        {"label": "Phi 3 (vía Ollama)", "value": "phi3:latest"},
        {"label": "Gemini 2.5 Flash (API)", "value": "gemini-2.5-flash"}
    ]
