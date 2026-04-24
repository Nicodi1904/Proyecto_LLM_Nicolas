import dspy

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
        
    elif "deepseek" in model_name:
        # Ejemplo: 'deepseek-r1:latest' o 'deepseek-r1:8b'
        # Aseguramos que el nombre del modelo sea exactamente lo que Ollama espera
        return dspy.LM(f'ollama_chat/{model_name}', api_base=base_url, api_key='')
        
    else:
        # Fallback a deepseek-r1:8b dado que es el único garantizado en el entorno
        print(f"Advertencia: Modelo {model_name} no reconocido. Usando deepseek-r1:8b por defecto.")
        return dspy.LM('ollama_chat/deepseek-r1:8b', api_base=base_url, api_key='')

def listar_modelos_configurados():
    """
    Devuelve la lista de modelos que queremos mostrar en el dropdown.
    """
    return [
        {"label": "DeepSeek R1 8B (vía Ollama)", "value": "deepseek-r1:8b"},
        {"label": "Llama 3.1 8B (vía Ollama)", "value": "llama3.1:latest"},
        {"label": "Gemma 2 9B (vía Ollama)", "value": "gemma2:9b"}
    ]
