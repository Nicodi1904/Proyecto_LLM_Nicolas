# %%
import dspy
import os
from dotenv import load_dotenv

# Importar summary del script hermano if needed (retained from original structure)
try:
    from MCP_C_obtener_summary import system_summary
except ImportError:
    # Fallback si se ejecuta desde otro directorio
    import sys
    sys.path.append(os.path.dirname(__file__))
    try:
        from MCP_C_obtener_summary import system_summary
    except ImportError:
        pass # Not strictly needed for Interpretador but kept for consistency

# Cargar API Keys
env_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(env_path)

APIKEY_GOOGLE = os.getenv("apikey_google_ai_studio")
APIKEY_OPENROUTER = os.getenv("apikey_openrouter")
print("Librerías y AppiKeys cargadas correctamente")
# %%
# -------------------------------------------------------------------------
# Configuración de LLMs
# -------------------------------------------------------------------------

# LLMs Locales con Ollama
llama_31_8b = dspy.LM('ollama_chat/llama3.1:latest', api_base='http://localhost:11434', api_key='')
deepseek_r1_8b = dspy.LM('ollama_chat/deepseek-r1:8b', api_base='http://localhost:11434', api_key='')
gemma_7b = dspy.LM('ollama_chat/gemma:latest', api_base='http://localhost:11434', api_key='')
mistral_7b = dspy.LM('ollama_chat/mistral:latest', api_base='http://localhost:11434', api_key='')
qwen3_4b = dspy.LM('ollama_chat/qwen3:4b', api_base='http://localhost:11434', api_key='')

# LLMs OpenRouter
openrouter_gemini2flash = dspy.LM(model="openrouter/google/gemini-2.0-flash-exp:free",
                            api_base="https://openrouter.ai/api/v1",
                            api_key=APIKEY_OPENROUTER)

openrouter_llama33_70b = dspy.LM(model="openrouter/meta-llama/llama-3.3-70b-instruct:free",
                            api_base="https://openrouter.ai/api/v1",
                            api_key=APIKEY_OPENROUTER)

openrouter_mistral_devstral2_123b = dspy.LM(model="openrouter/mistralai/devstral-2512:free",
                            api_base="https://openrouter.ai/api/v1",
                            api_key=APIKEY_OPENROUTER)

openrouter_Xiaomi_mimoV2_flash_15b_309b = dspy.LM(model="openrouter/xiaomi/mimo-v2-flash:free",
                            api_base="https://openrouter.ai/api/v1",
                            api_key=APIKEY_OPENROUTER)
openrouter_qwen3_coder_35b_480b = dspy.LM(model="openrouter/qwen/qwen3-coder:free",
                            api_base="https://openrouter.ai/api/v1",
                            api_key=APIKEY_OPENROUTER)

openrouter_deepseek_r1t2_chimera_671b = dspy.LM(model="openrouter/tngtech/deepseek-r1t2-chimera:free",
                            api_base="https://openrouter.ai/api/v1",
                            api_key=APIKEY_OPENROUTER)


print("Modelos cargados correctamente")

# %%
# -------------------------------------------------------------------------
# Definición de Signatures
# -------------------------------------------------------------------------

class Interpretador(dspy.Signature):
    "El interpretador se encarga de identificar y categorizar las solicitudes e indicaciones adicionales realizadas por el usuario, para facilitar una posterior interpretacion."

    prompt_usuario: str = dspy.InputField(
        desc="prompt del usuario en lenguaje natural."
    )

    escenarios_entrada: dict = dspy.InputField(
        desc=(
            "Escenarios estructurados admitidos por el sistema."
        )
    )   

    solicitudes_categorizadas: dict[str, dict] = dspy.OutputField(
    desc=(
        "Solicitudes segmentadas y categorizadas por el sistema. "
        "El resultado debe ser un único diccionario JSON, donde cada clave "
        "tiene el formato '@N' (N es un entero positivo consecutivo comenzando en 1), "
        "y cada valor es un diccionario con las siguientes claves:\n"
        "'solicitud' (string): solicitud específica y detallada, completamente autocontenida.\n"
        "'escenario' (string): escenario de entrada admitido por el sistema.\n"
        "'formato' (string): preferencia de presentación de la respuesta (valores: 'texto', 'grafico', 'mixto', 'no_especificado')."
        )
    )

    notas: str = dspy.OutputField(
    desc=(
        "Fragmentos textuales del prompt del usuario que no constituyen solicitudes."
        "No debe incluir explicaciones, interpretaciones, justificaciones ni metacomentarios del sistema."
        "El contenido debe presentarse como texto literal o parafraseado del usuario, sin análisis adicional."
        )
    )


# -------------------------------------------------------------------------
# Datos de Prueba
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

prompt_usuario=("Necesito saber cuánto consumió mi nevera ayer por la noche, "
    "y también cuánto consumió mi lavadora el sábado pasado en la mañana. "
    "Además quiero que me digas si entre esos dos días cuál gastó más energía. "
    "Ah, y por cierto, mientras miraba esos consumos se me descargó el celular "
    "y me dio mucha pereza pararme a buscar el cargador, pero igual quiero la comparación."
    "Papá también pidió que le dijeras cuánto fue el consumo de todos los dispositivos en el año 2024, quiero ver gráficas de todo lo que se pueda")

print("Signature y Datos de prueba cargados correctamente")
# %%

# -------------------------------------------------------------------------
# Ejecución de Modelos (Comentar/Descomentar según necesidad)
# -------------------------------------------------------------------------

# --- Llama 3.1 8b ---
dspy.configure(lm=llama_31_8b)

try:
    interpretador_llama31 = dspy.Predict(Interpretador)
    resultado_llama31 = interpretador_llama31(
        prompt_usuario = prompt_usuario,
        escenarios_entrada=escenarios_entrada,
    )

    print("\nInterpretador Llama 3.1 8b")
    print(resultado_llama31.solicitudes_categorizadas)
    print("\nNotas Llama 3.1 8b")
    print(resultado_llama31.notas)
except Exception as e:
    print(f"\nError en Interpretador Llama 3.1 8b: {e}")
print("\n###############################################")


#%%
# --- DeepSeek R1 8b ---
dspy.configure(lm=deepseek_r1_8b)

try:
    interpretador_deepseek_r1_8b = dspy.Predict(Interpretador)
    resultado_deepseek_r1_8b = interpretador_deepseek_r1_8b(
        prompt_usuario = prompt_usuario,
        escenarios_entrada=escenarios_entrada,
    )

    print("\n###############################################\nInterpretador DeepSeek R1 8b")
    print(resultado_deepseek_r1_8b.solicitudes_categorizadas)
    print("\nNotas DeepSeek R1 8b")
    print(resultado_deepseek_r1_8b.notas) 
except Exception as e:
    print(f"\nError en Interpretador DeepSeek R1 8b: {e}")
print("\n###############################################")

# %%

# --- Gemma 7b ---
dspy.configure(lm=gemma_7b)

try:
    interpretador_gemma_7b = dspy.Predict(Interpretador)
    resultado_gemma_7b = interpretador_gemma_7b(
        prompt_usuario = prompt_usuario,
        escenarios_entrada=escenarios_entrada,
    )

    print("\nInterpretador Gemma 7b")
    print(resultado_gemma_7b.solicitudes_categorizadas)
    print("\nNotas Gemma 7b")
    print(resultado_gemma_7b.notas)
except Exception as e:
    print(f"\nError en Interpretador Gemma 7b: {e}")
print("\n###############################################")

# %%

# --- Mistral 7b ---
dspy.configure(lm=mistral_7b)

try:
    interpretador_mistral_7b = dspy.Predict(Interpretador)
    resultado_mistral_7b = interpretador_mistral_7b(
        prompt_usuario = prompt_usuario,
        escenarios_entrada=escenarios_entrada,
    )

    print("\nInterpretador Mistral 7b")
    print(resultado_mistral_7b.solicitudes_categorizadas)
    print("\nNotas Mistral 7b")
    print(resultado_mistral_7b.notas)
except Exception as e:
    print(f"\nError en Interpretador Mistral 7b: {e}")
print("\n###############################################")
 
#%%

# --- Qwen 3 4b ---
dspy.configure(lm=qwen3_4b)

try:
    interpretador_qwen3_4b = dspy.Predict(Interpretador)
    resultado_qwen3_4b = interpretador_qwen3_4b(
        prompt_usuario = prompt_usuario,
        escenarios_entrada=escenarios_entrada,
    )

    print("\nInterpretador Qwen 3 4b")
    print(resultado_qwen3_4b.solicitudes_categorizadas)
    print("\nNotas Qwen 3 4b")
    print(resultado_qwen3_4b.notas)
except Exception as e:
    print(f"\nError en Interpretador Qwen 3 4b: {e}")
print("\n###############################################")

# %%
 
# --- OpenRouter Llama 3.3 70b ---
dspy.configure(lm=openrouter_llama33_70b)

try:
    interpretador_openrouter = dspy.Predict(Interpretador)
    resultado_openrouter = interpretador_openrouter(
        prompt_usuario = prompt_usuario,
        escenarios_entrada=escenarios_entrada,
    )

    print("\nInterpretador Llama 3.3 70b")
    print(resultado_openrouter.solicitudes_categorizadas)
    print("\nNotas Llama 3.3 70b")
    print(resultado_openrouter.notas) 
except Exception as e:
    print(f"\nError en Interpretador Llama 3.3 70b: {e}")
print("\n###############################################")
# %%

# --- OpenRouter Gemini 2.0 Flash ---
dspy.configure(lm=openrouter_gemini2flash)

try:
    interpretador_gemini2flash = dspy.Predict(Interpretador)
    resultado_gemini2flash = interpretador_gemini2flash(
        prompt_usuario = prompt_usuario,
        escenarios_entrada=escenarios_entrada,
    )

    print("\nInterpretador Gemini 2.0 Flash")
    print(resultado_gemini2flash.solicitudes_categorizadas)
    print("\nNotas Gemini 2.0 Flash")
    print(resultado_gemini2flash.notas)
except Exception as e:
    print(f"\nError en Interpretador Gemini 2.0 Flash: {e}")
print("\n###############################################")
# %%

# --- OpenRouter Mistral Devstral 2.123b ---
dspy.configure(lm=openrouter_mistral_devstral2_123b)

try:
    interpretador_openrouter = dspy.Predict(Interpretador)
    resultado_openrouter = interpretador_openrouter(
        prompt_usuario = prompt_usuario,
        escenarios_entrada=escenarios_entrada,
    )

    print("\nInterpretador Mistral Devstral2_123b")
    print(resultado_openrouter.solicitudes_categorizadas)
    print("\nNotas Mistral Devstral2_123b")
    print(resultado_openrouter.notas) 
except Exception as e:
    print(f"\nError en Interpretador Mistral Devstral2_123b: {e}")
print("\n###############################################")

# %%

# --- OpenRouter Xiaomi MimoV2 Flash 15b 309b ---
dspy.configure(lm=openrouter_Xiaomi_mimoV2_flash_15b_309b)

try:
    interpretador_openrouter = dspy.Predict(Interpretador)
    resultado_openrouter = interpretador_openrouter(
        prompt_usuario = prompt_usuario,
        escenarios_entrada=escenarios_entrada,
    )

    print("\nInterpretador Xiaomi MimoV2 Flash 15b 309b")
    print(resultado_openrouter.solicitudes_categorizadas)
    print("\nNotas Xiaomi MimoV2 Flash 15b 309b")
    print(resultado_openrouter.notas) 
except Exception as e:
    print(f"\nError en Interpretador Xiaomi MimoV2 Flash 15b 309b: {e}")
print("\n###############################################") 

# %%

# --- OpenRouter Qwen 3 Coder 35b 480b ---
dspy.configure(lm=openrouter_qwen3_coder_35b_480b)

try:
    interpretador_openrouter = dspy.Predict(Interpretador)
    resultado_openrouter = interpretador_openrouter(
        prompt_usuario = prompt_usuario,
        escenarios_entrada=escenarios_entrada,
    )

    print("\nInterpretador Qwen 3 Coder 35b 480b")
    print(resultado_openrouter.solicitudes_categorizadas)
    print("\nNotas Qwen 3 Coder 35b 480b")
    print(resultado_openrouter.notas) 
except Exception as e:
    print(f"\nError en Interpretador Qwen 3 Coder 35b 480b: {e}")
print("\n###############################################")
# %%

# --- OpenRouter Deepseek R1 T2 Chimera 671b ---
dspy.configure(lm=openrouter_deepseek_r1t2_chimera_671b)

try:
    interpretador_openrouter = dspy.Predict(Interpretador)
    resultado_openrouter = interpretador_openrouter(
        prompt_usuario = prompt_usuario,
        escenarios_entrada=escenarios_entrada,
    )

    print("\nInterpretador Deepseek R1 T2 Chimera 671b")
    print(resultado_openrouter.solicitudes_categorizadas)
    print("\nNotas Deepseek R1 T2 Chimera 671b")
    print(resultado_openrouter.notas) 
except Exception as e:
    print(f"\nError en Interpretador Deepseek R1 T2 Chimera 671b: {e}")
print("\n###############################################")

