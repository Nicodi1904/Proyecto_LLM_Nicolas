# %%
import dspy
import os
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
        print("No se pudo importar el summary del script hermano.")
        system_summary = None

# Cargar API Keys
env_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(env_path)

APIKEY_GOOGLE = os.getenv("APIKEY_GOOGLE")
APIKEY_OPENROUTER = os.getenv("APIKEY_OPENROUTER")
#APIKEY_GEMINI_JUANC = os.getenv("APIKEY_GEMINI_JUANC")
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

""" # Gemini 2.5 Flash (via Google OpenAI-compatible endpoint)
gemini_25_flash = dspy.LM(model="openai/gemini-2.5-flash", 
                            api_base="https://generativelanguage.googleapis.com/v1beta/openai/",
                            api_key=APIKEY_GEMINI_JUANC)
 """
print("Modelos cargados correctamente")

# %%
# -------------------------------------------------------------------------
# Definición de Signatures
# -------------------------------------------------------------------------

class Planeador(dspy.Signature):
    """
    Traduce las solicitudes categorizadas por el inferenciador
    en un conjunto estructurado de acciones sobre las herramientas disponibles del sistema.
    Su función es decidir qué herramientas utilizar, en qué orden y con qué parámetros,
    evaluando la viabilidad funcional de cada solicitud.
    """

    solicitudes_categorizadas: dict[str, dict] = dspy.InputField(
        desc=(
            "Conjunto de solicitudes previamente segmentadas y categorizadas por el inferenciador. "
            "Cada clave representa una solicitud individual identificada con un identificador '@N'. "
            "Cada valor es un diccionario con las siguientes claves:\n"
            "- 'solicitud' (string): formulación autocontenida de la intención del usuario, utilizada "
            "como base principal para la planificación de acciones.\n"
            "- 'escenario' (string): clasificación funcional inferida (por ejemplo, consumo_basico, "
            "comparacion_consumos, deteccion_anomalias). Este campo sirve como guía semántica para "
            "priorizar y seleccionar herramientas.\n"
            "- 'formato' (string): preferencia de presentación de la respuesta esperada por el usuario, "
            "puede ser ('texto', 'grafico', 'mixto' o 'no_especificado') y debe ser considerada como parte de los requisitos de la solicitud."
        )
    )

    system_summary: dict = dspy.InputField(
        desc=(
            "Resumen estructurado de las herramientas disponibles en el sistema, organizado por servidores. "
            "Incluye, para cada herramienta, su propósito, criterios de uso, esquema de entradas (input_schema) "
            "y esquema de salidas (output_schema). Esta información debe ser utilizada por el planeador para:\n"
            "- determinar qué herramientas son funcionalmente adecuadas para cada solicitud,\n"
            "- construir cadenas de acciones donde la salida de una herramienta pueda servir como entrada de otra,\n"
            "- y evaluar la viabilidad de un plan antes de su ejecución."
        )
    )
    temporal_context: dict = dspy.InputField(
    desc=(
        "Contexto temporal del sistema, utilizado como referencia para interpretar "
        "expresiones temporales relativas presentes en las solicitudes del usuario. "
        "Los rangos son disjuntos, no se solapan y no deben combinarse ni extenderse "
        "fuera de sus límites definidos."
        )
    )



    plan_acciones: list[dict] = dspy.OutputField(
        desc=(
            "Lista estructurada de acciones planificadas para resolver las solicitudes del usuario. "
            "Cada elemento de la lista representa una acción individual e incluye:\n"
            "- 'id' (string): identificador único de la acción, con el formato '@N.M', donde '@N' "
            "corresponde a la solicitud de origen y 'M' indica el orden secuencial de la acción dentro "
            "de dicha solicitud.\n"
            "- 'server_id' (string): identificador del servidor donde se encuentra la herramienta a invocar.\n"
            "- 'tool' (string): nombre de la herramienta seleccionada.\n"
            "- 'inputs' (dict): parámetros de entrada de la herramienta, que pueden incluir referencias "
            "a salidas de acciones previas mediante identificadores '@N.M'.\n"
            "- 'descripcion' (string): breve descripción semántica de la acción y su propósito dentro del plan."
        )
    )

    estado_solicitudes: dict[str, dict] = dspy.OutputField(
    desc=(
        "Estado final de cada solicitud procesada. Cada clave '@N' corresponde "
        "a cada solicitud de entrada y su valor es un diccionario con las claves:\n"
        "- 'estado':\n"
        "* 'resuelta': cuando la solicitud, criterios y preferencias se pueden satisfacer completamente.\n"
        "* 'parcial': cuando solo una parte de la solicitud puede resolverse, "
        "o alguno de los requisitos o preferencias del usuario no pueden cumplirse.\n"
        "* 'no_resuelta': cuando la solicitud no puede resolverse con las herramientas disponibles.\n"
        "- 'motivo' (obligatorio si estado != 'resuelta'): diccionario con las claves:\n"
        "- 'tipo': categoría del problema detectado. Valores permitidos:\n"
        "* 'falta_informacion_usuario': la solicitud no contiene información suficiente "
        "para construir un plan ejecutable completo, aun cuando el sistema tendría "
        "capacidad para resolverla si dicha información estuviera disponible.\n"
        "* 'parametros_incompatibles': la solicitud especifica parámetros que no pueden coexistir de forma válida " 
        "según los esquemas y restricciones de las herramientas disponibles en el sistema.\n"
        "* 'limitacion_sistema': la solicitud requiere una capacidad que el sistema no posee "
        "según la definición actual de herramientas disponibles.\n"
        "- 'detalle': descripción breve y concreta del problema, indicando qué parte "
        "de la solicitud no pudo resolverse."
    )
)


# -------------------------------------------------------------------------
# Datos de Prueba
# -------------------------------------------------------------------------
# Simulación de salida del Interpretador (Usando ejemplo de Llama 3.3 70b)
#solicitudes_categorizadas = {'@1': {'solicitud': 'Necesito saber cuánto consumió mi nevera ayer por la noche.', 'escenario': 'consumo_basico', 'formato': 'texto'}, '@2': {'solicitud': 'Necesito saber cuánto consumió mi lavadora el sábado pasado en la mañana.', 'escenario': 'consumo_basico', 'formato': 'texto'}, '@3': {'solicitud': 'Quiero que me digas si entre el consumo de mi nevera ayer por la noche y el consumo de mi lavadora el sábado pasado en la mañana, cuál gastó más energía.', 'escenario': 'comparacion_consumos', 'formato': 'mixto'}, '@4': {'solicitud': 'Quiero saber cuánto fue el consumo de todos los dispositivos en el año 2024.', 'escenario': 'consumo_basico', 'formato': 'grafico'}}
solicitudes_categorizadas = {'@1': {'solicitud': 'Consumo de mi Ventilador ayer por la noche', 'escenario': 'consumo_basico', 'formato': 'texto'}, '@2': {'solicitud': 'Consumo de mi PC el sábado pasado en la mañana', 'escenario': 'consumo_basico', 'formato': 'texto'}, '@3': {'solicitud': 'Comparación del consumo de mi Ventilador y mi PC entre ayer y el sábado pasado', 'escenario': 'comparacion_consumos', 'formato': 'mixto'}, '@4': {'solicitud': 'Consumo total de todos los dispositivos hasta noviembre del 2024', 'escenario': 'consumo_basico', 'formato': 'grafico'}}
temporal_context = {
    "referencia_actual": "2024-11-15T10:30",
    "zona_horaria": "America/Bogota",
    "rangos_horarios": {
        "madrugada": {"inicio": "00:00", "fin": "05:59"},
        "mañana": {"inicio": "06:00", "fin": "11:59"},
        "tarde": {"inicio": "12:00", "fin": "17:59"},
        "noche": {"inicio": "18:00", "fin": "23:59"}
    }
}

print("Signature y Datos de prueba cargados correctamente")

# %%
if __name__ == "__main__":
    # -------------------------------------------------------------------------
    # Ejecución de Modelos (Comentar/Descomentar según necesidad)
    # -------------------------------------------------------------------------

    # --- Llama 3.1 8b ---
    dspy.configure(lm=llama_31_8b)

    try:
        planeador_llama31 = dspy.Predict(Planeador)
        resultado_llama31 = planeador_llama31(
            solicitudes_categorizadas = solicitudes_categorizadas,
            system_summary = system_summary,
            temporal_context = temporal_context
        )

        print("\nPlaneador Llama 3.1 8b\n")
        print("\nPlan:\n", resultado_llama31.plan_acciones)
        print("\nEstado Solicitudes:\n", resultado_llama31.estado_solicitudes)
    except Exception as e:
        print(f"\nError en Planeador Llama 3.1 8b: {e}")
    print("\n###############################################")

    # %%
    # --- DeepSeek R1 8b ---
    dspy.configure(lm=deepseek_r1_8b)

    try:
        planeador_deepseek_r1_8b = dspy.Predict(Planeador)
        resultado_deepseek_r1_8b = planeador_deepseek_r1_8b(
            solicitudes_categorizadas = solicitudes_categorizadas,
            system_summary = system_summary,
            temporal_context = temporal_context 
        )

        print("\nPlaneador DeepSeek R1 8b\n")
        print("\nPlan:\n", resultado_deepseek_r1_8b.plan_acciones)
        print("\nEstado Solicitudes:\n", resultado_deepseek_r1_8b.estado_solicitudes)
    except Exception as e:
        print(f"\nError en Planeador DeepSeek R1 8b: {e}")
    print("\n###############################################")

    # %%
    # --- Gemma 7b ---
    dspy.configure(lm=gemma_7b)

    try:
        planeador_gemma_7b = dspy.Predict(Planeador)
        resultado_gemma_7b = planeador_gemma_7b(
            solicitudes_categorizadas = solicitudes_categorizadas,
            system_summary = system_summary,
            temporal_context = temporal_context
        )

        print("\nPlaneador Gemma 7b\n")
        print("\nPlan:\n", resultado_gemma_7b.plan_acciones)
        print("\nEstado Solicitudes:\n", resultado_gemma_7b.estado_solicitudes)
    except Exception as e:
        print(f"\nError en Planeador Gemma 7b: {e}")
    print("\n###############################################")

    # %%
    # --- Mistral 7b ---
    dspy.configure(lm=mistral_7b)

    try:
        planeador_mistral_7b = dspy.Predict(Planeador)
        resultado_mistral_7b = planeador_mistral_7b(
            solicitudes_categorizadas = solicitudes_categorizadas,
            system_summary = system_summary,
            temporal_context = temporal_context
        )

        print("\nPlaneador Mistral 7b\n")
        print("\nPlan:\n", resultado_mistral_7b.plan_acciones)
        print("\nEstado Solicitudes:\n", resultado_mistral_7b.estado_solicitudes)
    except Exception as e:
        print(f"\nError en Planeador Mistral 7b: {e}")
    print("\n###############################################")

    # %%
    # --- Qwen 3 4b ---
    dspy.configure(lm=qwen3_4b)

    try:
        planeador_qwen3_4b = dspy.Predict(Planeador)
        resultado_qwen3_4b = planeador_qwen3_4b(
            solicitudes_categorizadas = solicitudes_categorizadas,
            system_summary = system_summary,
            temporal_context = temporal_context
        )

        print("\nPlaneador Qwen 3 4b\n")
        print("\nPlan:\n", resultado_qwen3_4b.plan_acciones)
        print("\nEstado Solicitudes:\n", resultado_qwen3_4b.estado_solicitudes)
    except Exception as e:
        print(f"\nError en Planeador Qwen 3 4b: {e}")
    print("\n###############################################")

    # %%
    # --- OpenRouter Llama 3.3 70b ---
    dspy.configure(lm=openrouter_llama33_70b)

    try:
        planeador_openrouter = dspy.Predict(Planeador)
        resultado_openrouter = planeador_openrouter(
            solicitudes_categorizadas = solicitudes_categorizadas,
            system_summary = system_summary,
            temporal_context = temporal_context
        )

        print("\nPlaneador Llama 3.3 70b\n")
        print("\nPlan:\n", resultado_openrouter.plan_acciones)
        print("\nEstado Solicitudes:\n", resultado_openrouter.estado_solicitudes)
    except Exception as e:
        print(f"\nError en Planeador Llama 3.3 70b: {e}")
    print("\n###############################################")

    # %%
    # --- OpenRouter Gemini 2.0 Flash ---
    dspy.configure(lm=openrouter_gemini2flash)

    try:
        planeador_gemini2flash = dspy.Predict(Planeador)
        resultado_gemini2flash = planeador_gemini2flash(
            solicitudes_categorizadas = solicitudes_categorizadas,
            system_summary = system_summary,
            temporal_context = temporal_context
        )

        print("\nPlaneador Gemini 2.0 Flash\n")
        print("\nPlan:\n", resultado_gemini2flash.plan_acciones)
        print("\nEstado Solicitudes:\n", resultado_gemini2flash.estado_solicitudes)
    except Exception as e:
        print(f"\nError en Planeador Gemini 2.0 Flash: {e}")
    print("\n###############################################")

    # %%
    # --- OpenRouter Mistral Devstral 2.123b ---
    dspy.configure(lm=openrouter_mistral_devstral2_123b)

    try:
        planeador_openrouter = dspy.Predict(Planeador)
        resultado_openrouter = planeador_openrouter(
            solicitudes_categorizadas = solicitudes_categorizadas,
            system_summary = system_summary,
            temporal_context = temporal_context
        )

        print("\nPlaneador Mistral Devstral2_123b\n")
        print("\nPlan:\n", resultado_openrouter.plan_acciones)
        print("\nEstado Solicitudes:\n", resultado_openrouter.estado_solicitudes)
    except Exception as e:
        print(f"\nError en Planeador Mistral Devstral2_123b: {e}")
    print("\n###############################################")

    # %%
    # --- OpenRouter Xiaomi MimoV2 Flash 15b 309b ---
    dspy.configure(lm=openrouter_Xiaomi_mimoV2_flash_15b_309b)

    try:
        planeador_openrouter = dspy.Predict(Planeador)
        resultado_openrouter = planeador_openrouter(
            solicitudes_categorizadas = solicitudes_categorizadas,
            system_summary = system_summary,
            temporal_context = temporal_context
        )

        print("\nPlaneador Xiaomi MimoV2 Flash 15b 309b\n")
        print("\nPlan:\n", resultado_openrouter.plan_acciones)
        print("\nEstado Solicitudes:\n", resultado_openrouter.estado_solicitudes)
    except Exception as e:
        print(f"\nError en Planeador Xiaomi MimoV2 Flash 15b 309b: {e}")
    print("\n###############################################")

    # %%
    # --- OpenRouter Qwen 3 Coder 35b 480b ---
    dspy.configure(lm=openrouter_qwen3_coder_35b_480b)

    try:
        planeador_openrouter = dspy.Predict(Planeador)
        resultado_openrouter = planeador_openrouter(
            solicitudes_categorizadas = solicitudes_categorizadas,
            system_summary = system_summary,
            temporal_context = temporal_context
        )

        print("\nPlaneador Qwen 3 Coder 35b 480b\n")
        print("\nPlan:\n", resultado_openrouter.plan_acciones)
        print("\nEstado Solicitudes:\n", resultado_openrouter.estado_solicitudes)
    except Exception as e:
        print(f"\nError en Planeador Qwen 3 Coder 35b 480b: {e}")
    print("\n###############################################")

    # %%
    # --- OpenRouter Deepseek R1 T2 Chimera 671b ---
    dspy.configure(lm=openrouter_deepseek_r1t2_chimera_671b)

    try:
        planeador_openrouter = dspy.Predict(Planeador)
        resultado_openrouter = planeador_openrouter(
            solicitudes_categorizadas = solicitudes_categorizadas,
            system_summary = system_summary,
            temporal_context = temporal_context
        )

        print("\nPlaneador Deepseek R1 T2 Chimera 671b\n")
        print("\nPlan:\n", resultado_openrouter.plan_acciones)
        print("\nEstado Solicitudes:\n", resultado_openrouter.estado_solicitudes)
    except Exception as e:
        print(f"\nError en Planeador Deepseek R1 T2 Chimera 671b: {e}")
    print("\n###############################################")


    """ # %%
    # --- Gemini 2.5 Flash ---
    dspy.configure(lm=gemini_25_flash)

    try:
        planeador_gemini = dspy.Predict(Planeador)
        resultado_gemini = planeador_gemini(
            solicitudes_categorizadas = solicitudes_categorizadas,
            system_summary = system_summary,
            temporal_context = temporal_context
        )

        print("\nPlaneador Gemini 2.5 Flash\n")
        print("\nPlan:\n", resultado_gemini.plan_acciones)
        print("\nEstado Solicitudes:\n", resultado_gemini.estado_solicitudes)
    except Exception as e:
        print(f"\nError en Planeador Gemini 2.5 Flash: {e}")
    print("\n###############################################") """