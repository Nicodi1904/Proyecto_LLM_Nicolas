# %%
import dspy
from dotenv import load_dotenv
import os

# %%
# -------------------------------------------------------------------------
# 1. Configuración de PATH y carga de Dataset
# -------------------------------------------------------------------------

# Cargar variables de entorno desde archivo .env
env_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(dotenv_path=env_path)

# %%
# -------------------------------------------------------------------------
# 2. Configuración de LLMs
# -------------------------------------------------------------------------

APIKEY_GOOGLE = os.getenv("APIKEY_GOOGLE")
APIKEY_OPENROUTER = os.getenv("APIKEY_OPENROUTER")
# APIKEY_GEMINI_JUANC = os.getenv("APIKEY_GEMINI_JUANC")

# %%
# --- Modelos Ollama (Local) ---
llama_31_8b = dspy.LM("ollama_chat/llama3.1", api_base="http://localhost:11434", api_key="")
deepseek_r1_8b = dspy.LM("ollama_chat/deepseek-r1:8b", api_base="http://localhost:11434", api_key="")
gemma_7b = dspy.LM("ollama_chat/gemma2:9b", api_base="http://localhost:11434", api_key="")
mistral_7b = dspy.LM("ollama_chat/mistral:7b", api_base="http://localhost:11434", api_key="")
qwen3_4b = dspy.LM("ollama_chat/qwen2.5:3b", api_base="http://localhost:11434", api_key="")

# %%
# --- Modelos OpenRouter ---
openrouter_llama33_70b = dspy.LM(
    model="openrouter/meta-llama/llama-3.3-70b-instruct",
    api_key=APIKEY_OPENROUTER,
    api_base="https://openrouter.ai/api/v1"
)

openrouter_gemini2flash = dspy.LM(
    model="openrouter/google/gemini-2.0-flash-exp:free",
    api_key=APIKEY_OPENROUTER,
    api_base="https://openrouter.ai/api/v1"
)

openrouter_mistral_devstral2_123b = dspy.LM(
    model="openrouter/mistralai/devstral-2501",
    api_key=APIKEY_OPENROUTER,
    api_base="https://openrouter.ai/api/v1"
)

openrouter_Xiaomi_mimoV2_flash_15b_309b = dspy.LM(
    model="openrouter/microsoft/phi-4:free",
    api_key=APIKEY_OPENROUTER,
    api_base="https://openrouter.ai/api/v1"
)

openrouter_qwen3_coder_35b_480b = dspy.LM(
    model="openrouter/qwen/qwq-32b-preview",
    api_key=APIKEY_OPENROUTER,
    api_base="https://openrouter.ai/api/v1"
)

openrouter_deepseek_r1t2_chimera_671b = dspy.LM(
    model="openrouter/deepseek/deepseek-chat",
    api_key=APIKEY_OPENROUTER,
    api_base="https://openrouter.ai/api/v1"
)

# %%
""" # --- Gemini 2.5 Flash ---
gemini_25_flash = dspy.LM(
    model="gemini/gemini-2.0-flash-exp",
    api_key=APIKEY_GEMINI_JUANC
) """

print("Librerías y AppiKeys cargadas correctamente")
print("Modelos cargados correctamente")

# %%
# -------------------------------------------------------------------------
# 3. Definición de Signature
# -------------------------------------------------------------------------

class Gerente(dspy.Signature):
    """
    El Gerente es responsable de generar la respuesta final al usuario a partir de:
    - las solicitudes originales,
    - y un reporte unificado de acciones planificadas y ejecutadas.

    Su función es interpretar los resultados disponibles y comunicarlos de forma clara,
    coherente y alineada con las preferencias del usuario.
    """

    solicitudes_categorizadas: dict[str, dict] = dspy.InputField(
        desc=(
            "Solicitudes del usuario previamente segmentadas y categorizadas. "
            "Cada clave '@N' identifica una solicitud individual e incluye su formulación original, "
            "escenario funcional y preferencias de formato."
        )
    )

    reporte_acciones: dict[str, list[dict]] = dspy.InputField(
        desc=(
            "Reporte unificado de acciones asociadas a cada solicitud '@N'. "
            "Incluye tanto acciones ejecutadas como acciones no ejecutadas. "
            "Cada acción contiene:\n"
            "- su identificador,\n"
            "- la herramienta asociada,\n"
            "- una descripción semántica,\n"
            "- el resultado producido cuando existe,\n"
            "- y la causa documentada cuando la acción no pudo ejecutarse."
        )
    )

    respuesta_usuario: str = dspy.OutputField(
        desc=(
            "Respuesta final presentada al usuario, organizada por solicitud '@N'. "
            "Debe sintetizar la información contenida en el reporte de acciones, "
            "explicando de forma clara los resultados obtenidos y el alcance efectivo "
            "de cada solicitud."
        )
    )


print("Signature y Datos de prueba cargados correctamente")

# %%
# -------------------------------------------------------------------------
# Datos de Prueba
# -------------------------------------------------------------------------

# Ejemplo de solicitudes categorizadas (salida del Interpretador)
# Incluye solicitudes válidas (@1-@4) e inválidas (@5-@6)
solicitudes_categorizadas = {
    '@1': {'solicitud': 'Consumo de mi Ventilador ayer por la noche', 'escenario': 'consumo_basico', 'formato': 'texto'},
    '@2': {'solicitud': 'Consumo de mi PC el sábado pasado en la mañana', 'escenario': 'consumo_basico', 'formato': 'texto'},
    '@3': {'solicitud': 'Comparación del consumo de mi Ventilador y mi PC entre ayer y el sábado pasado', 'escenario': 'comparacion_consumos', 'formato': 'mixto'},
    '@4': {'solicitud': 'Consumo total de todos los dispositivos hasta noviembre del 2024', 'escenario': 'consumo_basico', 'formato': 'grafico'},
    '@5': {'solicitud': 'Quiero saber el consumo de mi Nevera la semana pasada', 'escenario': 'consumo_basico', 'formato': 'texto'},
    '@6': {'solicitud': 'Predice cuánto voy a gastar el próximo mes', 'escenario': 'prediccion', 'formato': 'texto'}
}


# Ejemplo de plan de acciones (salida del Planeador)
# Incluye acciones válidas (@1.1-@4.1) e inválidas (@5.1 con dispositivo desconocido, @6.1 con herramienta inexistente)
plan_acciones = [
    {
        'id': '@1.1',
        'server_id': 'mcp_server_gravity',
        'tool': 'obtener_consumo',
        'inputs': {
            'dispositivos': ['Ventilador'],
            'fecha_inicio': '2024-11-14T00:00',
            'fecha_fin': '2024-11-14T23:59'
        },
        'descripcion': 'Obtención de consumo del Ventilador en la noche del 14 de noviembre'
    },
    {
        'id': '@2.1',
        'server_id': 'mcp_server_gravity',
        'tool': 'obtener_consumo',
        'inputs': {
            'dispositivos': ['PC'],
            'fecha_inicio': '2024-11-10T00:00',
            'fecha_fin': '2024-11-10T23:59'
        },
        'descripcion': 'Obtención de consumo del PC en la mañana del 10 de noviembre'
    },
    {
        'id': '@3.1',
        'server_id': 'mcp_server_gravity',
        'tool': 'analizar_comparacion',
        'inputs': {
            'objetivo_a': {
                'dispositivo': 'Ventilador',
                'fecha_inicio': '2024-11-14T00:00',
                'fecha_fin': '2024-11-14T23:59'
            },
            'objetivo_b': {
                'dispositivo': 'PC',
                'fecha_inicio': '2024-11-10T00:00',
                'fecha_fin': '2024-11-10T23:59'
            }
        },
        'descripcion': 'Comparación del consumo entre el Ventilador y el PC'
    },
    {
        'id': '@4.1',
        'server_id': 'mcp_server_gravity',
        'tool': 'obtener_consumo',
        'inputs': {
            'dispositivos': ['Total_Casa'],
            'fecha_inicio': '2024-01-01T00:00',
            'fecha_fin': '2024-11-15T23:59'
        },
        'descripcion': 'Obtención del consumo total de la casa desde enero hasta noviembre'
    },
    {
        'id': '@5.1',
        'server_id': 'mcp_server_gravity',
        'tool': 'obtener_consumo',
        'inputs': {
            'dispositivos': ['Nevera'],  # Dispositivo NO válido (no está en dispositivos_conocidos)
            'fecha_inicio': '2024-11-08T00:00',
            'fecha_fin': '2024-11-15T23:59'
        },
        'descripcion': 'Obtención de consumo de la Nevera la semana pasada'
    },
    {
        'id': '@6.1',
        'server_id': 'mcp_server_gravity',
        'tool': 'predecir_consumo',  # Herramienta NO existente en el sistema
        'inputs': {
            'periodo': 'mes',
            'fecha_base': '2024-11-15'
        },
        'descripcion': 'Predicción del consumo para el próximo mes'
    }
]

# Reporte de ejecución (salida del MCP_C - SOLO acciones válidas ejecutadas)
reporte_ejecucion = {
  "@1": [
    {
      "accion_id": "@1.1",
      "tool": "obtener_consumo",
      "descripcion": "Obtención de consumo del Ventilador en la noche del 14 de noviembre",
      "resultado": {
        "status": "success",
        "periodo": {
          "inicio": "2024-11-14T00:00",
          "fin": "2024-11-14T23:59"
        },
        "granularidad": "total",
        "datos": {
          "Ventilador": 0.7234
        }
      },
      "error": None
    }
  ],
  "@2": [
    {
      "accion_id": "@2.1",
      "tool": "obtener_consumo",
      "descripcion": "Obtención de consumo del PC en la mañana del 10 de noviembre",
      "resultado": {
        "status": "success",
        "periodo": {
          "inicio": "2024-11-10T00:00",
          "fin": "2024-11-10T23:59"
        },
        "granularidad": "total",
        "datos": {
          "PC": 0.0029000000000000002
        }
      },
      "error": None
    }
  ],
  "@3": [
    {
      "accion_id": "@3.1",
      "tool": "analizar_comparacion",
      "descripcion": "Comparación del consumo entre el Ventilador y el PC",
      "resultado": {
        "status": "success",
        "comparacion": {
          "valor_a": 0.7234,
          "valor_b": 0.0029000000000000002,
          "diferencia_absoluta": 0.7205,
          "diferencia_porcentual": 24844.83,
          "mayor_consumo": "A"
        }
      },
      "error": None
    }
  ],
  "@4": [
    {
      "accion_id": "@4.1",
      "tool": "obtener_consumo",
      "descripcion": "Obtención del consumo total de la casa desde enero hasta noviembre",
      "resultado": {
        "status": "success",
        "periodo": {
          "inicio": "2024-01-01T00:00",
          "fin": "2024-11-15T23:59"
        },
        "granularidad": "total",
        "datos": {
          "Total_Casa": 2609.1866999999997
        }
      },
      "error": None
    }
  ]
}

# Acciones inválidas (salida del verificador - acciones rechazadas)
acciones_invalidas = [
    {
        'id': '@5.1',
        'server_id': 'mcp_server_gravity',
        'tool': 'obtener_consumo',
        'inputs': {
            'dispositivos': ['Nevera'],
            'fecha_inicio': '2024-11-08T00:00',
            'fecha_fin': '2024-11-15T23:59'
        },
        'descripcion': 'Obtención de consumo de la Nevera la semana pasada',
        'error_verificacion': "[Lógico] Dispositivo desconocido: 'Nevera'"
    },
    {
        'id': '@6.1',
        'server_id': 'mcp_server_gravity',
        'tool': 'predecir_consumo',
        'inputs': {
            'periodo': 'mes',
            'fecha_base': '2024-11-15'
        },
        'descripcion': 'Predicción del consumo para el próximo mes',
        'error_verificacion': "[Estructural] Herramienta desconocida: 'predecir_consumo' en servidor 'mcp_server_gravity'."
    }
]

# Importar función de consolidación desde MCP_C
import sys
import os
sys.path.append(os.path.dirname(__file__))
from MCP_C import consolidar_reportes

# Generar reporte consolidado usando la función de MCP_C
reporte_acciones = consolidar_reportes(reporte_ejecucion, acciones_invalidas)

print("Datos de prueba cargados correctamente (reporte consolidado generado)")
# %%
# -------------------------------------------------------------------------
# Ejecución de Modelos (Comentar/Descomentar según necesidad)
# -------------------------------------------------------------------------

if __name__ == "__main__":
    # --- Llama 3.1 8b ---
    dspy.configure(lm=llama_31_8b)

    try:
        gerente_llama31 = dspy.Predict(Gerente)
        resultado_llama31 = gerente_llama31(
            solicitudes_categorizadas=solicitudes_categorizadas,
            reporte_acciones=reporte_acciones
        )

        print("\nGerente Llama 3.1 8b\n")
        print("\nRespuesta Usuario:\n", resultado_llama31.respuesta_usuario)
    except Exception as e:
        print(f"\nError en Gerente Llama 3.1 8b: {e}")
    print("\n###############################################")

    # %%
    # --- DeepSeek R1 8b ---
    dspy.configure(lm=deepseek_r1_8b)

    try:
        gerente_deepseek = dspy.Predict(Gerente)
        resultado_deepseek = gerente_deepseek(
            solicitudes_categorizadas=solicitudes_categorizadas,
            reporte_acciones=reporte_acciones
        )

        print("\nGerente DeepSeek R1 8b\n")
        print("\nRespuesta Usuario:\n", resultado_deepseek.respuesta_usuario)
    except Exception as e:
        print(f"\nError en Gerente DeepSeek R1 8b: {e}")
    print("\n###############################################")

    """ # %%
    # --- Gemma 7b ---
    dspy.configure(lm=gemma_7b)

    try:
        gerente_gemma = dspy.Predict(Gerente)
        resultado_gemma = gerente_gemma(
            solicitudes_categorizadas=solicitudes_categorizadas,
            reporte_acciones=reporte_acciones
        )

        print("\nGerente Gemma 7b\n")
        print("\nRespuesta Usuario:\n", resultado_gemma.respuesta_usuario)
    except Exception as e:
        print(f"\nError en Gerente Gemma 7b: {e}")
    print("\n###############################################")

    # %%
    # --- Mistral 7b ---
    dspy.configure(lm=mistral_7b)

    try:
        gerente_mistral = dspy.Predict(Gerente)
        resultado_mistral = gerente_mistral(
            solicitudes_categorizadas=solicitudes_categorizadas,
            reporte_acciones=reporte_acciones
        )

        print("\nGerente Mistral 7b\n")
        print("\nRespuesta Usuario:\n", resultado_mistral.respuesta_usuario)
    except Exception as e:
        print(f"\nError en Gerente Mistral 7b: {e}")
    print("\n###############################################")

    # %%
    # --- Qwen 3 4b ---
    dspy.configure(lm=qwen3_4b)

    try:
        gerente_qwen = dspy.Predict(Gerente)
        resultado_qwen = gerente_qwen(
            solicitudes_categorizadas=solicitudes_categorizadas,
            reporte_acciones=reporte_acciones
        )

        print("\nGerente Qwen 3 4b\n")
        print("\nRespuesta Usuario:\n", resultado_qwen.respuesta_usuario)
    except Exception as e:
        print(f"\nError en Gerente Qwen 3 4b: {e}")
    print("\n###############################################")

    # %%
    # --- OpenRouter Llama 3.3 70b ---
    dspy.configure(lm=openrouter_llama33_70b)

    try:
        gerente_llama33 = dspy.Predict(Gerente)
        resultado_llama33 = gerente_llama33(
            solicitudes_categorizadas=solicitudes_categorizadas,
            reporte_acciones=reporte_acciones
        )

        print("\nGerente Llama 3.3 70b\n")
        print("\nRespuesta Usuario:\n", resultado_llama33.respuesta_usuario)
    except Exception as e:
        print(f"\nError en Gerente Llama 3.3 70b: {e}")
    print("\n###############################################")


    # %%
    # --- OpenRouter Gemini 2.0 Flash ---
    dspy.configure(lm=openrouter_gemini2flash)

    try:
        gerente_gemini2 = dspy.Predict(Gerente)
        resultado_gemini2 = gerente_gemini2(
            solicitudes_categorizadas=solicitudes_categorizadas,
            reporte_acciones=reporte_acciones
        )

        print("\nGerente Gemini 2.0 Flash\n")
        print("\nRespuesta Usuario:\n", resultado_gemini2.respuesta_usuario)
    except Exception as e:
        print(f"\nError en Gerente Gemini 2.0 Flash: {e}")
    print("\n###############################################")

    # %%
    # --- OpenRouter Mistral Devstral 2.123b ---
    dspy.configure(lm=openrouter_mistral_devstral2_123b)

    try:
        gerente_mistral_dev = dspy.Predict(Gerente)
        resultado_mistral_dev = gerente_mistral_dev(
            solicitudes_categorizadas=solicitudes_categorizadas,
            reporte_acciones=reporte_acciones
        )

        print("\nGerente Mistral Devstral2 123b\n")
        print("\nRespuesta Usuario:\n", resultado_mistral_dev.respuesta_usuario)
    except Exception as e:
        print(f"\nError en Gerente Mistral Devstral2 123b: {e}")
    print("\n###############################################")

    # %%
    # --- OpenRouter Xiaomi MimoV2 Flash ---
    dspy.configure(lm=openrouter_Xiaomi_mimoV2_flash_15b_309b)

    try:
        gerente_xiaomi = dspy.Predict(Gerente)
        resultado_xiaomi = gerente_xiaomi(
            solicitudes_categorizadas=solicitudes_categorizadas,
            reporte_acciones=reporte_acciones
        )

        print("\nGerente Xiaomi MimoV2 Flash\n")
        print("\nRespuesta Usuario:\n", resultado_xiaomi.respuesta_usuario)
    except Exception as e:
        print(f"\nError en Gerente Xiaomi MimoV2 Flash: {e}")
    print("\n###############################################")

    # %%
    # --- OpenRouter Qwen 3 Coder ---
    dspy.configure(lm=openrouter_qwen3_coder_35b_480b)

    try:
        gerente_qwen_coder = dspy.Predict(Gerente)
        resultado_qwen_coder = gerente_qwen_coder(
            solicitudes_categorizadas=solicitudes_categorizadas,
            reporte_acciones=reporte_acciones
        )

        print("\nGerente Qwen 3 Coder\n")
        print("\nRespuesta Usuario:\n", resultado_qwen_coder.respuesta_usuario)
    except Exception as e:
        print(f"\nError en Gerente Qwen 3 Coder: {e}")
    print("\n###############################################")

    # %%
    # --- OpenRouter Deepseek R1 T2 Chimera ---
    dspy.configure(lm=openrouter_deepseek_r1t2_chimera_671b)

    try:
        gerente_deepseek_chimera = dspy.Predict(Gerente)
        resultado_deepseek_chimera = gerente_deepseek_chimera(
            solicitudes_categorizadas=solicitudes_categorizadas,
            reporte_acciones=reporte_acciones
        )

        print("\nGerente Deepseek R1 T2 Chimera\n")
        print("\nRespuesta Usuario:\n", resultado_deepseek_chimera.respuesta_usuario)
    except Exception as e:
        print(f"\nError en Gerente Deepseek R1 T2 Chimera: {e}")
    print("\n###############################################")
 """