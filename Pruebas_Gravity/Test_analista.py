# %%
import dspy
from dotenv import load_dotenv
import os

# %%
# -------------------------------------------------------------------------
# 1. Configuración de PATH y carga de Dataset
# -------------------------------------------------------------------------

# Cargar variables de entorno desde archivo .env
env_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '.env'))
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

class Analista(dspy.Signature):
    """
    Signature para el Analista (Por definir).
    """
    
    # TODO: Definir inputs y outputs del Analista
    pass

print("Signature y Datos de prueba cargados correctamente")

# %%
# -------------------------------------------------------------------------
# Datos de Prueba
# -------------------------------------------------------------------------

# TODO: Definir datos de prueba para el Analista

# %%
# -------------------------------------------------------------------------
# Ejecución de Modelos (Comentar/Descomentar según necesidad)
# -------------------------------------------------------------------------

if __name__ == "__main__":
    print("\n" + "="*60)
    print("ANALISTA - PENDIENTE DE IMPLEMENTACIÓN")
    print("="*60 + "\n")
    
    # --- Llama 3.1 8b ---
    # dspy.configure(lm=llama_31_8b)
    # try:
    #     analista_llama31 = dspy.Predict(Analista)
    #     resultado_llama31 = analista_llama31(...)
    #     print("\nAnalista Llama 3.1 8b\n")
    #     print(resultado_llama31)
    # except Exception as e:
    #     print(f"\nError en Analista Llama 3.1 8b: {e}")
    # print("\n###############################################")

    # %%
    # --- DeepSeek R1 8b ---
    # dspy.configure(lm=deepseek_r1_8b)
    # try:
    #     analista_deepseek = dspy.Predict(Analista)
    #     resultado_deepseek = analista_deepseek(...)
    #     print("\nAnalista DeepSeek R1 8b\n")
    #     print(resultado_deepseek)
    # except Exception as e:
    #     print(f"\nError en Analista DeepSeek R1 8b: {e}")
    # print("\n###############################################")

    # %%
    # --- Gemma 7b ---
    # dspy.configure(lm=gemma_7b)
    # try:
    #     analista_gemma = dspy.Predict(Analista)
    #     resultado_gemma = analista_gemma(...)
    #     print("\nAnalista Gemma 7b\n")
    #     print(resultado_gemma)
    # except Exception as e:
    #     print(f"\nError en Analista Gemma 7b: {e}")
    # print("\n###############################################")

    # %%
    # --- Mistral 7b ---
    # dspy.configure(lm=mistral_7b)
    # try:
    #     analista_mistral = dspy.Predict(Analista)
    #     resultado_mistral = analista_mistral(...)
    #     print("\nAnalista Mistral 7b\n")
    #     print(resultado_mistral)
    # except Exception as e:
    #     print(f"\nError en Analista Mistral 7b: {e}")
    # print("\n###############################################")

    # %%
    # --- Qwen 3 4b ---
    # dspy.configure(lm=qwen3_4b)
    # try:
    #     analista_qwen = dspy.Predict(Analista)
    #     resultado_qwen = analista_qwen(...)
    #     print("\nAnalista Qwen 3 4b\n")
    #     print(resultado_qwen)
    # except Exception as e:
    #     print(f"\nError en Analista Qwen 3 4b: {e}")
    # print("\n###############################################")

    # %%
    # --- OpenRouter Llama 3.3 70b ---
    # dspy.configure(lm=openrouter_llama33_70b)
    # try:
    #     analista_llama33 = dspy.Predict(Analista)
    #     resultado_llama33 = analista_llama33(...)
    #     print("\nAnalista Llama 3.3 70b\n")
    #     print(resultado_llama33)
    # except Exception as e:
    #     print(f"\nError en Analista Llama 3.3 70b: {e}")
    # print("\n###############################################")

    # %%
    # --- OpenRouter Gemini 2.0 Flash ---
    # dspy.configure(lm=openrouter_gemini2flash)
    # try:
    #     analista_gemini2 = dspy.Predict(Analista)
    #     resultado_gemini2 = analista_gemini2(...)
    #     print("\nAnalista Gemini 2.0 Flash\n")
    #     print(resultado_gemini2)
    # except Exception as e:
    #     print(f"\nError en Analista Gemini 2.0 Flash: {e}")
    # print("\n###############################################")

    # %%
    # --- OpenRouter Mistral Devstral 2.123b ---
    # dspy.configure(lm=openrouter_mistral_devstral2_123b)
    # try:
    #     analista_mistral_dev = dspy.Predict(Analista)
    #     resultado_mistral_dev = analista_mistral_dev(...)
    #     print("\nAnalista Mistral Devstral2 123b\n")
    #     print(resultado_mistral_dev)
    # except Exception as e:
    #     print(f"\nError en Analista Mistral Devstral2 123b: {e}")
    # print("\n###############################################")

    # %%
    # --- OpenRouter Xiaomi MimoV2 Flash ---
    # dspy.configure(lm=openrouter_Xiaomi_mimoV2_flash_15b_309b)
    # try:
    #     analista_xiaomi = dspy.Predict(Analista)
    #     resultado_xiaomi = analista_xiaomi(...)
    #     print("\nAnalista Xiaomi MimoV2 Flash\n")
    #     print(resultado_xiaomi)
    # except Exception as e:
    #     print(f"\nError en Analista Xiaomi MimoV2 Flash: {e}")
    # print("\n###############################################")

    # %%
    # --- OpenRouter Qwen 3 Coder ---
    # dspy.configure(lm=openrouter_qwen3_coder_35b_480b)
    # try:
    #     analista_qwen_coder = dspy.Predict(Analista)
    #     resultado_qwen_coder = analista_qwen_coder(...)
    #     print("\nAnalista Qwen 3 Coder\n")
    #     print(resultado_qwen_coder)
    # except Exception as e:
    #     print(f"\nError en Analista Qwen 3 Coder: {e}")
    # print("\n###############################################")

    # %%
    # --- OpenRouter Deepseek R1 T2 Chimera ---
    # dspy.configure(lm=openrouter_deepseek_r1t2_chimera_671b)
    # try:
    #     analista_deepseek_chimera = dspy.Predict(Analista)
    #     resultado_deepseek_chimera = analista_deepseek_chimera(...)
    #     print("\nAnalista Deepseek R1 T2 Chimera\n")
    #     print(resultado_deepseek_chimera)
    # except Exception as e:
    #     print(f"\nError en Analista Deepseek R1 T2 Chimera: {e}")
    # print("\n###############################################")
