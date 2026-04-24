import dspy
import json
import os
import time
from typing import Any, List, Dict, Optional
from dspy.teleprompt import BootstrapFewShot

# Desactivar cache de DSPy para asegurar que cada llamada sea real y fresca
os.environ["DSP_CACHEBOOL"] = "False"

# --- 1. CONFIGURACIÓN DE ENTORNO Y RUTAS ---
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
CASOS_DIR = os.path.join(BASE_PATH, 'Set_casos')
OUTPUT_DIR = os.path.join(BASE_PATH, 'resultados_fewShots')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =================================================================
# 🔑 CONFIGURACIÓN DE MODELOS (COMENTADOS)
# =================================================================
# APIKEY_OPENROUTER = "TU_API_KEY_AQUI"
# lm = dspy.LM(
#     model="openrouter/meta-llama/llama-3.3-70b-instruct",
#     api_base="https://openrouter.ai/api/v1",
#     api_key=APIKEY_OPENROUTER,
#     max_tokens=4000
# )
# dspy.settings.configure(lm=lm)
# =================================================================

# --- 2. DEFINICIÓN DE SIGNATURES Y AGENTE (IDENTICOS AL ORIGINAL) ---

class Interpretador(dspy.Signature):
    "Identifica las solicitudes realizadas por el usuario y las categoriza."

    prompt_usuario: str = dspy.InputField(
        desc="prompt del usuario en lenguaje natural."
    )

    solicitudes_categorizadas: dict[str, dict] = dspy.OutputField(
        desc=(
            "Solicitudes segmentadas y categorizadas por el sistema. "
            "El resultado debe ser un único diccionario JSON, donde cada clave "
            "tiene el formato '@N' (N es un entero positivo consecutivo comenzando en 1), "
            "y cada valor es un diccionario con las siguientes claves:\n"
            "'solicitud' (string): solicitud específica y detallada, completamente autocontenida, no debe depender de otras solicitudes.\n"
            "'escenario' (string): escenario de entrada admitido por el sistema."
        )
    )

    notas: str = dspy.OutputField(
        desc=("razonamiento que llevó a elegir el escenario para cada solicitud")
    )

class InterpretadorAgente(dspy.Module):
    def __init__(self, escenarios: dict = None):
        super().__init__()
        
        # Si no se proporcionan escenarios, los cargamos desde el archivo local
        if escenarios is None:
            escenarios = self._cargar_escenarios()
        
        # Almacenamos los escenarios para acceso posterior (ej. validación)
        self.escenarios = escenarios
        
        # Se transforma el Json para que le llegue mejor al modelo y se inicia como instrucciones del sistema
        instruccion_sistema = f"Escenarios disponibles admitidos por el sistema:\n{json.dumps(escenarios, indent=2, ensure_ascii=False)}"

        # Creamos el predictor con la Signature modificada
        self.predictor = dspy.Predict(Interpretador.with_instructions(instruccion_sistema))

    def _cargar_escenarios(self) -> dict:
        """Busca y carga el archivo escenarios.json en el mismo directorio que el script."""
        # Nota: Ajustado para buscar en la carpeta del agente original si es necesario, 
        # o puedes copiar escenarios.json a la carpeta de pruebas.
        ruta_json = os.path.join(os.path.dirname(__file__), 'escenarios.json')
        if not os.path.exists(ruta_json):
            # Intento de ruta relativa al proyecto
            ruta_alt = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
                                  'Agente_energetico', 'Sistema_entrada', 'Interpretador', 'escenarios.json')
            ruta_json = ruta_alt if os.path.exists(ruta_alt) else ruta_json

        try:
            with open(ruta_json, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return {}

    def validar_formato(self, solicitudes_categorizadas: Any) -> List[str]:
        errores = []
        if not isinstance(solicitudes_categorizadas, dict):
            return ["La salida de solicitudes categorizadas debe ser un diccionario."]
        if not solicitudes_categorizadas:
            return ["No se encontraron solicitudes en la salida."]

        for key, val in solicitudes_categorizadas.items():
            if not (isinstance(key, str) and key.startswith("@")):
                errores.append(f"Clave de petición con formato incorrecto: '{key}'. Debe comenzar con '@'.")
            if not isinstance(val, dict):
                errores.append(f"El valor de la petición '{key}' debe ser un diccionario.")
                continue
            if "solicitud" not in val or not val["solicitud"]:
                errores.append(f"Falta el campo 'solicitud' o está vacío en '{key}'.")
            if "escenario" not in val or not val["escenario"]:
                errores.append(f"Falta el campo 'escenario' o está vacío en '{key}'.")
        return errores

    def validar_coherencia(self, solicitudes_categorizadas: Dict[str, Any]) -> List[str]:
        errores = []
        escenarios_validos = self.escenarios
        if not isinstance(solicitudes_categorizadas, dict): return []

        for key, val in solicitudes_categorizadas.items():
            if not isinstance(val, dict): continue
            escenario_nombre = val.get("escenario")
            if escenario_nombre and escenario_nombre not in escenarios_validos:
                errores.append(f"Escenario desconocido en '{key}': '{escenario_nombre}'.")
        return errores

    def worker1(self, prediction: Any) -> Dict[str, Any]:
        solicitudes = getattr(prediction, "solicitudes_categorizadas", None)
        errores_formato = self.validar_formato(solicitudes)
        errores_coherencia = self.validar_coherencia(solicitudes)
        todos_los_errores = errores_formato + errores_coherencia
        return {
            "valido": len(todos_los_errores) == 0,
            "errores": todos_los_errores,
            "conteo_peticiones": len(solicitudes) if isinstance(solicitudes, dict) else 0
        }

    def __call__(self, prompt_usuario: str):
        resultado = self.predictor(prompt_usuario=prompt_usuario)
        return resultado

# --- 3. LÓGICA DE OPTIMIZACIÓN (BOOTSTRAP FEW SHOTS) ---

def metrica_interpretador(example, prediction, trace=None):
    if not hasattr(prediction, "solicitudes_categorizadas") or not prediction.solicitudes_categorizadas:
        return False
    # Podrías agregar comparaciones con example.solicitudes_categorizadas
    return True

def cargar_fewshots_desde_json() -> List[dspy.Example]:
    ruta_json = os.path.join(os.path.dirname(__file__), 'FewShots_inferenciador.json')
    try:
        with open(ruta_json, 'r', encoding='utf-8') as f:
            ejemplos_raw = json.load(f)
            return [
                dspy.Example(**ej).with_inputs('prompt_usuario') 
                for ej in ejemplos_raw
            ]
    except Exception as e:
        print(f"⚠️ Error cargando fewshots: {e}")
        return []

TRAINSET = cargar_fewshots_desde_json()

def optimizar_agente(agente, trainset):
    if not trainset:
        print("⚠️ Trainset vacío.")
        return agente
    print(f"✨ Optimizando agente con BootstrapFewShot ({len(trainset)} ejemplos)...")
    teleprompter = BootstrapFewShot(metric=metrica_interpretador)
    agente_optimizado = teleprompter.compile(agente, trainset=trainset)
    return agente_optimizado

# --- 4. LÓGICA DE PROCESAMIENTO ---

def procesar_modelo(config_modelo, casos_evaluacion, system_summary, optimizado=False):
    print(f"\n--- 🤖 Procesando: {config_modelo['display']} (Optimizado: {optimizado}) ---")
    
    agente = InterpretadorAgente(escenarios=system_summary)
    if optimizado:
        agente = optimizar_agente(agente, TRAINSET)

    ruta_guardado = os.path.join(OUTPUT_DIR, f"respuestas_{config_modelo['display']}{'_fewshots' if optimizado else ''}.json")

    if os.path.exists(ruta_guardado):
        with open(ruta_guardado, 'r', encoding='utf-8') as f:
            resultados_guardados = json.load(f)
    else:
        resultados_guardados = {}

    for id_caso, datos_caso in casos_evaluacion.items():
        if id_caso in resultados_guardados: continue

        print(f"   ⏳ [{config_modelo['display']}] {id_caso}...")
        try:
            prediccion = agente(prompt_usuario=datos_caso.get("prompt_usuario", ""))
            resultados_guardados[id_caso] = {
                "solicitudes_categorizadas": getattr(prediccion, "solicitudes_categorizadas", {}),
                "notas": getattr(prediccion, "notas", "")
            }
            with open(ruta_guardado, 'w', encoding='utf-8') as f:
                json.dump(resultados_guardados, f, ensure_ascii=False, indent=2)
            time.sleep(1)
        except Exception as e:
            print(f"   ⚠️ ERROR en {id_caso}: {e}")

def main():
    # Cargar datasets (Placeholders - Ajusta según existan)
    try:
        # Ejemplo: Golden_questions_interpretador.json en Set_casos
        ruta_preguntas = os.path.join(CASOS_DIR, 'Golden_questions_interpretador.json')
        if os.path.exists(ruta_preguntas):
            with open(ruta_preguntas, 'r', encoding='utf-8') as f:
                casos_evaluacion = json.load(f)
        else:
            print("⚠️ No se encontró Golden_questions_interpretador.json en Set_casos. Usando diccionario vacío.")
            casos_evaluacion = {}

        # Escenarios (System Summary)
        ruta_escenarios = os.path.join(os.path.dirname(__file__), 'escenarios.json')
        if not os.path.exists(ruta_escenarios):
            ruta_alt = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
                                  'Agente_energetico', 'Sistema_entrada', 'Interpretador', 'escenarios.json')
            ruta_escenarios = ruta_alt if os.path.exists(ruta_alt) else ruta_escenarios
        
        with open(ruta_escenarios, 'r', encoding='utf-8') as f:
            system_summary = json.load(f)
    except Exception as e:
        print(f"❌ Error cargando datasets: {e}")
        return

    print("\n==============================================")
    print("🌟 EJECUTOR DE FEW-SHOTS PARA INFERENCIADOR")
    print("==============================================\n")
    
    # LISTA_MODELOS = [{"display": "Llama3.3-70b", "id": "..."}]
    print("⚠️ Configura el LM y el dataset de evaluación para ejecutar.")
    
if __name__ == "__main__":
    main()
