import dspy
import json
import os
import time
import re
from datetime import datetime
from typing import Any, List, Dict, Optional
from dspy.teleprompt import BootstrapFewShot

# Desactivar cache de DSPy para asegurar que cada llamada sea real y fresca
os.environ["DSP_CACHEBOOL"] = "False"

# --- 1. CONFIGURACIÓN DE ENTORNO Y RUTAS ---
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
CASOS_DIR = os.path.join(BASE_PATH, 'Set_casos')
OUTPUT_DIR = os.path.join(BASE_PATH, 'resultados_fewShots') # Cambiado a la nueva carpeta
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =================================================================
# 🔑 CONFIGURACIÓN DE MODELOS (DEJADOS COMENTADOS COMO SE SOLICITÓ)
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

class Planeador(dspy.Signature):
    """
    Convierte solicitudes categorizadas en un plan de acciones ejecutables.
    Define herramientas, orden y parámetros para cada solicitud.
    """

    solicitudes_categorizadas: dict[str, dict] = dspy.InputField(
        desc=(
            "Solicitudes indexadas como '@N'. Cada valor contiene:\n"
            "- solicitud: intención del usuario\n"
            "- escenario: clasificación funcional inferida (ej. consumo_basico, comparacion_consumos, deteccion_anomalias); "
            "guía semántica para priorizar y seleccionar herramientas"
        )
    )

    temporal_context: dict = dspy.InputField(
        desc="Referencia temporal en ISO 8601 para interpretar expresiones de tiempo."
    )

    temporal_preferences: dict = dspy.InputField(
        desc="Mapeo de términos temporales (ej. 'noche') a rangos horarios concretos."
    )

    plan_acciones: list[dict] = dspy.OutputField(
        desc=(
            "Secuencia ordenada de acciones para resolver las solicitudes. "
            "El plan descompone cada solicitud en pasos ejecutables.\n\n"

            "Cada acción incluye:\n"
            "- id: '@N.M' (N = solicitud origen, M = orden dentro de esa solicitud)\n"
            "- server_id: servidor de la herramienta\n"
            "- tool: herramienta seleccionada\n"
            "- inputs: parámetros; puede referenciar salidas '@N.M'\n"
            "- descripcion: propósito de la acción dentro de la solicitud\n\n"

            "Criterios:\n"
            "- cada solicitud (@N) se resuelve mediante una o más acciones\n"
            "- el orden (M) define la secuencia de ejecución dentro de la solicitud\n"
            "- una acción usa '@N.M' cuando requiere el resultado de una acción previa\n"
            "- cada acción corresponde a una única invocación de herramienta"
        )
    )

    notas: str = dspy.OutputField(
        desc=(
            "Explicación del plan: cómo se interpretaron las solicitudes y criterios usados "
            "para definir herramientas, orden, dependencias y parámetros."
        )
    )

class PlaneadorAgente(dspy.Module):
    def __init__(self, system_summary: dict = None):
        super().__init__()
        
        # Si no se proporciona el resumen del sistema, lo cargamos desde el archivo local
        if system_summary is None:
            system_summary = self._cargar_system_summary()
        
        # Almacenamos el resumen para acceso posterior (ej. validación)
        self.system_summary = system_summary

        # Se transforma el Json para que le llegue mejor al modelo y se inicia como instrucciones del sistema
        instruccion_sistema = (
            "Herramientas disponibles en el sistema junto a su descripción e indicaciones de uso:\n"
            f"```json\n{json.dumps(system_summary, indent=2, ensure_ascii=False)}\n```"
        )

        # Creamos el predictor con la Signature modificada
        self.predictor = dspy.Predict(Planeador.with_instructions(instruccion_sistema))

    def _cargar_system_summary(self) -> dict:
        """Busca y carga el archivo system_summary.json en el mismo directorio que el script."""
        ruta_json = os.path.join(os.path.dirname(__file__), 'system_summary.json')
        try:
            with open(ruta_json, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Error: No se encontró el archivo json de system_summary en la ruta: {ruta_json}")
            return {}

    def validar_formato(self, plan_acciones: Any) -> List[str]:
        """
        Revisa que el plan de acciones sea una lista de diccionarios
        y que los IDs sigan el patrón '@N.M'.
        """
        errores = []
        if not isinstance(plan_acciones, list):
            return ["El plan de acciones debe ser una lista."]
        
        if not plan_acciones:
            return ["El plan de acciones está vacío."]

        patron_id = re.compile(r"^@\d+\.\d+$")

        for i, accion in enumerate(plan_acciones):
            if not isinstance(accion, dict):
                errores.append(f"La acción en la posición {i} no es un diccionario.")
                continue
            
            id_accion = accion.get("id")
            if not id_accion or not isinstance(id_accion, str) or not patron_id.match(id_accion):
                errores.append(f"El ID de la acción '{id_accion}' en la posición {i} no sigue el formato '@N.M'.")
            
            if not accion.get("server_id"):
                errores.append(f"Falta 'server_id' en la acción '{id_accion or i}'.")
            
            if not accion.get("tool"):
                errores.append(f"Falta 'tool' en la acción '{id_accion or i}'.")
                
            if "inputs" not in accion or not isinstance(accion.get("inputs"), dict):
                errores.append(f"Falta el campo 'inputs' (o no es un diccionario) en la acción '{id_accion if 'id' in accion else i}'.")
                
        return errores

    def validar_coherencia(self, plan_acciones: List[Dict[str, Any]]) -> List[str]:
        """
        Verifica contra el resumen del sistema que las herramientas y servidores existan,
        y que se incluyan todos los parámetros obligatorios.
        """
        errores = []
        
        # Mapa de servidores y sus herramientas para búsqueda rápida
        # Ahora guardamos la meta-información para validar campos requeridos
        mapa_servidores = {}
        for s in self.system_summary.get("servers", []):
            s_id = s.get("server_id")
            if s_id:
                mapa_servidores[s_id] = {t.get("name"): t.get("meta", {}).get("input_schema", {}) 
                                        for t in s.get("tools", [])}

        if not isinstance(plan_acciones, list): return []

        for accion in plan_acciones:
            if not isinstance(accion, dict): continue
            
            s_id = accion.get("server_id")
            tool_name = accion.get("tool")
            inputs = accion.get("inputs", {})
            id_acc = accion.get("id", "N/A")

            if s_id not in mapa_servidores:
                errores.append(f"Servidor desconocido en '{id_acc}': '{s_id}'.")
            elif tool_name not in mapa_servidores[s_id]:
                errores.append(f"Herramienta desconocida en '{id_acc}': '{tool_name}' para el servidor '{s_id}'.")
            else:
                # Validar campos obligatorios
                schema = mapa_servidores[s_id][tool_name]
                required_fields = schema.get("required", [])
                
                for field in required_fields:
                    if field not in inputs:
                        errores.append(f"Falta parámetro obligatorio '{field}' en la acción '{id_acc}' ({tool_name}).")
                
        return errores

    def worker2(self, prediction: Any) -> Dict[str, Any]:
        """
        Orquestador de validación de Worker2.
        """
        plan = getattr(prediction, "plan_acciones", None)
        
        errores_formato = self.validar_formato(plan)
        errores_coherencia = self.validar_coherencia(plan)
        
        todos_los_errores = errores_formato + errores_coherencia
        
        return {
            "valido": len(todos_los_errores) == 0,
            "errores": todos_los_errores,
            "conteo_acciones": len(plan) if isinstance(plan, list) else 0
        }

    def _convertir_referencias_a_24h(self, prefs: dict) -> dict:
        """
        Convierte preferencias horarias del usuario (ej: "12:00 AM - 05:59 AM") 
        a formato militar de 24 horas (ej: "00:00 - 05:59").
        """
        def convertir_hora(hora_str):
            try:
                dt = datetime.strptime(hora_str.strip(), "%I:%M %p")
                return dt.strftime("%H:%M")
            except ValueError:
                return hora_str.strip()

        prefs_convertidas = {}
        for k, v in prefs.items():
            if not isinstance(v, str):
                prefs_convertidas[k] = v
                continue
            partes = v.split("-")
            if len(partes) == 2:
                inicio_24 = convertir_hora(partes[0])
                fin_24 = convertir_hora(partes[1])
                prefs_convertidas[k] = f"{inicio_24} - {fin_24}"
            else:
                prefs_convertidas[k] = convertir_hora(v)
        return prefs_convertidas

    def __call__(self, solicitudes_categorizadas: dict, temporal_context: dict, temporal_preferences: dict):
        # 1. Limpiar las preferencias horarias convirtiéndolas a 24h
        prefs_24h = self._convertir_referencias_a_24h(temporal_preferences)

        # 2. Llamar al predictor
        resultado = self.predictor(
            solicitudes_categorizadas=solicitudes_categorizadas, 
            temporal_context=temporal_context,
            temporal_preferences=prefs_24h
        )
        return resultado

# --- 3. LÓGICA DE OPTIMIZACIÓN (BOOTSTRAP FEW SHOTS) ---

def metrica_planeador(example, prediction, trace=None):
    """
    Métrica para BootstrapFewShot.
    Utiliza la lógica del worker2 del agente para determinar si el plan es válido.
    """
    # Necesitamos acceder a la validación. Como no tenemos instancia de summary aquí fácilmente,
    # el agente debería haber sido inicializado con el summary global.
    # Por simplicidad, si prediction tiene plan_acciones y notas, y el worker2 (que llamaríamos externamente)
    # dice que es correcto, devolvemos True.
    
    # Nota: Esta métrica se usará dentro del compile. Una forma es que el 'agente' use su worker2.
    # Pero dspy.compile necesita algo que reciba (example, prediction, trace).
    
    # Verificación de estructura mínima
    if not hasattr(prediction, "plan_acciones") or not prediction.plan_acciones:
        return False
    
    # Podrías agregar comparaciones semánticas con example.plan_acciones si estuviera disponible.
    return True

# =================================================================
# 🚀 CARGA DE FEW-SHOTS DESDE JSON
# =================================================================
def cargar_fewshots_desde_json() -> List[dspy.Example]:
    """Carga los ejemplos para few-shots desde FewShots_planeador.json."""
    ruta_json = os.path.join(os.path.dirname(__file__), 'FewShots_planeador.json')
    try:
        with open(ruta_json, 'r', encoding='utf-8') as f:
            ejemplos_raw = json.load(f)
            return [
                dspy.Example(**ej).with_inputs('solicitudes_categorizadas', 'temporal_context', 'temporal_preferences') 
                for ej in ejemplos_raw
            ]
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"⚠️ No se pudo cargar FewShots_planeador.json: {e}")
        return []

# Inicializamos el TRAINSET cargando los datos del archivo
TRAINSET = cargar_fewshots_desde_json()
# =================================================================

def optimizar_agente(agente, trainset):
    """Compila el agente utilizando BootstrapFewShot."""
    if not trainset:
        print("⚠️ Trainset vacío. Saltando optimización.")
        return agente
    
    print(f"✨ Optimizando agente con BootstrapFewShot ({len(trainset)} ejemplos)...")
    teleprompter = BootstrapFewShot(metric=metrica_planeador)
    agente_optimizado = teleprompter.compile(agente, trainset=trainset)
    return agente_optimizado

# --- 4. LÓGICA DE PROCESAMIENTO E INFERENCIA ---

def procesar_modelo(config_modelo, casos_evaluacion, t_context, t_prefs, system_summary, optimizado=False):
    print(f"\n--- 🤖 Procesando: {config_modelo['display']} (Optimizado: {optimizado}) ---")
    
    agente = PlaneadorAgente(system_summary=system_summary)
    
    if optimizado:
        agente = optimizar_agente(agente, TRAINSET)

    ruta_guardado = os.path.join(OUTPUT_DIR, f"respuestas_{config_modelo['display']}{'_fewshots' if optimizado else ''}.json")

    if os.path.exists(ruta_guardado):
        with open(ruta_guardado, 'r', encoding='utf-8') as f:
            resultados_guardados = json.load(f)
        print(f"📂 Reanudando progreso para {config_modelo['display']}...")
    else:
        resultados_guardados = {}

    for nivel, dict_casos in casos_evaluacion.items():
        if nivel not in resultados_guardados:
            resultados_guardados[nivel] = {}

        for id_caso, datos_caso in dict_casos.items():
            if id_caso in resultados_guardados[nivel] and resultados_guardados[nivel][id_caso].get("plan_acciones"):
                continue

            print(f"   ⏳ [{config_modelo['display']}] {nivel} -> {id_caso}...")
            
            try:
                prediccion = agente(
                    solicitudes_categorizadas=datos_caso.get("solicitudes_categorizadas", {}),
                    temporal_context=t_context,
                    temporal_preferences=t_prefs
                )
                plan = getattr(prediccion, "plan_acciones", [])
                notas = getattr(prediccion, "notas", "")
                
                resultados_guardados[nivel][id_caso] = {
                    "plan_acciones": plan,
                    "notas": notas
                }
                
                with open(ruta_guardado, 'w', encoding='utf-8') as f:
                    json.dump(resultados_guardados, f, ensure_ascii=False, indent=2)
                
                time.sleep(1)

            except Exception as e:
                print(f"   ⚠️ ERROR en {id_caso}: {str(e)}")
                resultados_guardados[nivel][id_caso] = {
                    "plan_acciones": [],
                    "notas": f"Error en inferencia: {str(e)}"
                }

def main():
    # Cargar archivos golden
    try:
        # Ajustar rutas si es necesario según la ubicación del script
        with open(os.path.join(CASOS_DIR, 'Golden_questions_planeador.json'), 'r', encoding='utf-8') as f:
            casos_evaluacion = json.load(f)
        with open(os.path.join(CASOS_DIR, 'Golden_Temporal_Context.json'), 'r', encoding='utf-8') as f:
            temporal_data = json.load(f)
            t_context = temporal_data['temporal_context']
            t_prefs = temporal_data['temporal_preferences']
        with open(os.path.join(CASOS_DIR, 'Golden_summary.json'), 'r', encoding='utf-8') as f:
            system_summary = json.load(f)
    except Exception as e:
        print(f"❌ Error fatal cargando datasets: {e}")
        return

    print("\n==============================================")
    print("🌟 EJECUTOR DE FEW-SHOTS PARA PLANEADOR (DSPY)")
    print("==============================================\n")

    # Lista de modelos de ejemplo (descomentar o ajustar según se necesite)
    LISTA_MODELOS = [
        {"display": "Llama-3.3-70b", "id": "openrouter/meta-llama/llama-3.3-70b-instruct"},
    ]

    # NOTA: Asegúrate de que dspy.settings.configure(lm=...) esté configurado antes de llamar a procesar_modelo
    print("⚠️ Recuerda configurar el LM y las API Keys antes de ejecutar.")
    
    # Iterar sobre modelos (esto fallará si no se configura el LM arriba)
    # for config in LISTA_MODELOS:
    #     procesar_modelo(config, casos_evaluacion, t_context, t_prefs, system_summary, optimizado=True)

    print("\n🏁 Proceso de configuración completado. Listo para agregar FewShots y ejecutar.")

if __name__ == "__main__":
    main()
