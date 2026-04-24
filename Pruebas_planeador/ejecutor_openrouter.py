import dspy
import json
import os
import time
from datetime import datetime
from typing import Any, List, Dict

# Desactivar cache de DSPy para asegurar que cada llamada sea real y fresca
os.environ["DSP_CACHEBOOL"] = "False"

# --- 1. CONFIGURACIÓN DE ENTORNO Y RUTAS ---
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
CASOS_DIR = os.path.join(BASE_PATH, 'Set_casos')
OUTPUT_DIR = os.path.join(BASE_PATH, 'Resultados')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =================================================================
# 🔑 COLOCA AQUÍ TU API KEY DE OPENROUTER
# =================================================================
APIKEY_OPENROUTER = ""
# =================================================================

# Definición de modelos para iterar
# Definición de modelos para iterar (Versiones de Pago para estabilidad)
LISTA_MODELOS = [
    {"display": "Llama-3.3-70b", "id": "openrouter/meta-llama/llama-3.3-70b-instruct"},
    {"display": "Mistral-Dev_Devstral", "id": "openrouter/mistralai/devstral-2512"},
    {"display": "Xiaomi-MimoV2", "id": "openrouter/xiaomi/mimo-v2-flash"},
    {"display": "Qwen3-Coder", "id": "openrouter/qwen/qwen3-coder"},
    {"display": "DeepSeek-R1-Chimera", "id": "openrouter/tngtech/deepseek-r1t2-chimera"}
]

# --- 2. DEFINICIÓN DE SIGNATURES Y AGENTE ---
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

        for i, accion in enumerate(plan_acciones):
            if not isinstance(accion, dict):
                errores.append(f"La acción en la posición {i} no es un diccionario.")
                continue
            
            id_accion = accion.get("id")
            if not id_accion or not str(id_accion).startswith("@") or "." not in str(id_accion):
                errores.append(f"El ID de la acción '{id_accion}' en la posición {i} no sigue el formato '@N.M'.")
            
            if not accion.get("server_id"):
                errores.append(f"Falta 'server_id' en la acción '{id_accion or i}'.")
            
            if not accion.get("tool"):
                errores.append(f"Falta 'tool' en la acción '{id_accion or i}'.")
                
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
        notas = getattr(prediction, "notas", "")
        
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
                # Intenta parsear como "04:30 PM"
                dt = datetime.strptime(hora_str.strip(), "%I:%M %p")
                return dt.strftime("%H:%M")
            except ValueError:
                # Si falla o no tiene AM/PM, lo devuelve como estaba
                return hora_str.strip()

        prefs_convertidas = {}
        for k, v in prefs.items():
            if not isinstance(v, str):
                prefs_convertidas[k] = v
                continue
            
            # Buscamos si es un rango "Inicio - Fin"
            partes = v.split("-")
            if len(partes) == 2:
                inicio_24 = convertir_hora(partes[0])
                fin_24 = convertir_hora(partes[1])
                prefs_convertidas[k] = f"{inicio_24} - {fin_24}"
            else:
                # Si es un valor único
                prefs_convertidas[k] = convertir_hora(v)
                
        return prefs_convertidas

    def _cargar_fewshots(self) -> List[dspy.Example]:
        """Carga los ejemplos para few-shots desde FewShots_planeador.json."""
        ruta_json = os.path.join(os.path.dirname(__file__), 'FewShots_planeador.json')
        try:
            with open(ruta_json, 'r', encoding='utf-8') as f:
                ejemplos_raw = json.load(f)
                return [dspy.Example(**ej).with_inputs('solicitudes_categorizadas', 'temporal_context', 'temporal_preferences') for ej in ejemplos_raw]
        except (FileNotFoundError, json.JSONDecodeError):
            return []

    def entrenar_con_fewshots(self):
        """
        Función para entrenar/compilar el predictor usando BootstrapFewShot.
        No se llama automáticamente por defecto.
        """
        from dspy.teleprompt import BootstrapFewShot
        
        trainset = self._cargar_fewshots()
        if not trainset:
            return

        trainer = BootstrapFewShot()
        compiled_predictor = trainer.compile(
            student=self.predictor,
            trainset=trainset
        )
        self.predictor = compiled_predictor

    def __call__(self, solicitudes_categorizadas: dict, temporal_context: dict, temporal_preferences: dict):
        # 1. Limpiar las preferencias horarias convirtiéndolas a 24h
        prefs_24h = self._convertir_referencias_a_24h(temporal_preferences)

        # 2. Llamar al predictor (objeto Prediction de dspy)
        resultado = self.predictor(
            solicitudes_categorizadas=solicitudes_categorizadas, 
            temporal_context=temporal_context,
            temporal_preferences=prefs_24h
        )
        
        print(f"\n📊 [PlaneadorAgente] OUTPUT DIRECTO PRE-WORKER:")
        print(f"Plan de Acciones: {getattr(resultado, 'plan_acciones', 'N/A')}\n")
        
        return resultado

# --- 3. LÓGICA DE PROCESAMIENTO ---
def procesar_modelo(config_modelo, casos_evaluacion, temporal_context, temporal_prefs, system_summary):
    print(f"\n--- 🤖 Invocando Modelo: {config_modelo['display']} ---")
    
    # Configurar el LM de DSPy para este modelo específico
    lm = dspy.LM(
        model=config_modelo['id'],
        api_base="https://openrouter.ai/api/v1",
        api_key=APIKEY_OPENROUTER,
        max_tokens=4000 # Ajustado para evitar exceder límites de salida comunes en free
    )
    dspy.settings.configure(lm=lm)
    
    agente = PlaneadorAgente(system_summary=system_summary)
    ruta_guardado = os.path.join(OUTPUT_DIR, f"respuestas_{config_modelo['display']}.json")

    # Reanudar si existe archivo
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
            # Saltar si ya existe el resultado y tiene plan
            if id_caso in resultados_guardados[nivel] and resultados_guardados[nivel][id_caso].get("plan_acciones"):
                continue

            print(f"   ⏳ [{config_modelo['display']}] {nivel} -> {id_caso}...")
            
            try:
                prediccion = agente(
                    solicitudes_categorizadas=datos_caso.get("solicitudes_categorizadas", {}),
                    temporal_context=temporal_context,
                    temporal_preferences=temporal_prefs
                )
                plan = getattr(prediccion, "plan_acciones", [])
                notas = getattr(prediccion, "notas", "")
                
                resultados_guardados[nivel][id_caso] = {
                    "plan_acciones": plan,
                    "notas": notas
                }
                
                # Guardar inmediatamente tras éxito (Checkpoint)
                with open(ruta_guardado, 'w', encoding='utf-8') as f:
                    json.dump(resultados_guardados, f, ensure_ascii=False, indent=2)
                
                # Pequeño delay opcional para no saturar rate limits agresivos
                time.sleep(1)

            except Exception as e:
                # Si es un error de cuota o rate limit, registramos el error y decidimos si continuar con el siguiente modelo
                msg_error = str(e)
                print(f"   ⚠️ ERROR en {id_caso}: {msg_error}")
                
                # ERRORES CRÍTICOS QUE DEBEN SALTAR EL MODELO COMPLETO
                errores_fatales = ["429", "404", "401", "quota", "limit", "not found", "period has ended"]
                
                if any(err in msg_error.lower() for err in errores_fatales):
                    print(f"   🚨 ERROR FATAL/LÍMITE para {config_modelo['display']}. Saltando al siguiente modelo...")
                    return # Salir de este modelo y pasar al siguiente en el loop principal
                
                # Para otros errores, guardamos la traza y seguimos intentando el siguiente caso
                resultados_guardados[nivel][id_caso] = {
                    "plan_acciones": [],
                    "notas": f"Error en inferencia: {msg_error}"
                }

    print(f"✅ Finalizada la carga para {config_modelo['display']}")

def main():
    # Cargar archivos golden
    try:
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
    print("🌟 INICIADOR DE INFERENCIAS MULTI-MODELO (OPENROUTER)")
    print("==============================================\n")

    for config in LISTA_MODELOS:
        if APIKEY_OPENROUTER == "TU_API_KEY_AQUI":
            print("🛑 Error: No has configurado la APIKEY_OPENROUTER en el script.")
            return
            
        procesar_modelo(config, casos_evaluacion, t_context, t_prefs, system_summary)

    print("\n🏁 Proceso de inferencia completado para todos los modelos.")

if __name__ == "__main__":
    main()
