import dspy
import os
import json
from typing import Dict, Any, List, Optional
from datetime import datetime
from dotenv import load_dotenv

# Cargar API Keys
env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
load_dotenv(env_path)

# -------------------------------------------------------------------------
# Definición de Signatures
# -------------------------------------------------------------------------

class Planeador(dspy.Signature):
    """
    Su función es traducir las solicitudes categorizadas por el inferenciador
    en un conjunto estructurado de acciones con las herramientas disponibles del sistema.
    Decide qué herramientas utilizar, en qué orden y con qué parámetros,
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

    temporal_context: dict = dspy.InputField(
        desc=(
            "Contexto temporal (formato ISO 8601) del momento en que fue formulada la solicitud, "
            "utilizado como referencia para interpretar expresiones temporales presentes en las solicitudes del usuario. "
        )
    )

    temporal_preferences: dict = dspy.InputField(
        desc=(
            "Preferencias horarias del usuario. Es un diccionario que mapea conceptos abstractos (ej. 'madrugada', 'noche') "
            "a horas específicas. Utiliza esta información para traducir términos vagos en horas de inicio y fin claras."
        )
    )

    plan_acciones: list[dict] = dspy.OutputField(
        desc=(
            "Lista estructurada de acciones planificadas para resolver las solicitudes del usuario. "
            "Cada elemento de la lista representa una acción individual e incluye:\n"
            "- 'id' (string): identificador único de la acción, con el formato '@N.M', donde '@' es un símbolo diferenciador, 'N' "
            "corresponde al número de la solicitud de origen y 'M' indica el orden secuencial (1,2,3...) en que se realizarán las acciones.\n"
            "- 'server_id' (string): identificador del servidor donde se encuentra la herramienta a invocar.\n"
            "- 'tool' (string): nombre de la herramienta seleccionada.\n"
            "- 'inputs' (dict): parámetros de entrada de la herramienta, pueden incluir referencias a salidas de acciones previas mediante identificadores '@N.M'.\n"
            "- 'descripcion' (string): breve descripción semántica de la acción y su propósito dentro del plan."
        )
    )

    notas: str = dspy.OutputField(
        desc=(
            "Razonamiento y breve explicación de cómo se planificó cada acción."
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




if __name__ == "__main__":
    
    # Configuración del modelo Gemini
    gemini_model = dspy.LM(model='gemini/gemini-2.5-flash', api_key='AIzaSyAzFKB2KaM-NugOu2SkkQKqT9T8IHflrvk')
    dspy.configure(lm=gemini_model)

    # Datos de prueba
    solicitudes_categorizadas = {
        '@1': {
            'solicitud': 'Mostrar el consumo energético de la nevera para el día de ayer.', 
            'escenario': 'consumo_basico'
        }, 
        '@2': {
            'solicitud': 'Comparar el consumo energético de la nevera de ayer con el consumo energético de la nevera del lunes.', 
            'escenario': 'comparacion_consumos'
        }
    }
    
    # Contexto temporal en 2024 (formato ISO 8601)
    temporal_context = {
        "fecha_actual": "2024-10-24T14:30:00"
    }

    # Preferencias horarias (Dummy basado en UI RightBar)
    temporal_preferences = {
        "madrugada": "12:00 AM - 05:59 AM",
        "mañana": "06:00 AM - 11:59 AM",
        "tarde": "12:00 PM - 03:59 PM",
        "media tarde": "04:00 PM - 06:59 PM",
        "noche": "07:00 PM - 08:59 PM",
        "media noche": "09:00 PM - 11:59 PM"
    }

    # Inicialización del agente
    agente = PlaneadorAgente()
    
    # Demostración del parseo interno
    prefs_24h = agente._convertir_referencias_a_24h(temporal_preferences)
    print("--- PREFERENCIAS HORARIAS DUMMY (ENTRADA UI vs CONVERSIÓN INTERNA) ---")
    print(f"UI Original: {json.dumps(temporal_preferences, indent=2, ensure_ascii=False)}")
    print(f"Convertidas a 24H: {json.dumps(prefs_24h, indent=2, ensure_ascii=False)}")
    print("-" * 50 + "\n")

    # Ejecución de prueba
    resultado = agente(
        solicitudes_categorizadas=solicitudes_categorizadas, 
        temporal_context=temporal_context,
        temporal_preferences=temporal_preferences
    )
    
    print("--- RESULTADO DEL PLANIFICADOR ---")
    print(f"Plan: {resultado.plan_acciones}")
    print(f"Notas: {resultado.notas}")

    # Validación manual (Nuevo Worker 2)
    print("\n--- REPORTE DE VALIDACIÓN (WORKER 2) ---")
    reporte = agente.worker2(resultado)
    print(json.dumps(reporte, indent=2, ensure_ascii=False))
    




"""
plan_acciones=[{'id': '@1.1', 'server_id': 'mcp_server_gravity', 'tool': 'obtener_consumo', 'inputs': {'dispositivos': ['nevera'], 'fecha_inicio': '2024-10-23T00:00:00', 'fecha_fin': '2024-10-23T23:59:59', 'granularidad': 'hora'}, 'descripcion': 'Obtener el consumo energético horario de la nevera para el día de ayer (23 de octubre de 2024).'}, {'id': '@2.1', 'server_id': 'mcp_server_gravity', 'tool': 'analizar_comparacion', 'inputs': {'objetivo_a': {'dispositivo': 'nevera', 'fecha_inicio': '2024-10-23T00:00:00', 'fecha_fin': '2024-10-23T23:59:59'}, 'objetivo_b': {'dispositivo': 'nevera', 'fecha_inicio': '2024-10-21T00:00:00', 'fecha_fin': '2024-10-21T23:59:59'}}, 'descripcion': 'Comparar el consumo total de la nevera del día de ayer (23 de octubre de 2024) con el consumo total de la nevera del lunes (21 de octubre de 2024).'}]
notas="Para la solicitud '@1', se utiliza la herramienta `obtener_consumo` para recuperar el consumo de la nevera del día de ayer (23 de octubre de 2024). Se elige una granularidad por 'hora' para proporcionar una vista detallada de la evolución del consumo a lo largo del día.\n\nPara la solicitud '@2', se emplea la herramienta `analizar_comparacion` para contrastar el consumo de la nevera de ayer (23 de octubre de 2024) con el consumo de la nevera del lunes (21 de octubre de 2024). Esta herramienta es adecuada para realizar comparaciones directas entre dos periodos de consumo acumulado."
"""