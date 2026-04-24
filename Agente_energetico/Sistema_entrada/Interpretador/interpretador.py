import dspy
import os
import json
from typing import Dict, Any, List


# -------------------------------------------------------------------------
# Definición de la Signature
# -------------------------------------------------------------------------

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

# -------------------------------------------------------------------------
# Agente Interpretador
# -------------------------------------------------------------------------

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
        ruta_json = os.path.join(os.path.dirname(__file__), 'escenarios.json')
        try:
            with open(ruta_json, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Error: No se encontró el archivo json de escenarios en la ruta: {ruta_json}")
            return {}


    def validar_formato(self, solicitudes_categorizadas: Any) -> List[str]:
        """
        Revisa que la estructura cumpla con el formato establecido en la Signature.
        Retorna una lista con los errores encontrados.
        """
        errores = []
        
        if not isinstance(solicitudes_categorizadas, dict):
            return ["La salida de solicitudes categorizadas debe ser un diccionario."]
            
        if not solicitudes_categorizadas:
            return ["No se encontraron solicitudes en la salida."]

        for key, val in solicitudes_categorizadas.items():
            # Validar formato de la clave @N
            if not (isinstance(key, str) and key.startswith("@")):
                errores.append(f"Clave de petición con formato incorrecto: '{key}'. Debe comenzar con '@'.")
            
            # Validar estructura interna de cada petición
            if not isinstance(val, dict):
                errores.append(f"El valor de la petición '{key}' debe ser un diccionario.")
                continue
                
            if "solicitud" not in val or not val["solicitud"]:
                errores.append(f"Falta el campo 'solicitud' o está vacío en '{key}'.")
            
            if "escenario" not in val or not val["escenario"]:
                errores.append(f"Falta el campo 'escenario' o está vacío en '{key}'.")
                
        return errores

    def validar_coherencia(self, solicitudes_categorizadas: Dict[str, Any]) -> List[str]:
        """
        Revisa que los escenarios categorizados existan en el sistema.
        Retorna una lista con los errores encontrados.
        """
        errores = []
        escenarios_validos = self.escenarios
        
        if not isinstance(solicitudes_categorizadas, dict):
            return [] # El error de tipo lo maneja validar_formato

        for key, val in solicitudes_categorizadas.items():
            if not isinstance(val, dict): continue
            
            escenario_nombre = val.get("escenario")
            if escenario_nombre and escenario_nombre not in escenarios_validos:
                errores.append(f"Escenario desconocido en '{key}': '{escenario_nombre}'. No figura en la lista permitida.")
                
        return errores

    def worker1(self, prediction: Any) -> Dict[str, Any]:
        """
        Orquestador de validación de Worker1. Gestiona formato y coherencia.
        """
        # Extraer datos de la predicción de dspy (objeto Prediction)
        solicitudes = getattr(prediction, "solicitudes_categorizadas", None)
        notas = getattr(prediction, "notas", "")
        
        errores_formato = self.validar_formato(solicitudes)
        errores_coherencia = self.validar_coherencia(solicitudes)
        
        todos_los_errores = errores_formato + errores_coherencia
        
        return {
            "valido": len(todos_los_errores) == 0,
            "errores": todos_los_errores,
            "conteo_peticiones": len(solicitudes) if isinstance(solicitudes, dict) else 0
        }

    def _cargar_fewshots(self) -> List[dspy.Example]:
        """Carga los ejemplos para few-shots desde FewShots_interpretador.json."""
        ruta_json = os.path.join(os.path.dirname(__file__), 'FewShots_interpretador.json')
        try:
            with open(ruta_json, 'r', encoding='utf-8') as f:
                ejemplos_raw = json.load(f)
                return [dspy.Example(**ej).with_inputs('prompt_usuario') for ej in ejemplos_raw]
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

    def __call__(self, prompt_usuario: str):
        # El __call__ es para poder llamar al agente como si fuera una función.
        # Devuelve el resultado directo del predictor (objeto Prediction de dspy)
        resultado = self.predictor(prompt_usuario=prompt_usuario)
        
        print(f"\n📊 [InterpretadorAgente] OUTPUT DIRECTO PRE-WORKER:")
        print(f"Solicitudes: {getattr(resultado, 'solicitudes_categorizadas', 'N/A')}\n")
        
        return resultado

# -------------------------------------------------------------------------
# Inicialización y Carga de Datos
# -------------------------------------------------------------------------

# Cargar API Keys
""" env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
load_dotenv(env_path) """

if __name__ == "__main__":

    # Configuración del modelo Gemini
    gemini_model = dspy.LM(model='gemini/gemini-2.5-flash', api_key='')
    dspy.configure(lm=gemini_model)

    # Ejemplo de uso: El agente ahora es autogestionado
    agente = InterpretadorAgente()
    
    # Simulación de prompt
    test_prompt = "Quiero ver cuánto consumió la nevera ayer y compararlo con el lunes."
    resultado = agente(test_prompt)
    print("--- RESULTADO DEL INTERPRETADOR ---")
    print(f"Solicitudes: {resultado.solicitudes_categorizadas}")
    print(f"Notas: {resultado.notas}")

    # Validación manual (Nuevo Worker 1)
    print("\n--- REPORTE DE VALIDACIÓN (WORKER 1) ---")
    reporte = agente.worker1(resultado)
    print(json.dumps(reporte, indent=2, ensure_ascii=False))


"""
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

notas='La solicitud del usuario se ha dividido en dos partes para asegurar que cada una sea autocontenida y específica.\nLa primera parte, "Quiero ver cuánto consumió la nevera ayer", se categoriza como \'consumo_basico\' porque solicita un valor cuantitativo de consumo energético para un dispositivo específico ("la nevera") en un periodo de tiempo definido ("ayer").\nLa segunda parte, "y compararlo con el lunes", se categoriza como \'comparacion_consumos\' porque la intención principal es comparar el consumo de la nevera de ayer con el consumo de la nevera del lunes, lo cual implica una comparación explícita entre dos periodos para el mismo dispositivo. Se ha detallado la solicitud para que sea completamente autocontenida.'

"""
