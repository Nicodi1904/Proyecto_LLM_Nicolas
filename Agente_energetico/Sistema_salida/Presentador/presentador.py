import dspy
import os
import json
from dotenv import load_dotenv

# Cargar variables de entorno desde archivo .env (ahora a dos niveles de profundidad)
env_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '.env'))
load_dotenv(dotenv_path=env_path)

# -------------------------------------------------------------------------
# Definición de Signature: Presentador
# -------------------------------------------------------------------------

class Presentador(dspy.Signature):
    """
    Módulo de razonamiento final que genera una respuesta estructurada para el usuario.
    Toma todo el contexto del proceso (notas de inferencia y planificación) junto con los 
    resultados crudos de herramientas, y lo sintetiza en un informe claro, analítico 
    y con formato específico.
    """

    notas_inferenciador: str = dspy.InputField(
        desc=("Razonamiento de categorización de la petición del usuario.")
    )

    notas_planeador: str = dspy.InputField(
        desc=("Razonamiento sobre cómo se construyó el plan de acción para resolver la petición.")
    )

    Informe_LLM: dict = dspy.InputField(
        desc=(
            "Reporte de ejecución de herramientas desde el servidor MCP. "
            "Contiene las acciones realizadas, sus herramientas correspondientes y los datos "
            "obtenidos o errores generados."
        )
    )

    Formato_respuesta: str = dspy.InputField(
        desc="Instrucciones específicas sobre cómo el usuario desea que se formatee o estructure la respuesta final."
    )

    Resumen_op: str = dspy.OutputField(
        desc="Un resumen narrativo y claro de las operaciones que el sistema realizó para intentar cumplir la solicitud."
    )
    
    Resultados_op: str = dspy.OutputField(
        desc="Presentación literal o tabulada de los datos, mediciones o hallazgos clave extraídos del Informe_LLM."
    )
    
    analisis: str = dspy.OutputField(
        desc="Interpretación de los resultados. ¿Qué significan los datos en el contexto de la solicitud original?"
    )
    
    sugerencia: str = dspy.OutputField(
        desc="Recomendaciones energéticas, de uso de dispositivos o siguientes pasos basados en el análisis."
    )



# -------------------------------------------------------------------------
# Agente Presentador
# -------------------------------------------------------------------------

class PresentadorAgente(dspy.Module):
    def __init__(self):
        super().__init__()
        # Inicializamos el predictor directamente con la signature
        self.predictor = dspy.Predict(Presentador)
        # Cargar el formato de respuesta de manera consistente
        self.formato_respuesta = self._cargar_formato_respuesta()

    def _cargar_formato_respuesta(self) -> str:
        """
        Carga las instrucciones de formato desde el archivo formato_respuesta.json
        ubicado en el mismo directorio que este script.
        """
        formato_path = os.path.join(os.path.dirname(__file__), "formato_respuesta.json")
        try:
            with open(formato_path, 'r', encoding='utf-8') as f:
                return json.dumps(json.load(f), ensure_ascii=False)
        except FileNotFoundError:
            print(f"Error: No se encontró el archivo de formato en {formato_path}")
            return "Responde de forma clara y concisa."
        except json.JSONDecodeError:
            print(f"Error: El archivo {formato_path} no tiene un formato JSON válido.")
            return "Responde de forma clara y concisa."
        except Exception as e:
            print(f"Error inesperado al cargar formato_respuesta.json: {e}")
            return "Responde de forma clara y concisa."

    def __call__(self, notas_inferenciador: str, notas_planeador: str, Informe_LLM: dict):
        # Ejecuta la predicción inyectando el formato del objeto
        return self.predictor(
            notas_inferenciador=notas_inferenciador,
            notas_planeador=notas_planeador,
            Informe_LLM=Informe_LLM,
            Formato_respuesta=self.formato_respuesta
        )

# -------------------------------------------------------------------------
# Ejecución de Prueba
# -------------------------------------------------------------------------

if __name__ == "__main__":
    import json

    # Configuración del modelo Gemini
    gemini_model = dspy.LM(model='gemini/gemini-2.5-flash', api_key='AIzaSyCDi0fhhRVNPVokMOvd5T1Dg9TStT4oD9U')
    dspy.configure(lm=gemini_model)

    # Datos de prueba simulados
    notas_inf='La solicitud del usuario se ha dividido en dos partes para asegurar que cada una sea autocontenida y específica.\nLa primera parte, "Quiero ver cuánto consumió la nevera ayer", se categoriza como \'consumo_basico\' porque solicita un valor cuantitativo de consumo energético para un dispositivo específico ("la nevera") en un periodo de tiempo definido ("ayer").\nLa segunda parte, "y compararlo con el lunes", se categoriza como \'comparacion_consumos\' porque la intención principal es comparar el consumo de la nevera de ayer con el consumo de la nevera del lunes, lo cual implica una comparación explícita entre dos periodos para el mismo dispositivo. Se ha detallado la solicitud para que sea completamente autocontenida.'
    notas_plan="Para la solicitud '@1', se utiliza la herramienta `obtener_consumo` para recuperar el consumo de la nevera del día de ayer (23 de octubre de 2024). Se elige una granularidad por 'hora' para proporcionar una vista detallada de la evolución del consumo a lo largo del día.\n\nPara la solicitud '@2', se emplea la herramienta `analizar_comparacion` para contrastar el consumo de la nevera de ayer (23 de octubre de 2024) con el consumo de la nevera del lunes (21 de octubre de 2024). Esta herramienta es adecuada para realizar comparaciones directas entre dos periodos de consumo acumulado."

    informe_test ={
  "@1": [
    {
      "accion_id": "@1.1",
      "tool": "obtener_consumo",
      "descripcion": "Obtener el consumo energético horario de la nevera para el día de ayer (23 de octubre de 2024).",
      "resultado": {
        "status": "success",
        "periodo": {
          "inicio": "2024-10-23 00:00:00",
          "fin": "2024-10-23 23:59:59"
        },
        "granularidad": "hora",
        "datos": {
          "PC": {
            "2024-10-23T00:00:00": 0.0,
            "2024-10-23T01:00:00": 0.0001,
            ". . .": ". . .",
            "2024-10-23T22:00:00": 0.0,
            "2024-10-23T23:00:00": 0.0
          }
        }
      },
      "error": None
    }
  ],
  "@2": [
    {
      "accion_id": "@2.1",
      "tool": "analizar_comparacion",
      "descripcion": "Comparar el consumo total de la nevera del día de ayer (23 de octubre de 2024) con el consumo total de la nevera del lunes (21 de octubre de 2024).",
      "resultado": {
        "status": "success",
        "comparacion": {
          "valor_a": 0.0118,
          "valor_b": 0.0024000000000000002,
          "diferencia_absoluta": 0.009399999999999999,
          "diferencia_porcentual": 391.67,
          "mayor_consumo": "A"
        }
      },
      "error": None
    }
  ]
}
    
    # Inicializar el agente
    agente = PresentadorAgente()

    # Ejecutar la simulación
    print("Generando respuesta del Presentador...")
    resultado = agente(
        notas_inferenciador=notas_inf,
        notas_planeador=notas_plan,
        Informe_LLM=informe_test
    )

    # Mostrar salidas
    print("\n--- RESUMEN OP ---")
    print(resultado.Resumen_op)
    
    print("\n--- RESULTADOS OP ---")
    print(resultado.Resultados_op)
    
    print("\n--- ANÁLISIS ---")
    print(resultado.analisis)
    
    print("\n--- SUGERENCIA ---")
    print(resultado.sugerencia)
