import os
import json
from dotenv import load_dotenv

# Asegúrate de instalar liteLLM y deepeval
# pip install deepeval litellm python-dotenv
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.models import GPTModel

# Fallback por si la versión de deepeval no trae Rubric de forma nativa ahí mismo
try:
    from deepeval.metrics import Rubric
except ImportError:
    class Rubric:
        def __init__(self, score_range, expected_outcome):
            self.score_range = score_range
            self.expected_outcome = expected_outcome
        def __str__(self):
            return f"Score {self.score_range[0]}-{self.score_range[1]}: {self.expected_outcome}"


# Cargar variables de entorno si existe un .env en la raíz o en esta carpeta
load_dotenv()
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))

# Usa MINA_API_KEY o GEMINI_API_KEY si estás usando gemini vía la API OpenAI-compatible
#API_KEY = os.environ.get("MINA_API_KEY") or os.environ.get("GEMINI_API_KEY") or os.environ.get("OPENAI_API_KEY")
API_KEY = "AIzaSyCxejQhlCY9S5E7YEQb4zgMyr3bNHS3Aes"


if not API_KEY:
    print("⚠️ ADVERTENCIA: No se encontró MINA_API_KEY, GEMINI_API_KEY ni OPENAI_API_KEY en el entorno.")
    print("Asegúrate de configurar la variable de entorno antes de ejecutar.")

print("🤖 Conectando Juez a Gemini-2.5-Flash a través de la API OpenAI-Compatible...")
model_juez = GPTModel(
    model="gemini-2.5-flash",
    api_key=API_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)


CRITERIOS_PLANEADOR = f"""
Evalúa el desempeño del modelo planeador en sus funciones principales:
1) Generar un plan de acciones correcto a partir de las solicitudes categorizadas.
2) Seleccionar correctamente las herramientas disponibles.
3) Definir correctamente parámetros, orden y dependencias entre acciones.
4) Mantener coherencia interna entre el razonamiento del modelo (campo de notas) y la salida generada.


Contexto disponible para el modelo:
- SYSTEM_SUMMARY: descripción de herramientas, parámetros y propósito.
- TEMPORAL_CONTEXT: reglas para la interpretación de expresiones temporales.


CRITERIOS A CALIFICAR:

A1. Traducción correcta de solicitudes a acciones:
Cada solicitud categorizada es representada mediante al menos una acción
dentro del plan.

A2. Selección correcta de herramientas:
El modelo utiliza herramientas válidas y adecuadas según el "SYSTEM_SUMMARY".

A3. Parametrización correcta:
Los parámetros definidos en cada acción son consistentes con la solicitud original,
el contexto temporal y las especificaciones del sistema.

A4. Interpretación correcta del contexto temporal:
El modelo resuelve correctamente expresiones temporales implícitas o relativas
utilizando el "TEMPORAL_CONTEXT".

A5. Coherencia interna:
El razonamiento en el campo de "notas" es claro, consistente y alineado con
las decisiones tomadas en el plan.

A6. Orden y dependencias correctas:
Las acciones están organizadas en una secuencia lógica, sin inconsistencias,
y respetan posibles dependencias entre pasos.
"""


RUBRICA_PLANEADOR = [
    Rubric(
        score_range=(0, 4),
        expected_outcome=(
            "El modelo genera un plan incompleto o incorrecto, "
            "utiliza herramientas no válidas o inventa parámetros, "
            "presenta una interpretación deficiente del contexto temporal "
            "y muestra incoherencias significativas entre el razonamiento y la salida. "
            "La calidad semántica y funcional (criterios A1-A6) es insuficiente."
        )
    ),
    Rubric(
        score_range=(5, 7),
        expected_outcome=(
            "El modelo traduce correctamente las solicitudes en acciones y selecciona "
            "herramientas válidas, con parámetros en su mayoría consistentes (A1-A3 correctos). "
            "Puede presentar imprecisiones en la interpretación del contexto temporal (A4) "
            "y/o inconsistencias parciales en el razonamiento (A5), así como errores en el orden "
            "o dependencias del plan (A6), sin comprometer completamente la viabilidad general del plan."
        )
    ),
    Rubric(
        score_range=(8, 10),
        expected_outcome=(
            "El modelo genera un plan completo y correcto a partir de las solicitudes, "
            "selecciona y parametriza adecuadamente las herramientas, "
            "interpreta correctamente el contexto temporal, "
            "mantiene coherencia interna (A1-A5 correctos) "
            "y define un orden lógico con dependencias correctas entre acciones (A6 correcto)."
        )
    )
]

metric_Planeador = GEval(
    name="Eval Planeador Tesis",
    criteria=CRITERIOS_PLANEADOR,
    rubric=RUBRICA_PLANEADOR,
    evaluation_steps=[
        "1. Lee la 'Input' (la solicitud original del usuario validada en contexto temporal).",
        "2. Lee el 'Expected Output' (el plan de acciones ideal y sus notas esperadas).",
        "3. Lee el 'Actual Output' (el plan propuesto por el modelo que está siendo evaluado).",
        "4. Compara el uso de herramientas, parámetros de fechas/dispositivos y lógica temporal (A1-A6).",
        "5. Asigna una puntuación basada en la rúbrica de 0 a 10, razonando detalladamente qué falló comparado con el Expected Output."
    ],
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT, LLMTestCaseParams.CONTEXT],
    model=model_juez
)

def evaluar_resultados_modelo(ruta_resultados_modelo, ruta_golden_answer, ruta_golden_questions, ruta_summary, ruta_temporal, eval_filepath):
    """
    Evalúa los resultados de un modelo caso por caso, guardando progreso instantáneamente.
    """
    with open(ruta_golden_answer, 'r', encoding='utf-8') as f:
        golden_answers = json.load(f)
        
    with open(ruta_golden_questions, 'r', encoding='utf-8') as f:
        golden_questions = json.load(f)
        
    with open(ruta_resultados_modelo, 'r', encoding='utf-8') as f:
        resultados_modelo = json.load(f)

    with open(ruta_summary, 'r', encoding='utf-8') as f:
        golden_summary = json.load(f)
        
    with open(ruta_temporal, 'r', encoding='utf-8') as f:
        temporal_context = json.load(f)

    # Convertimos los contextos a strings
    str_summary = json.dumps(golden_summary, indent=2, ensure_ascii=False)
    str_temporal = json.dumps(temporal_context, indent=2, ensure_ascii=False)

    # Intentar cargar progreso previo
    if os.path.exists(eval_filepath):
        with open(eval_filepath, 'r', encoding='utf-8') as f:
            reporte_existente = json.load(f)
        print(f"   📂 Cargando progreso previo ({len(reporte_existente.get('niveles', {}))} niveles detectados)...")
    else:
        reporte_existente = {
            "promedio_global": 0,
            "total_casos_evaluados": 0,
            "niveles": {}
        }

    # Procesar niveles y casos
    for nivel, casos in resultados_modelo.items():
        if nivel not in reporte_existente["niveles"]:
            reporte_existente["niveles"][nivel] = {"casos": {}, "promedio": 0}
            
        for id_caso, salida_actual in casos.items():
            # Validación por si el golden no tiene el caso
            if id_caso not in golden_answers.get(nivel, {}):
                continue
                
            # SALTAR SI YA ESTÁ EVALUADO
            if id_caso in reporte_existente["niveles"][nivel]["casos"]:
                continue

            expected_salida = golden_answers[nivel][id_caso]
            input_solicitud = golden_questions[nivel][id_caso]
            
            # Formateamos para el juez
            input_text = json.dumps(input_solicitud, indent=2, ensure_ascii=False)
            actual_text = json.dumps(salida_actual, indent=2, ensure_ascii=False)
            expected_text = json.dumps(expected_salida, indent=2, ensure_ascii=False)
            
            test_case = LLMTestCase(
                input=input_text,
                actual_output=actual_text,
                expected_output=expected_text,
                context=[f"SYSTEM_SUMMARY:\n{str_summary}", f"TEMPORAL_CONTEXT:\n{str_temporal}"]
            )
            
            try:
                print(f"   ⏳ Evaluando {nivel} -> {id_caso}...")
                metric_Planeador.measure(test_case)
                
                # Actualizar reporte
                reporte_existente["niveles"][nivel]["casos"][id_caso] = {
                    "score": metric_Planeador.score,
                    "reason": metric_Planeador.reason
                }
                
                # Recalcular métricas parciales para el modelo en este punto
                all_scores = []
                for n_val in reporte_existente["niveles"].values():
                    for c_val in n_val["casos"].values():
                        all_scores.append(c_val["score"])
                
                reporte_existente["total_casos_evaluados"] = len(all_scores)
                reporte_existente["promedio_global"] = sum(all_scores) / len(all_scores) if all_scores else 0
                
                # Recalcular promedio del nivel actual
                scores_nivel = [c["score"] for c in reporte_existente["niveles"][nivel]["casos"].values()]
                reporte_existente["niveles"][nivel]["promedio"] = sum(scores_nivel) / len(scores_nivel) if scores_nivel else 0

                # GUARDADO ATÓMICO TRAS CADA EVALUACIÓN
                with open(eval_filepath, "w", encoding="utf-8") as f:
                    json.dump(reporte_existente, f, indent=2, ensure_ascii=False)
                
                print(f"   ✅ Score: {metric_Planeador.score} (Guardado)")
                
            except Exception as e:
                print(f"   ❌ Error evaluando el caso {nivel}-{id_caso}: {str(e)}")

    return reporte_existente

def main():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    RESULTADOS_DIR = os.path.join(BASE_DIR, "Resultados")
    CASOS_DIR = os.path.join(BASE_DIR, "Set_casos")
    EVALUACION_DIR = os.path.join(BASE_DIR, "resultados_evaluador")
    
    os.makedirs(EVALUACION_DIR, exist_ok=True)
    
    golden_answer_path = os.path.join(CASOS_DIR, "Golden_answer_planeador.json")
    golden_questions_path = os.path.join(CASOS_DIR, "Golden_questions_planeador.json")
    golden_summary_path = os.path.join(CASOS_DIR, "Golden_summary.json")
    golden_temporal_path = os.path.join(CASOS_DIR, "Golden_Temporal_Context.json")
    
    print("\n==================================")
    print("🚀 INICIANDO EVALUACIÓN DEL PLANEADOR")
    print("==================================\n")

    # Identificar modelos basados en archivos respuestas_*.json
    modelos_a_evaluar = [f for f in os.listdir(RESULTADOS_DIR) if f.startswith("respuestas_") and f.endswith(".json")]
    
    for filename in modelos_a_evaluar:
        modelo_nombre = filename.replace("respuestas_", "").replace(".json", "")
        print(f"\n[{modelo_nombre}] - Procesando resultados...")
        
        ruta_modelo = os.path.join(RESULTADOS_DIR, filename)
        eval_filepath = os.path.join(EVALUACION_DIR, f"evaluacion_{modelo_nombre}.json")
        
        # Evaluar modelo (la función maneja internamente la reanudación y el guardado incremental)
        reporte_final_modelo = evaluar_resultados_modelo(
            ruta_modelo, 
            golden_answer_path, 
            golden_questions_path, 
            golden_summary_path, 
            golden_temporal_path,
            eval_filepath
        )
            
        print(f"\n📊 [{modelo_nombre}] EVALUACIÓN FINALIZADA")
        print(f"   Promedio General: {reporte_final_modelo['promedio_global']:.3f}/10")
        print(f"   Casos Evaluados: {reporte_final_modelo['total_casos_evaluados']}")

    print("\n==================================")
    print("✅ PROCESO DE EVALUACIÓN GLOBAL FINALIZADO")
    print("==================================\n")

if __name__ == "__main__":
    main()
