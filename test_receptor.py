import dspy
from MCP_cliente import system_summary
from typing import List, Dict, Any, Optional

################################################################
class Rag_Generator(dspy.Signature):
    """
    Crea una lista simplificada y organizada de las peticiones del usuario basándose en las especificaciones del sistema.
    """
    # ============================
    # ENTRADAS
    # ============================
    pregunta: str = dspy.InputField(
    desc="Mensaje original del usuario."
    )
    feedback: Optional[Dict[str, Any]]= dspy.InputField(
    default=None,
    desc="Contexto previo relevante (historial, preferencias, restricciones, dispositivos conocidos)."
    )
    tiempo_actual: str = dspy.InputField(
    desc="Fecha y hora actuales del sistema en formato 'YYYY-MM-DD HH:MM' para interpretar referencias temporales relativas."
    )
    system_summary: Dict[str, Any] = dspy.InputField(
    desc=(
        "Resumen de herramientas disponibles y sus parámetros. "
        "Incluye cómo deben llamarse y en qué casos se usan."
        )
    )

    # ============================
    # SALIDAS
    # ============================
    intenciones_principales: List[Dict[str, Any]] = dspy.OutputField(
    desc=(
        "Lista ordenada de acciones derivadas de la solicitud del usuario. "
        "Cada acción debe tener:\n"
        "- id: str (formato 'a0', 'a1', ...)\n"
        "- tipo: str (nombre de la operación del system_summary, o 'no_soportado')\n"
        "- parametros: dict (valores requeridos por la operación; si no pueden deducirse, usar None)\n"
        "- dependencias: list[str] (ids de acciones previas si aplica)\n"
        "El orden debe seguir el flujo lógico de ejecución.\n"
        "Si ninguna operación del system_summary corresponde, usar tipo='no_soportado' y "
        "parametros={'solicitud_original_fragmento': 'texto exacto de la solicitud concerniente'}."
        )
    )
    razonamiento: str = dspy.OutputField(
    desc=(
        "Explica cómo se asignó cada acción, qué parámetros quedaron como None y por qué surgieron acciones 'no_soportado' si las hubo."
        )
    )
################################################################

# Configurar LLM base
llama_31 = dspy.LM('ollama_chat/llama3.1:latest', api_base='http://localhost:11434', api_key='')
dspy.configure(lm=llama_31)


# Crear el predictor semántico
Rag = dspy.Predict(Rag_Generator)

################################################################

pregunta=("Necesito saber cuánto consumió mi nevera ayer por la noche, "
    "y también cuánto consumió mi lavadora el sábado pasado en la mañana. "
    "Además quiero que me digas si entre esos dos días cuál gastó más energía. "
    "Ah, y por cierto, mientras miraba esos consumos se me descargó el celular "
    "y me dio mucha pereza pararme a buscar el cargador, pero igual quiero la comparación."
    "Papá también pidió que le dijeras cuánto fue el consumo de todos los dispositivos en el año 2024, quiero ver gráficas de todo lo que se pueda")

################################################################
#Llamada al LLM
resultado = Rag(
    pregunta = pregunta,
    tiempo_actual="2025-5-15 16:30",
    feedback=None,
    system_summary=system_summary,
)

################################################################

intenciones_principales=resultado.intenciones_principales
razonamiento=resultado.razonamiento

################################################################

print("==================Mensaje enviado por el usuario==================")
print(pregunta)
print("=====================")


print("=== COOKED RESULTADO ===")
print('las intenciones principales detectadas fueron:\n',intenciones_principales)
print('el razonamiento realizado fue:\n',razonamiento)
