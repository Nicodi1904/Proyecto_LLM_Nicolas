import dspy
from MCP_cliente import system_summary
from typing import List, Dict, Any, Optional


################################################################
class Toolformer(dspy.Signature):
    pregunta: str = dspy.InputField(
    desc="Mensaje original del usuario."
    )
    system_summary:Dict[str, Any] = dspy.InputField(
        desc=(
        "Resumen de herramientas disponibles y sus parámetros. "
        "Incluye cómo deben llamarse y en qué casos se usan."
        )
    )
    Toolformer:str=dspy.OutputField(
        desc=("Mensaje del usuario original con llamadas a herramientas incluidas, las llamadas a herramientas se deben incluir delimitando con asteriscos: Texto*@nombre_herramienta,@...*Texto ")

    )

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
pregunta=("Necesito saber cuánto consumió mi nevera ayer por la noche, "
    "y también cuánto consumió mi lavadora el sábado pasado en la mañana. "
    "Además quiero que me digas si entre esos dos días cuál gastó más energía. "
    "Ah, y por cierto, mientras miraba esos consumos se me descargó el celular "
    "y me dio mucha pereza pararme a buscar el cargador, pero igual quiero la comparación."
    "Papá también pidió que le dijeras cuánto fue el consumo de todos los dispositivos en el año 2024, quiero ver gráficas de todo lo que se pueda")

################################################################




# Configurar LLMs 
llama_31 = dspy.LM('ollama_chat/llama3.1:latest', api_base='http://localhost:11434', api_key='')
deepseek = dspy.LM('ollama_chat/deepseek-r1:8b', api_base='http://localhost:11434', api_key='')
mistral = dspy.LM('ollama_chat/mistral', api_base='http://localhost:11434', api_key='')
gemma = dspy.LM('ollama_chat/gemma', api_base='http://localhost:11434', api_key='')



################################################################
print("inicializando LLMs")
#inicializar la signature con LLAMA 3.1
dspy.configure(lm=llama_31) 
Rag_llama31 = dspy.Predict(Toolformer)

#Llamada al LLM
resultado = Rag_llama31(
    pregunta = pregunta,
    #tiempo_actual="2025-5-15 16:30",
    #feedback=None,
    system_summary=system_summary,
)


#inicializar la signature con deepseek
dspy.configure(lm=deepseek)
Rag_deepseek = dspy.Predict(Toolformer)

resultado2 = Rag_deepseek(
    pregunta = pregunta,
    #tiempo_actual="2025-5-15 16:30",
    #feedback=None,
    system_summary=system_summary,
)

#inicializar la signature con mistral
dspy.configure(lm=mistral)
Rag_mistral = dspy.Predict(Toolformer)

resultado3 = Rag_mistral(
    pregunta = pregunta,
    #tiempo_actual="2025-5-15 16:30",
    #feedback=None,
    system_summary=system_summary,
)

#inicializar la signature con gemma
dspy.configure(lm=gemma)
Rag_gemma = dspy.Predict(Toolformer)

resultado4 = Rag_gemma(
    pregunta = pregunta,
    #tiempo_actual="2025-5-15 16:30",
    #feedback=None,
    system_summary=system_summary,
)

################################################################


razonamiento=resultado.Toolformer


razonamiento2=resultado2.Toolformer


razonamiento3=resultado3.Toolformer


razonamiento4=resultado4.Toolformer

################################################################


""" 
################################################################

intenciones_principales=resultado.intenciones_principales
razonamiento=resultado.razonamiento

intenciones_principales2=resultado2.intenciones_principales
razonamiento2=resultado2.razonamiento

intenciones_principales3=resultado3.intenciones_principales
razonamiento3=resultado3.razonamiento

intenciones_principales4=resultado4.intenciones_principales
razonamiento4=resultado4.razonamiento

################################################################

print("==================Mensaje enviado por el usuario==================")
print(pregunta)
print("=====================")


print("=== COOKED Ollama 3.1 ===")
print('las intenciones principales detectadas fueron:\n',intenciones_principales)
print('el razonamiento realizado fue:\n',razonamiento)

print("=== COOKED Deepseek ===")
print('las intenciones principales detectadas fueron:\n',intenciones_principales2)
print('el razonamiento realizado fue:\n',razonamiento2)

print("=== COOKED Mistral ===")
print('las intenciones principales detectadas fueron:\n',intenciones_principales3)
print('el razonamiento realizado fue:\n',razonamiento3)

print("=== COOKED Gemma ===")
print('las intenciones principales detectadas fueron:\n',intenciones_principales4)
print('el razonamiento realizado fue:\n',razonamiento4) """


print("==================Mensaje enviado por el usuario==================")
print(pregunta)
print("=====================")


print("=== COOKED Ollama 3.1 ===")

print('el razonamiento realizado fue:\n',razonamiento)

print("=== COOKED Deepseek ===")

print('el razonamiento realizado fue:\n',razonamiento2)

print("=== COOKED Mistral ===")

print('el razonamiento realizado fue:\n',razonamiento3)

print("=== COOKED Gemma ===")

print('el razonamiento realizado fue:\n',razonamiento4)