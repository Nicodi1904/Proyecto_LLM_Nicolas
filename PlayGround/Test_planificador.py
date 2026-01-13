import dspy
from Tools import tools_catalogo
from Wrapped_Tools import fewshot_ejemplos

# Configuración del LM
lm = dspy.LM('ollama_chat/llama3.1', api_base='http://localhost:11434', api_key='')
dspy.configure(lm=lm)

class Guia_Planificador(dspy.Signature):
    """
    Descompone una pregunta de usuario en subtareas claras y estructuradas,
    considerando el catálogo de herramientas disponibles.
    """

    pregunta: str = dspy.InputField(
        desc="Pregunta original del usuario en lenguaje natural sobre consumo energético"
    )

    tools_disponibles: list[dict] = dspy.InputField(
        desc=(
            "Catálogo de herramientas disponibles. Cada tool es un diccionario con:\n"
            "  - nombre (str)\n"
            "  - descripcion (str): explicación de qué hace y qué variables espera\n"
            "  - funcion (callable): función Python a ejecutar\n"
            "El Planificador debe usar este catálogo para decidir qué subtareas y secuencia generar."
        )
    )

    plan: list[dict] = dspy.OutputField(
        desc=(
            "Lista de subtareas en formato estrictamente definido. Cada subtarea es un diccionario con:\n"
            "  - id: identificador único de la subtarea (p.ej., 't1', 't2')\n"
            "  - descripcion: explicación clara y concisa de la acción a realizar\n"
            "  - variables: diccionario con variables abstractas necesarias (p.ej., dispositivo, dia_inicio, dia_fin, mes, año, hora_inicio, hora_fin)\n"
            "  - dependencias: lista de ids de subtareas previas de las cuales depende esta"
        )
    )

# ---------------------------
# Entrenamiento con BootstrapFewShot
# ---------------------------
trainer = dspy.BootstrapFewShot()

planificador = trainer.compile(
    student=dspy.Predict(Guia_Planificador),
    trainset=fewshot_ejemplos
)
    
# ---------------------------
# Ejemplo de uso
# ---------------------------
resultado = planificador(
    pregunta="cuánto fue el consumo de mi aire acondicionado el 5 de marzo y el del PC el 8 de octubre, cuál de ellos consume menos?",
    tools_disponibles=tools_catalogo
)

planeacion=resultado.planeacion
plan=resultado.plan
print("razonamiento para la planeación:\n", planeacion)
print("plan con la lista de procesos:\n", plan)
