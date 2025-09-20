import dspy
from Tools import tools_catalogo
from Wrapped_Tools import fewshot_ejemplos

# Configuración del LM
lm = dspy.LM('ollama_chat/llama3.1', api_base='http://localhost:11434', api_key='')
dspy.configure(lm=lm)

class PlanificarProceso(dspy.Signature):
    pregunta = dspy.InputField(dtype=str, desc="La pregunta o petición realizada por el usuario.")
    tools_disponibles = dspy.InputField(dtype=list[dict], desc="Lista de funciones disponibles con su descripción y entradas/salidas.")

    planeacion = dspy.OutputField(dtype=str, desc="Explicación en lenguaje natural de cómo se resolverá la petición usando las funciones disponibles.")
    plan = dspy.OutputField(dtype=list[dict], desc="Lista de pasos con id, funcion, desc y dependencias.")


# ---------------------------
# Entrenamiento con BootstrapFewShot
# ---------------------------
trainer = dspy.BootstrapFewShot()

planificador = trainer.compile(
    student=dspy.Predict(PlanificarProceso),
    trainset=fewshot_ejemplos
)
    
# ---------------------------
# Ejemplo de uso
# ---------------------------
resultado = planificador(
    pregunta="cuánto fue el consumo de mi aire acondicionado el 5 de marzo y el del PC el 8 de octubre, cuál de ellos consume menos?",
    tools_disponibles=tools_catalogo
)

print("planeación:\n", resultado.planeacion)
print("lista procesos:\n", resultado.plan)
