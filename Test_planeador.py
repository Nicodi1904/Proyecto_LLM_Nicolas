import dspy
from Tools import tools_catalogo

# Configuración del LM
lm = dspy.LM('ollama_chat/llama3.1', api_base='http://localhost:11434', api_key='')
dspy.configure(lm=lm)

class PlanificarProceso(dspy.Signature):
    pregunta = dspy.InputField(dtype=str, desc="La pregunta o petición realizada por el usuario.")
    tools_disponibles = dspy.InputField(dtype=list[dict], desc="Lista de funciones disponibles con su descripción y entradas/salidas.")

    planeacion = dspy.OutputField(dtype=str, desc="Explicación en lenguaje natural de cómo se resolverá la petición usando las funciones disponibles.")
    plan = dspy.OutputField(dtype=list[dict], desc="Lista de pasos con id, funcion, desc y dependencias.")

# ---------------------------
# Few-shot de ejemplo
# ---------------------------
# ---------------------------
# Ejemplos Few-Shot
# ---------------------------
fewshot_ejemplos = [

    dspy.Example(
        pregunta="¿Cuánto consumió la nevera el 10 de enero del 2024?",
        tools_disponibles=tools_catalogo,
        planeacion="El usuario quiere el consumo de la nevera en un único día (10 de enero de 2024). "
                   "Esto se resuelve usando la función consumo_rango_dias.",
        plan=[
            {"id": 0, "funcion": "consumo_rango_dias", "desc": "Calcular consumo de la nevera el 10 de enero 2024",
             "dependencias": []}
        ]
    ).with_inputs("pregunta", "tools_disponibles"),

    dspy.Example(
        pregunta="¿Cuál fue el consumo de mi PC durante todo febrero del 2024?",
        tools_disponibles=tools_catalogo,
        planeacion="El usuario quiere el consumo de la PC en un mes completo (febrero 2024). "
                   "Esto se resuelve usando la función consumo_rango_meses.",
        plan=[
            {"id": 0, "funcion": "consumo_rango_meses", "desc": "Calcular consumo de la PC en febrero 2024",
             "dependencias": []}
        ]
    ).with_inputs("pregunta", "tools_disponibles"),

    dspy.Example(
        pregunta="¿Cuánto consumieron juntos mi aire acondicionado y mi PC el 15 de marzo del 2024?",
        tools_disponibles=tools_catalogo,
        planeacion="El usuario quiere el consumo conjunto de dos dispositivos (AC y PC) en un día específico. "
                   "Se calcula cada consumo con consumo_rango_dias y luego se suman.",
        plan=[
            {"id": 0, "funcion": "consumo_rango_dias", "desc": "Calcular consumo del aire acondicionado el 15 de marzo 2024", "dependencias": []},
            {"id": 1, "funcion": "consumo_rango_dias", "desc": "Calcular consumo del PC el 15 de marzo 2024", "dependencias": []},
            {"id": 2, "funcion": "sumar", "desc": "Sumar los consumos del AC y PC", "dependencias": [0, 1]}
        ]
    ).with_inputs("pregunta", "tools_disponibles"),

    dspy.Example(
        pregunta="¿Cuál fue el consumo total de mi aire acondicionado, PC y nevera en junio del 2024?",
        tools_disponibles=tools_catalogo,
        planeacion="El usuario quiere el consumo total de tres dispositivos durante un mes completo (junio 2024). "
                   "Se calcula cada consumo con consumo_rango_meses y luego se suman.",
        plan=[
            {"id": 0, "funcion": "consumo_rango_meses", "desc": "Calcular consumo del aire acondicionado en junio 2024", "dependencias": []},
            {"id": 1, "funcion": "consumo_rango_meses", "desc": "Calcular consumo del PC en junio 2024", "dependencias": []},
            {"id": 2, "funcion": "consumo_rango_meses", "desc": "Calcular consumo de la nevera en junio 2024", "dependencias": []},
            {"id": 3, "funcion": "sumar", "desc": "Sumar consumos de AC, PC y nevera", "dependencias": [0, 1, 2]}
        ]
    ).with_inputs("pregunta", "tools_disponibles"),

    dspy.Example(
        pregunta="¿Qué dispositivo fue el que más consumió entre mi PC y mi AC en abril del 2024?",
        tools_disponibles=tools_catalogo,
        planeacion="El usuario quiere comparar qué dispositivo consumió más en un mes específico (abril 2024). "
                   "Se calcula cada consumo con consumo_rango_meses y luego se comparan con calcular_max.",
        plan=[
            {"id": 0, "funcion": "consumo_rango_meses", "desc": "Calcular consumo del PC en abril 2024", "dependencias": []},
            {"id": 1, "funcion": "consumo_rango_meses", "desc": "Calcular consumo del aire acondicionado en abril 2024", "dependencias": []},
            {"id": 2, "funcion": "calcular_max", "desc": "Comparar consumos de PC y aire acondicionado", "dependencias": [0, 1]}
        ]
    ).with_inputs("pregunta", "tools_disponibles")
]

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
