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
# Few-shots de ejemplo
# ---------------------------
fewshot_ejemplos = [

    dspy.Example(
        pregunta="¿Cuál es el número más pequeño entre 3.5, 7.8 y 2.1?",
        tools_disponibles=tools_catalogo,
        planeacion=("El usuario quiere encontrar el valor mínimo dentro de una lista de números. "
                    "La tool apropiada es calcular_min, que recibe una lista de valores y devuelve el mínimo. "
                    "Solo se necesita un paso."),
        plan=[
            {"id": 0, "funcion": "calcular_min", "desc": "Obtener el mínimo de [3.5, 7.8, 2.1]", "dependencias": {"valores": [3.5, 7.8, 2.1]}}
        ]
    ).with_inputs("pregunta", "tools_disponibles"),

    dspy.Example(
        pregunta="Dime el promedio del consumo de mi nevera en enero, febrero y marzo de 2024.",
        tools_disponibles=tools_catalogo,
        planeacion=("El usuario quiere el promedio mensual del consumo de la nevera para enero–marzo de 2024. "
                    "La tool consumo_rango_meses devuelve el consumo total acumulado entre mes_inicio y mes_fin. "
                    "Para obtener el promedio mensual se calcula primero el total con consumo_rango_meses (enero a marzo) "
                    "y luego se divide ese total entre 3 usando la tool dividir. "
                    "Se usa una constante 3 como divisor."),
        plan=[
            {"id": 0, "funcion": "consumo_rango_meses", "desc": "Calcular consumo total de la nevera desde mes_inicio=1 hasta mes_fin=3 en 2024 (enero-marzo).", "dependencias": {"dispositivo": "nevera", "mes_inicio": int(1), "mes_fin": int(3), "año": int(2024)}},
            {"id": 1, "funcion": "dividir", "desc": "Dividir el consumo total (salida del paso 0) entre 3 para obtener el promedio mensual.", "dependencias": {"a": "@0", "b": int(3)}}
        ]
    ).with_inputs("pregunta", "tools_disponibles"),

    dspy.Example(
        pregunta="¿Cuánto consumieron juntos mi televisor y mi consola el 10 de mayo de 2024 entre las 14 y las 20 horas?",
        tools_disponibles=tools_catalogo,
        planeacion=("El usuario pide el consumo conjunto de dos dispositivos en un rango horario específico. "
                    "Se debe calcular el consumo del televisor y de la consola en el mismo rango de horas y fecha "
                    "usando consumo_rango_horas, y luego sumar ambos resultados con la tool sumar."),
        plan=[
            {"id": 0, "funcion": "consumo_rango_horas", "desc": "Calcular consumo del televisor entre 14 y 20 horas del 10 de mayo de 2024", "dependencias": {"dispositivo": "televisor", "hora_inicio": int(14), "hora_fin": int(20), "dia": int(10), "mes": int(5), "año": int(2024)}},
            {"id": 1, "funcion": "consumo_rango_horas", "desc": "Calcular consumo de la consola entre 14 y 20 horas del 10 de mayo de 2024", "dependencias": {"dispositivo": "consola", "hora_inicio": int(14), "hora_fin": int(20), "dia": int(10), "mes": int(5), "año": int(2024)}},
            {"id": 2, "funcion": "sumar", "desc": "Sumar los consumos del televisor y la consola", "dependencias": {"a": "@0", "b": "@1"}}
        ]
    ).with_inputs("pregunta", "tools_disponibles"),

    dspy.Example(
        pregunta="Por curiosidad, y solo si no es mucho problema, ¿podrías calcularme el máximo entre los valores 12.5, 8.3, 15.9 y 11.1? Sé que tú puedes hacerlo mejor que yo.",
        tools_disponibles=tools_catalogo,
        planeacion=("Aunque el usuario da rodeos, la intención central es encontrar el valor máximo de una lista de números. "
                    "Se ignoran las frases irrelevantes y se utiliza calcular_max directamente."),
        plan=[
            {"id": 0, "funcion": "calcular_max", "desc": "Obtener el máximo de [12.5, 8.3, 15.9, 11.1]", "dependencias": {"valores": [12.5, 8.3, 15.9, 11.1]}}
        ]
    ).with_inputs("pregunta", "tools_disponibles"),

    dspy.Example(
        pregunta="Quiero saber la diferencia de consumo entre mi lavadora y mi secadora durante la primera semana de abril de 2024.",
        tools_disponibles=tools_catalogo,
        planeacion=("El usuario busca la diferencia de consumos entre dos dispositivos en un rango de días. "
                    "Se calcula el consumo de la lavadora y de la secadora con consumo_rango_dias, luego se obtiene la diferencia con la tool restar."),
        plan=[
            {"id": 0, "funcion": "consumo_rango_dias", "desc": "Calcular consumo de la lavadora del 1 al 7 de abril de 2024", "dependencias": {"dispositivo": "lavadora", "dia_inicio": int(1), "dia_fin": int(7), "mes": int(4), "año": int(2024)}},
            {"id": 1, "funcion": "consumo_rango_dias", "desc": "Calcular consumo de la secadora del 1 al 7 de abril de 2024", "dependencias": {"dispositivo": "secadora", "dia_inicio": int(1), "dia_fin": int(7), "mes": int(4), "año": int(2024)}},
            {"id": 2, "funcion": "restar", "desc": "Restar el consumo de la secadora al de la lavadora", "dependencias": {"a": "@0", "b": "@1"}}
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
