import numpy as np
import dspy

planeacion=("El usuario pide el consumo conjunto de dos dispositivos en un rango horario específico. "
                    "Se debe calcular el consumo del televisor y de la consola en el mismo rango de horas y fecha "
                    "usando consumo_rango_horas, y luego sumar ambos resultados con la tool sumar.")
informe_worker={0: {'desc': 'Calcular consumo del televisor entre 14 y 20 horas del 10 de mayo de 2024','resultado': np.float64(0.1705)},
                1: {'desc': 'Calcular consumo de la lampara entre 14 y 20 horas del 10 de mayo de 2024','resultado': np.float64(0.07269999999999999)},
                2: {'desc': 'Sumar los consumos del televisor y la consola','resultado': np.float64(0.2432)}}
##########
planeacion2=("El usuario quiere encontrar el valor mínimo dentro de una lista de números. "
                    "La tool apropiada es calcular_min, que recibe una lista de valores y devuelve el mínimo. "
                    "Solo se necesita un paso."),
informe_worker2={0: {'desc': 'Obtener el mínimo de [3.5, 7.8, 2.1]','resultado': 2.1}}
#########
planeacion3=("Aunque el usuario da rodeos, la intención central es encontrar el valor máximo de una lista de números. "
                    "Se ignoran las frases irrelevantes y se utiliza calcular_max directamente.")
informe_worker3={0: {'desc': 'Obtener el máximo de [12.5, 8.3, 15.9, 11.1]','resultado': 15.9}}
##########
planeacion4=("El usuario busca la diferencia de consumos entre dos dispositivos en un rango de días. "
                    "Se calcula el consumo de la lavadora y de la secadora con consumo_rango_dias, luego se obtiene la diferencia con la tool restar.")
informe_worker4={0: {'desc': 'Calcular consumo de la lavadora del 1 al 7 de abril de 2024','resultado': np.float64(61.4751)},
                 1: {'desc': 'Calcular consumo de la secadora del 1 al 7 de abril de 2024','resultado': np.float64(7.1804)},
                 2: {'desc': 'Restar el consumo de la secadora al de la lavadora','resultado': np.float64(54.2947)}}


# Configuración del LM
lm = dspy.LM('ollama_chat/llama3.1', api_base='http://localhost:11434', api_key='')
dspy.configure(lm=lm)

class Gerente(dspy.Signature):
    """El gerente toma la planeación y los resultados del worker, y genera una respuesta clara y comprensible para el usuario final."""

    planeacion = dspy.InputField(dtype=str,desc="Explicación del plan que el planeador generó según la intención del usuario.")
    informe_worker = dspy.InputField(dtype=dict,desc="Diccionario con los pasos ejecutados: id, descripción y resultado de cada tool.")

    respuesta_usuario = dspy.OutputField(dtype=str,desc="Explicación en lenguaje natural, clara y resumida, con el resultado final y contexto si es necesario.")


gerente = dspy.Predict(Gerente)

resultado = gerente(
    planeacion=planeacion4,
    informe_worker=informe_worker4  # 👈 pásalo como string serializado
)

print(resultado)
