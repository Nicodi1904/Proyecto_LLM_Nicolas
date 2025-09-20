#Wrappear las tools creadas
from Tools import tools_catalogo,sumar,restar,consumo_rango_horas,consumo_rango_dias,consumo_rango_meses,calcular_min,calcular_max,calcular_promedio
import dspy
import pandas as pd

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




class SumarModule(dspy.Module):
    def __call__(self, a: int, b: int) -> int:
        return sumar(a, b)

class RestarModule(dspy.Module):
    def __call__(self, a: int, b: int) -> int:
        return restar(a, b)

class ConsumoRangoHorasModule(dspy.Module):
    def __call__(self, df: pd.DataFrame, dispositivo: str, dia: int, mes: int, año: int, hora_inicio: int, hora_fin: int) -> float:
        return consumo_rango_horas(df, dispositivo, hora_inicio, hora_fin, dia, mes, año)

class ConsumoRangoDiasModule(dspy.Module):
    def __call__(self, df: pd.DataFrame, dispositivo: str, dia_inicio: int, dia_fin: int, mes: int, año: int) -> float:
        return consumo_rango_dias(df, dispositivo, dia_inicio, dia_fin, mes, año)

class ConsumoRangoMesesModule(dspy.Module):
    def __call__(self, df: pd.DataFrame, dispositivo: str, mes_inicio: int, mes_fin: int, año: int) -> float:
        return consumo_rango_meses(df, dispositivo, mes_inicio, mes_fin, año)

class CalcularMinModule(dspy.Module):
    def __call__(self, valores: list[float]) -> float:
        return calcular_min(valores)

class CalcularMaxModule(dspy.Module):
    def __call__(self, valores: list[float]) -> float:
        return calcular_max(valores)

class CalcularPromedioModule(dspy.Module):
    def __call__(self, valores: list[float]) -> float:
        return calcular_promedio(valores)
