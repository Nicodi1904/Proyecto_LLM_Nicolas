#Wrappear las tools creadas
from Tools import sumar,restar,consumo_rango_horas,consumo_rango_dias,consumo_rango_meses,calcular_min,calcular_max,calcular_promedio
import dspy
import pandas as pd

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
