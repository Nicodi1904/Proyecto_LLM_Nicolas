from Tools import sumar,restar,consumo_rango_horas,consumo_rango_dias,consumo_rango_meses
import dspy
import pandas as pd

#Wrappear las tools creadas
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
    

class Planificador(dspy.Signature):
    """
    El modelo debe identificar todas las operaciones matemáticas
    (sumas o restas) dentro de la pregunta.
    """
    pregunta: str = dspy.InputField()
    operaciones: list[dict] = dspy.OutputField(
        desc="Lista de diccionarios con las indicaciones a realizar, cada diccionario traerá: 'tipo', 'reasoning', variables necesarias. Ejemplo de respuesta:  [{'tipo': 'resta', 'reasoning': el usuario solicitó una resta ,'a': 10, 'b': 3}, ...] "
    )
    respuesta: str = dspy.OutputField(desc="Respuesta final para el usuario")


#Lógica del agente
class Agente(dspy.Module):
    #Se declaran variables y funciones propias que el agente puede tener
    def __init__(self):
        super().__init__()
        self.sumar = SumarModule()
        self.restar = RestarModule()
        self.planificador = dspy.ChainOfThought(Planificador)

    #Se declara el "Accionador" que llamará las tools creadas
    def __call__(self, pregunta: str):
        # El LLM genera un plan usando la signature
        plan = self.planificador(pregunta=pregunta)
        print(plan)

        # Si el modelo detecta operaciones
        if plan.operaciones and isinstance(plan.operaciones, list): #isinstance sirve para corroborar que una variable sea de cierto tipo
            resultados = []
            
            for operacion in plan.operaciones:

                if operacion["tipo"] == "suma":
                    resultado = self.sumar(operacion["a"], operacion["b"])
                    resultados.append(f"{operacion['a']} + {operacion['b']} = {resultado}")

                elif operacion["tipo"] == "resta":
                    resultado = self.restar(operacion["a"], operacion["b"])
                    resultados.append(f"{operacion['a']} - {operacion['b']} = {resultado}")

            # Devolver resultados combinados
            return " | ".join(resultados)

        # Si no hay operaciones, devolver la respuesta normal
        return plan.respuesta
    
#Cargar LLM
lm = dspy.LM('ollama_chat/llama3.1', api_base='http://localhost:11434', api_key='')
dspy.configure(lm=lm)

