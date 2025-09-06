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
    Descompone la pregunta del usuario en subtareas lógicas necesarias
    para llegar a una respuesta óptima. 

    - No ejecuta nada.
    - No asigna herramientas.
    - Solo define el flujo de razonamiento y qué variables son importantes.
    """

    pregunta:str = dspy.InputField(desc="Pregunta original del usuario en lenguaje natural")
    plan:list[dict]= dspy.OutputField(desc="Lista de subtareas en formato estructurado. \
        Cada subtarea debe ser un diccionario con: \
        - descripcion: explicación clara de la acción \
        - variables: parámetros que serán útiles (ej. dispositivo, fecha, rango de horas, etc.) \
        - dependencias: si la subtarea depende de resultados anteriores")

class Gerente(dspy.Signature):
    """
    Asigna herramientas específicas a las subtareas generadas por el planner.
    """
    tools_disponibles: list[dict] = dspy.InputField(
        desc=(
            "Catálogo de herramientas disponibles. "
            "Cada tool es un diccionario con llaves: "
            "'nombre' (str), 'descripcion' (str que explica qué hace la tool "
            "y qué variables espera)."
        )
    )

    plan: list[dict] = dspy.InputField(
        desc="Lista de subtareas en formato estandarizado (con 'descripcion', 'variables', 'dependencias')."
    )

    procesos: list[dict] = dspy.OutputField(
       desc=(
            "Lista de subtareas con tool asignada. "
            "Cada subtarea es un diccionario con llaves: "
            "'tarea' (str, descripcion clara de la subtarea), "
            "'tool' (str, nombre de la tool seleccionada del catálogo), "
            "'variables' (dict con los parámetros necesarios), "
            "'dependencias' (list de dependencias si aplica)."
        )
    )






#Módulo principal
class Agente(dspy.Module):
    #Se declaran variables y funciones propias que el agente puede tener
    def __init__(self):
        super().__init__()
        self.sumar = SumarModule()
        self.restar = RestarModule()

        #Posibles Roles a interpretar
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

