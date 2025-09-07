from Tools import sumar,restar,consumo_rango_horas,consumo_rango_dias,consumo_rango_meses,calcular_min,calcular_max,calcular_promedio
from Tools import tools_catalogo
from cargar_CSV import cargar_dataset_sinselejo
import dspy
import pandas as pd

########################################################################################################################  
#Contratos (me parece que es la mejor definición)

class Planificador(dspy.Signature):
    """
    Descompone la pregunta del usuario en subtareas lógicas necesarias.
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
    IMPORTANTE: 
        -Los únicos dispositivos que existen en el dataset son: 
            Ventilador, PC, AC, Lampara, TV. 
        -Usa siempre los nombres exactamente como están escritos (respeta mayúsculas y sin tildes).
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

##############################################################################################################################
#Módulo principal
class Agente(dspy.Module):
    #Se declaran variables y funciones propias que el agente puede tener
    def __init__(self, tools_catalogo: list[dict], df: pd.DataFrame = None): #el dataframe puede que no siempre se solicite, entonces se inicializa en none
        #Se hereda clase padre módulo de dspy
        super().__init__()
        #Se cargan los roles de los LLMs
        self.planificador = dspy.ChainOfThought(Planificador)
        self.gerente = dspy.ChainOfThought(Gerente)
        #Cargar catalogo de tools (Se podría decir que esto ya cuenta como módulo de RAG..)
        self.tools_catalogo = tools_catalogo
        #Se cargan los datos
        self.df = df
        #más eficiente separar este dict al crear el objeto 
        self.funciones = {tool["nombre"]: tool.get("funcion") for tool in tools_catalogo}
    
    def __call__(self, pregunta: str): 

        #LLM1 genera un plan armando subtareas####
        reporte_planificador = self.planificador(pregunta=pregunta)
        print("Output LLM PLANIFICADOR\n",reporte_planificador)

        #LLM2 recibe nombre y descripcion de las subtareas y genera un reporte de tools por tarea####
        reporte_gerente = self.gerente(plan=reporte_planificador.plan, tools_disponibles=self.tools_catalogo)
        print("Output LLM GERENTE\n",reporte_gerente)

        #Se recogen los procesos hechos por el genrente y se le pasan al WORKER
        procesos = reporte_gerente.procesos
        #-----------------------------------------------------------WORKER----------------------------------------------
        # ----------------------------------------(Lógica de python que ejecutará las tareas)----------------------------------------------------
       
        work=[] #lista donde se almacenarán los trabajos hechos por cada worker (En realidad solo hay un worker acá, se podrán ejecutar operaciones en paralelo?)
        
        #Se recorren todos los procesos dejados por el gerente
        for proceso in procesos:
            nombre_tool = proceso["tool"] #Se extrae la tool de cada proceso
            variables = proceso.get("variables", {}) #Se extraen las variables necesarias para usar cada tool
            
            #Se usa "try" por si alguna tarea falla no afecte las demás
            try:
                # Si la tool necesita df, lo agregamos automáticamente 
                
                if "df" in self.funciones[nombre_tool].__code__.co_varnames:
                    if self.df is None:
                        raise ValueError(f"La tool {nombre_tool} requiere un DataFrame pero no se cargó ninguno")
                    variables["df"] = self.df

                resultado = self.funciones[nombre_tool](**variables) #Agarra la función de la tool del proceso y le pasa las variables extraidas
                work.append({
                    "tarea": proceso["tarea"],
                    "tool": nombre_tool,
                    "variables": variables,
                    "resultado": resultado
                })
            #Si el Try falla, avisa que falló y guarda en qué tool falló
            except Exception as e:
                print(f"[ERROR] Fallo al ejecutar tool '{nombre_tool}': {e}")
                work.append({
                    "tarea": proceso["tarea"],
                    "tool": nombre_tool,
                    "variables": variables,
                    "resultado": None,
                    "error": str(e)
                }) 
        return work


#Cargar LLM
lm = dspy.LM('ollama_chat/llama3.1', api_base='http://localhost:11434', api_key='')
dspy.configure(lm=lm)

#Cargar DataFrame hogar inteligente
df=cargar_dataset_sinselejo("Energy Consumption in KWh of a Typical House Sincelejo Colombia.csv")

#Crear agente
agente = Agente(tools_catalogo=tools_catalogo, df=df)

# ---------------------- PRUEBAS ----------------------
# Pregunta sencilla (1 subtarea, 1 tool)
print("\n[PREGUNTA 1]")
resultado1 = agente("¿Cuánto consumió el ventilador entre las 8 am y 5 pm del 15 de enero del 2024?")
for r in resultado1:
    print(r)

# Pregunta intermedia (requiere promedio)
print("\n[PREGUNTA 2]")
resultado2 = agente("¿Cuál fue el consumo promedio del TV durante febrero del 2024?")
for r in resultado2:
    print(r)


'''
Respuesta del LLM a las preguntas anteriores y algunos de los errores:
PS C:\Users\Nicolas\Documents\2025-2\Tesis-MAS-LLM> & C:/Users/Nicolas/Documents/2025-2/Tesis-MAS-LLM/agentes-vs-llm/Scripts/Activate.ps1
(agentes-vs-llm) PS C:\Users\Nicolas\Documents\2025-2\Tesis-MAS-LLM> & C:/Users/Nicolas/Documents/2025-2/Tesis-MAS-LLM/agentes-vs-llm/Scripts/python.exe c:/Users/Nicolas/Documents/2025-2/Tesis-MAS-LLM/Test_Agente_energia.py

[PREGUNTA 1]
Output LLM PLANIFICADOR
 Prediction(
    reasoning='Para responder a la pregunta del usuario sobre el consumo de energía del ventilador entre las 8 am y 5 pm del 15 de enero del 2024, debemos considerar los siguientes pasos lógicos:\n\n1. Identificar el dispositivo (ventilador) que se está consultando.\n2. Determinar la fecha específica para la que se requiere el consumo de energía (15 de enero del 2024).\n3. Establecer el rango horario durante el cual se consumió energía (entre las 8 am y 5 pm).\n4. Recopilar los datos de consumo de energía correspondientes a ese dispositivo, fecha y rango horario.',
    plan=[{'descripcion': 'Identificar el dispositivo', 'variables': {'dispositivo': 'ventilador'}, 'dependencias': []}, {'descripcion': 'Determinar la fecha específica', 'variables': {'fecha': '15 de enero del 2024'}, 'dependencias': []}, {'descripcion': 'Establecer el rango horario', 'variables': {'hora_inicio': '8 am', 'hora_fin': '5 pm'}, 'dependencias': ['dispositivo', 'fecha']}, {'descripcion': 'Recopilar datos de consumo de energía', 'variables': {'dispositivo': 'fecha', 'hora_inicio': 'hora_fin'}, 'dependencias': ['dispositivo', 'fecha', 'rango horario']}]
)
Output LLM GERENTE
 Prediction(
    reasoning='Se asignarán herramientas específicas a las subtareas generadas por el planner para realizar los cálculos necesarios.',
    procesos=[{'tarea': 'Identificar el dispositivo', 'tool': 'consumo_rango_horas', 'variables': {'dispositivo': 'ventilador'}, 'dependencias': []}, {'tarea': 'Determinar la fecha específica', 'tool': 'consumo_rango_horas', 'variables': {'fecha': '15 de enero del 2024'}, 'dependencias': []}, {'tarea': 'Establecer el rango horario', 'tool': 'consumo_rango_horas', 'variables': {'hora_inicio': '8 am', 'hora_fin': '5 pm'}, 'dependencias': ['dispositivo', 'fecha']}, {'tarea': 'Recopilar datos de consumo de energía', 'tool': 'consumo_rango_horas', 'variables': {'dispositivo': 'fecha', 'hora_inicio': 'hora_fin'}, 'dependencias': ['dispositivo', 'fecha', 'rango horario']}]
)
[ERROR] Fallo al ejecutar tool 'consumo_rango_horas': consumo_rango_horas() missing 5 required positional arguments: 'hora_inicio', 'hora_fin', 'dia', 'mes', and 'año'
[ERROR] Fallo al ejecutar tool 'consumo_rango_horas': consumo_rango_horas() got an unexpected keyword argument 'fecha'
[ERROR] Fallo al ejecutar tool 'consumo_rango_horas': consumo_rango_horas() missing 4 required positional arguments: 'dispositivo', 'dia', 'mes', and 'año'
[ERROR] Fallo al ejecutar tool 'consumo_rango_horas': consumo_rango_horas() missing 4 required positional arguments: 'hora_fin', 'dia', 'mes', and 'año'
{'tarea': 'Identificar el dispositivo', 'tool': 'consumo_rango_horas', 'variables': {'dispositivo': 'ventilador', 'df':                TimeStamp  Ventilador   PC      AC  Lampara      TV
0    2024-01-01 00:00:00      0.0310  0.0  0.2189   0.0153  0.0160
1    2024-01-01 01:00:00      0.0262  0.0  0.5725   0.0025  0.0160
2    2024-01-01 02:00:00      0.0162  0.0  0.4512   0.0026  0.0156
3    2024-01-01 03:00:00      0.0269  0.0  0.4097   0.0012  0.0157
4    2024-01-01 04:00:00      0.0654  0.0  0.4086   0.0013  0.0160
...                  ...         ...  ...     ...      ...     ...
8779 2024-12-31 19:00:00      0.0163  0.0  0.4789   0.0189  0.0434
8780 2024-12-31 20:00:00      0.0162  0.0  0.4862   0.0164  0.0407
8781 2024-12-31 21:00:00      0.0163  0.0  0.3386   0.0184  0.0282
8782 2024-12-31 22:00:00      0.0580  0.0  0.3183   0.0049  0.0154
8783 2024-12-31 23:00:00      0.0706  0.0  0.3279   0.0000  0.0153

[8784 rows x 6 columns]}, 'resultado': None, 'error': "consumo_rango_horas() missing 5 required positional arguments: 'hora_inicio', 'hora_fin', 'dia', 'mes', and 'año'"}
{'tarea': 'Determinar la fecha específica', 'tool': 'consumo_rango_horas', 'variables': {'fecha': '15 de enero del 2024', 'df':                TimeStamp  Ventilador   PC      AC  Lampara      TV
0    2024-01-01 00:00:00      0.0310  0.0  0.2189   0.0153  0.0160
1    2024-01-01 01:00:00      0.0262  0.0  0.5725   0.0025  0.0160
2    2024-01-01 02:00:00      0.0162  0.0  0.4512   0.0026  0.0156
3    2024-01-01 03:00:00      0.0269  0.0  0.4097   0.0012  0.0157
4    2024-01-01 04:00:00      0.0654  0.0  0.4086   0.0013  0.0160
...                  ...         ...  ...     ...      ...     ...
8779 2024-12-31 19:00:00      0.0163  0.0  0.4789   0.0189  0.0434
8780 2024-12-31 20:00:00      0.0162  0.0  0.4862   0.0164  0.0407
8781 2024-12-31 21:00:00      0.0163  0.0  0.3386   0.0184  0.0282
8782 2024-12-31 22:00:00      0.0580  0.0  0.3183   0.0049  0.0154
8783 2024-12-31 23:00:00      0.0706  0.0  0.3279   0.0000  0.0153

[8784 rows x 6 columns]}, 'resultado': None, 'error': "consumo_rango_horas() got an unexpected keyword argument 'fecha'"}
{'tarea': 'Establecer el rango horario', 'tool': 'consumo_rango_horas', 'variables': {'hora_inicio': '8 am', 'hora_fin': '5 pm', 'df':                TimeStamp  Ventilador   PC      AC  Lampara      TV
0    2024-01-01 00:00:00      0.0310  0.0  0.2189   0.0153  0.0160
1    2024-01-01 01:00:00      0.0262  0.0  0.5725   0.0025  0.0160
2    2024-01-01 02:00:00      0.0162  0.0  0.4512   0.0026  0.0156
3    2024-01-01 03:00:00      0.0269  0.0  0.4097   0.0012  0.0157
4    2024-01-01 04:00:00      0.0654  0.0  0.4086   0.0013  0.0160
...                  ...         ...  ...     ...      ...     ...
8779 2024-12-31 19:00:00      0.0163  0.0  0.4789   0.0189  0.0434
8780 2024-12-31 20:00:00      0.0162  0.0  0.4862   0.0164  0.0407
8781 2024-12-31 21:00:00      0.0163  0.0  0.3386   0.0184  0.0282
8782 2024-12-31 22:00:00      0.0580  0.0  0.3183   0.0049  0.0154
8783 2024-12-31 23:00:00      0.0706  0.0  0.3279   0.0000  0.0153

[8784 rows x 6 columns]}, 'resultado': None, 'error': "consumo_rango_horas() missing 4 required positional arguments: 'dispositivo', 'dia', 'mes', and 'año'"}      
{'tarea': 'Recopilar datos de consumo de energía', 'tool': 'consumo_rango_horas', 'variables': {'dispositivo': 'fecha', 'hora_inicio': 'hora_fin', 'df':            
    TimeStamp  Ventilador   PC      AC  Lampara      TV
0    2024-01-01 00:00:00      0.0310  0.0  0.2189   0.0153  0.0160
1    2024-01-01 01:00:00      0.0262  0.0  0.5725   0.0025  0.0160
2    2024-01-01 02:00:00      0.0162  0.0  0.4512   0.0026  0.0156
3    2024-01-01 03:00:00      0.0269  0.0  0.4097   0.0012  0.0157
4    2024-01-01 04:00:00      0.0654  0.0  0.4086   0.0013  0.0160
...                  ...         ...  ...     ...      ...     ...
8779 2024-12-31 19:00:00      0.0163  0.0  0.4789   0.0189  0.0434
8780 2024-12-31 20:00:00      0.0162  0.0  0.4862   0.0164  0.0407
8781 2024-12-31 21:00:00      0.0163  0.0  0.3386   0.0184  0.0282
8782 2024-12-31 22:00:00      0.0580  0.0  0.3183   0.0049  0.0154
8783 2024-12-31 23:00:00      0.0706  0.0  0.3279   0.0000  0.0153

[8784 rows x 6 columns]}, 'resultado': None, 'error': "consumo_rango_horas() missing 4 required positional arguments: 'hora_fin', 'dia', 'mes', and 'año'"}

[PREGUNTA 2]
Output LLM PLANIFICADOR
 Prediction(
    reasoning='Para responder a la pregunta del consumo promedio de TV durante febrero del 2024, debemos considerar los siguientes pasos lógicos:\n\n1. Recopilar datos de consumo de energía eléctrica para el mes de febrero del 2024.\n2. Identificar el dispositivo específico que se refiere a "TV" en la pregunta (por ejemplo, un televisor de pantalla plana o un televisor inteligente).\n3. Calcular el consumo promedio diario de energía eléctrica durante febrero del 2024.',
    plan=[{'descripcion': 'Recopilar datos de consumo de energía eléctrica para febrero del 2024', 'variables': ['fecha', 'dispositivo (TV)'], 'dependencias': []}, {'descripcion': "Identificar el dispositivo específico que se refiere a 'TV'", 'variables': ['tipo de TV'], 'dependencias': ['recopilar datos']}, {'descripcion': 'Calcular el consumo promedio diario de energía eléctrica durante febrero del 2024', 'variables': ['consumo total', 'número de días en febrero'], 'dependencias': ['identificar dispositivo y recopilar datos']}]
)
Output LLM GERENTE
 Prediction(
    reasoning='Se asignarán herramientas específicas a las subtareas generadas por el planner para realizar los cálculos necesarios.',
    procesos=[{'tarea': 'Recopilar datos de consumo de energía eléctrica para febrero del 2024', 'tool': 'consumo_rango_meses', 'variables': {'dispositivo': 'TV', 'mes_inicio': 2, 'mes_fin': 2, 'año': 2024}, 'dependencias': []}, {'tarea': "Identificar el dispositivo específico que se refiere a 'TV'", 'tool': 'sumar', 'variables': {}, 'dependencias': ['recopilar datos']}, {'tarea': 'Calcular el consumo promedio diario de energía eléctrica durante febrero del 2024', 'tool': 'calcular_promedio', 'variables': {'consumo total': 'None', 'número de días en febrero': 'None'}, 'dependencias': ['identificar dispositivo y recopilar datos']}]
)
[TOOL_USE] Ejecutando consumo_rango_meses(dispositivo=TV, mes_inicio=2, mes_fin=2, año=2024)
[ERROR] Fallo al ejecutar tool 'sumar': sumar() missing 2 required positional arguments: 'a' and 'b'
[ERROR] Fallo al ejecutar tool 'calcular_promedio': calcular_promedio() got an unexpected keyword argument 'consumo total'
{'tarea': 'Recopilar datos de consumo de energía eléctrica para febrero del 2024', 'tool': 'consumo_rango_meses', 'variables': {'dispositivo': 'TV', 'mes_inicio': 2, 'mes_fin': 2, 'año': 2024, 'df':                TimeStamp  Ventilador   PC      AC  Lampara      TV
0    2024-01-01 00:00:00      0.0310  0.0  0.2189   0.0153  0.0160
1    2024-01-01 01:00:00      0.0262  0.0  0.5725   0.0025  0.0160
2    2024-01-01 02:00:00      0.0162  0.0  0.4512   0.0026  0.0156
3    2024-01-01 03:00:00      0.0269  0.0  0.4097   0.0012  0.0157
4    2024-01-01 04:00:00      0.0654  0.0  0.4086   0.0013  0.0160
...                  ...         ...  ...     ...      ...     ...
8779 2024-12-31 19:00:00      0.0163  0.0  0.4789   0.0189  0.0434
8780 2024-12-31 20:00:00      0.0162  0.0  0.4862   0.0164  0.0407
8781 2024-12-31 21:00:00      0.0163  0.0  0.3386   0.0184  0.0282
8782 2024-12-31 22:00:00      0.0580  0.0  0.3183   0.0049  0.0154
8783 2024-12-31 23:00:00      0.0706  0.0  0.3279   0.0000  0.0153

[8784 rows x 6 columns]}, 'resultado': np.float64(9.548300000000001)}
{'tarea': "Identificar el dispositivo específico que se refiere a 'TV'", 'tool': 'sumar', 'variables': {}, 'resultado': None, 'error': "sumar() missing 2 required positional arguments: 'a' and 'b'"}
{'tarea': 'Calcular el consumo promedio diario de energía eléctrica durante febrero del 2024', 'tool': 'calcular_promedio', 'variables': {'consumo total': 'None', 'número de días en febrero': 'None'}, 'resultado': None, 'error': "calcular_promedio() got an unexpected keyword argument 'consumo total'"}
(agentes-vs-llm) PS C:\Users\Nicolas\Documents\2025-2\Tesis-MAS-LLM>

'''