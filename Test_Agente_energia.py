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

#Prueba
