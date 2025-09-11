from Tools import sumar,restar,consumo_rango_horas,consumo_rango_dias,consumo_rango_meses,calcular_min,calcular_max,calcular_promedio
from Tools import tools_catalogo
from cargar_CSV import cargar_dataset_sinselejo
import dspy
import pandas as pd


########################################################################################################################  
#Contratos (me parece que es la mejor definición)

class Planificador(dspy.Signature):
    """
    Descompone una pregunta de usuario en subtareas claras y estructuradas,
    considerando el catálogo de herramientas disponibles.
    """

    pregunta: str = dspy.InputField(
        desc="Pregunta original del usuario en lenguaje natural sobre consumo energético"
    )

    tools_disponibles: list[dict] = dspy.InputField(
        desc=(
            "Catálogo de herramientas disponibles. Cada tool es un diccionario con:\n"
            "  - nombre (str)\n"
            "  - descripcion (str): explicación de qué hace y qué variables espera\n"
            "  - funcion (callable): función Python a ejecutar\n"
            "El Planificador debe usar este catálogo para decidir qué subtareas y secuencia generar."
        )
    )

    plan: list[dict] = dspy.OutputField(
        desc=(
            "Lista de subtareas en formato estrictamente definido. Cada subtarea es un diccionario con:\n"
            "  - id: identificador único de la subtarea (p.ej., 't1', 't2')\n"
            "  - descripcion: explicación clara y concisa de la acción a realizar\n"
            "  - variables: diccionario con variables abstractas necesarias (p.ej., dispositivo, dia_inicio, dia_fin, mes, año, hora_inicio, hora_fin)\n"
            "  - dependencias: lista de ids de subtareas previas de las cuales depende esta"
        )
    )

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
        reporte_planificador = self.planificador(pregunta=pregunta,tools_disponibles=self.tools_catalogo)
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
                # Normalizar y validar dispositivo si aplica
                if "dispositivo" in variables:
                    variables["dispositivo"] = self._validar_dispositivo(variables["dispositivo"])

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
    #Función parche para evitar problemas de nombre con los dispositivos del dataset
    def _validar_dispositivo(self, nombre: str) -> str:
        """
        Normaliza y valida el nombre de un dispositivo.
        Lanza ValueError si el dispositivo no es válido.
        """
        NORMALIZACION_DISPOSITIVOS = {
            "ventilador": "Ventilador",
            "pc": "PC",
            "ac": "AC",
            "lampara": "Lampara",
            "tv": "TV"
        }

        nombre_lower = nombre.lower().strip()
        if nombre_lower in NORMALIZACION_DISPOSITIVOS:
            return NORMALIZACION_DISPOSITIVOS[nombre_lower]
        else:
            raise ValueError(f"Dispositivo inválido: {nombre}")



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


