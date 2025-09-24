from Tools import sumar,restar,consumo_rango_horas,consumo_rango_dias,consumo_rango_meses,calcular_min,calcular_max,calcular_promedio
from Tools import tools_catalogo
from Wrapped_Tools import fewshot_ejemplos
from cargar_CSV import cargar_dataset_sinselejo
import dspy
import pandas as pd


########################################################################################################################  
#Contratos o guías (me parece que son las mejores definiciones)

class Signature_Planificador(dspy.Signature):
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
            "Se debe usar este catálogo para decidir qué subtareas y secuencia generar así como los argumentos que se le pasan a la función."
        )
    )
    planeacion: str = dspy.OutputField(
        desc="Explicación breve en lenguaje natural de la estrategia y uso de tools antes del plan."
    )


    plan: list[dict] = dspy.OutputField(
        desc=(
            "Lista de subtareas estrictamente definida. Cada subtarea es un diccionario con:\n"
            "  - id: identificador único entero (0, 1, 2...)\n"
            "  - funcion: nombre exacto de la tool\n"
            "  - desc: explicación breve\n"
            "  - dependencias: diccionario con parámetros de entrada.\n"
            "    * Los nombres de las claves deben ser EXACTAMENTE los argumentos de la función. Nunca inventar nombres. \n"
            "    * Los valores pueden ser directos o referencias a procesos anteriores usando '@id'."
        )
)

class Signature_Gerente(dspy.Signature):
    """El gerente toma la planeación y los resultados del worker, y genera una respuesta clara y comprensible para el usuario final."""
    
    pregunta_usuario=dspy.InputField(dtype=str,desc="Pregunta o solicitud hecha por el usuario")
    planeacion = dspy.InputField(dtype=str,desc="Explicación del plan que el planeador generó según la intención del usuario.")
    informe_worker = dspy.InputField(dtype=dict,desc="Diccionario con los pasos ejecutados: id, descripción y resultado de cada tool.")

    respuesta_usuario = dspy.OutputField(dtype=str,desc="Explicación en lenguaje natural, clara y resumida, con el resultado final y contexto.")

###############################################################################################################################
#WORKER

def worker(plan, tools_catalogo,df=None):
    
    #Se hace un diccionario donde se guardarán los resultados respectivos de cada proceso y datos relevantes del mismo
    resultados = {}
    #Es necesario colocar este bloque que revisa si tiene dependencias primero, porque si no, puede que se ejecute una función con dependencias lo que daría error
    for proceso in plan:
        #Pasamos los procesos 1 por uno y extraemos los datos relevantes
        id_paso = proceso["id"]
        nombre_tool = proceso["funcion"]
        dependencias = proceso["dependencias"]
        descripcion = proceso["desc"]

        #Revisamos qué procesos dependen de otros con el @ que mandó el planeador, si no dependen de ninguno entonces se dejan los mismos argumentos que tenía
        new_args = {}
        for var_key, var in dependencias.items():
            if isinstance(var, str) and var.startswith("@"):  
                #sí hay dependencia, entonces extraemos la id de la dependencia encontrada 
                ref_id = int(var[1:]) #extraemos todo menos el @, ha de ser un número con la id, por eso int
                new_args[var_key] = resultados[ref_id]["resultado"] #guardamos el nuevo diccionario que tendrá los resultados de las dependencias
            else:
                new_args[var_key] = var #En caso de que no haya un @, osea no hayan dependencias los argumentos permanecen iguales


        #Se ejecuta las función mencionada en la lista dada mediante el catálogo
        
        funcion = tools_catalogo[nombre_tool]["funcion"]

        if "df" in funcion.__code__.co_varnames: # pero antes verificamos si la función necesita la base de datos "df"
            new_args["df"] = df

        resultado = funcion(**new_args)

        # Guardar salida con id + desc + resultado
        resultados[id_paso] = {
            "desc": descripcion,
            "resultado": resultado
        }

    return resultados

##############################################################################################################################
#Módulo principal

class Agente(dspy.Module):
    #Se declaran variables y funciones propias que el agente puede tener
    def __init__(self, tools_catalogo: list[dict], df: pd.DataFrame = None): #el dataframe puede que no siempre se solicite, entonces se inicializa en none
        #Se hereda clase padre módulo de dspy
        super().__init__()
        #Se cargan los roles de los LLMs
        self.planificador = dspy.ChainOfThought(Signature_Planificador)
        trainer = dspy.BootstrapFewShot()
        self.planificador_esp = trainer.compile(
            student=dspy.Predict(Signature_Planificador),
            trainset=fewshot_ejemplos
        )
        self.gerente = dspy.ChainOfThought(Signature_Gerente)
        #Cargar catalogo de tools (Se podría decir que esto ya cuenta como módulo de RAG...verdad?)
        self.tools_catalogo = tools_catalogo
        #Se cargan los datos
        self.df = df
    
    def __call__(self, pregunta: str):
        # 1. Planificador
        salida_planificador = self.planificador_esp(pregunta=pregunta,tools_disponibles=self.tools_catalogo)
        planeacion = salida_planificador.planeacion
        plan = salida_planificador.plan
        print("------------------------------------------------------------\nPlaneación LLM\n--------------------------------------")
        print(planeacion,"\n--------------------------------------")
        print("------------------------------------------------------------\nPlan LLM\n--------------------------------------")
        print(plan,"\n--------------------------------------")
        # 2. Worker
        informe_worker = worker(plan, self.tools_catalogo, self.df)
        # 3. Gerente
        respuesta = self.gerente(pregunta_usuario=pregunta,planeacion=str(planeacion), informe_worker=informe_worker).respuesta_usuario

        return respuesta

##############################################################################################################################

df=cargar_dataset_sinselejo("Energy Consumption in KWh of a Typical House Sincelejo Colombia.csv")

# ---------------------- PRUEBA CON LLAMA3.1 ----------------------
print("\n========== Prueba con Llama3.1 ==========")

lm_llama = dspy.LM('ollama_chat/llama3.1', api_base='http://localhost:11434', api_key='')
dspy.configure(lm=lm_llama)

# Crear agente con llama3.1
agente_llama = Agente(tools_catalogo=tools_catalogo, df=df)

print("\n[PREGUNTA 1:¿Cuánto consumió el AC entre las 8 am y 5 pm del 15 de enero del 2024?]")
resultado1 = agente_llama("hola cómo estás, ¿Cuánto consumió el AC entre las 8 am y 8 pm del 10 de enero del 2024?, me urge saber porque mi mamá me lo está preguntando")
print(resultado1)

print("\n[PREGUNTA 2: ¿Cuánto consumió el AC entre las 8 am y 5 pm del 15 de enero del 2024? y el TV en ese mismo rango de tiempo, cuál consumió más? ")
resultado2 = agente_llama("¿Cuánto consumió el AC entre las 8 am y 5 pm del 15 de enero del 2024? y el TV en ese mismo rango de tiempo, cuál consumió más? ")
print(resultado2)
 
# ---------------------- PRUEBA CON tinyllama ----------------------
""" print("\n========== Prueba con tinyllama ==========")

lm_deepseek = dspy.LM('ollama_chat/tinyllama', api_base='http://localhost:11434', api_key='')
dspy.configure(lm=lm_deepseek)

# Crear agente con tinyllama
agente_deepseek = Agente(tools_catalogo=tools_catalogo, df=df)

print("\n[PREGUNTA 1: ¿Cuánto consumió el AC entre las 8 am y 5 pm del 15 de enero del 2024?]")
resultado3 = agente_deepseek("¿Cuánto consumió el AC entre las 8 am y 5 pm del 15 de enero del 2024?")
print(resultado3) """


""" # ---------------------- PRUEBA CON DEEPSEEK-R1:8B ----------------------
print("\n========== Prueba con Deepseek-R1:8B ==========")

lm_deepseek = dspy.LM('ollama_chat/deepseek-r1:8b', api_base='http://localhost:11434', api_key='')
dspy.configure(lm=lm_deepseek)

# Crear agente con deepseek-r1:8b
agente_deepseek = Agente(tools_catalogo=tools_catalogo, df=df)

print("\n[PREGUNTA 1: ¿Cuánto consumió el AC entre las 8 am y 5 pm del 15 de enero del 2024? ")
resultado3 = agente_deepseek("¿Cuánto consumió el AC entre las 8 am y 5 pm del 15 de enero del 2024?")
print(resultado3)

 """