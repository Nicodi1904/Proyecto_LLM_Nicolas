from PlayGround.Tools import consumo_rango_dias,consumo_rango_horas,sumar
from PlayGround.Tools import tools_catalogo
from PlayGround.cargar_CSV import cargar_dataset_sinselejo
#El gerente se ha de encargar de dar la lista  de tools a ser procesadas, pero eso ya hace el planeador entonces por que es necesario?
#El planeador ya se encarga de hacer una lista de los procesos a seguir y asignar las variables a las tools, sin embargo tenemos el problema de las tareas con dependencias
#Pienso que luego de salir del planeador, hay que pasar por código de python que realice las tareas sin dependencias, a esas tareas podrá acceder el gerente y así dar una nueva lista con las nuevas tareas a realizar

df=cargar_dataset_sinselejo("Energy Consumption in KWh of a Typical House Sincelejo Colombia.csv")

ejemplo_plan=[
            {"id": 0, "funcion": "consumo_rango_horas", "desc": "Calcular consumo del televisor entre 14 y 20 horas del 10 de mayo de 2024", "dependencias": {"dispositivo": "TV", "hora_inicio": int(14), "hora_fin": int(20), "dia": int(10), "mes": int(5), "año": int(2024)}},
            {"id": 1, "funcion": "consumo_rango_horas", "desc": "Calcular consumo de la lampara entre 14 y 20 horas del 10 de mayo de 2024", "dependencias": {"dispositivo": "Lampara", "hora_inicio": int(14), "hora_fin": int(20), "dia": int(10), "mes": int(5), "año": int(2024)}},
            {"id": 2, "funcion": "sumar", "desc": "Sumar los consumos del televisor y la consola", "dependencias": {"a": "@0", "b": "@1"}}
        ]
ejemplo_plan2=[
            {"id": 0, "funcion": "calcular_min", "desc": "Obtener el mínimo de [3.5, 7.8, 2.1]", "dependencias": {"valores": [3.5, 7.8, 2.1]}}
        ]
ejemplo_plan3=plan=[
            {"id": 0, "funcion": "calcular_max", "desc": "Obtener el máximo de [12.5, 8.3, 15.9, 11.1]", "dependencias": {"valores": [12.5, 8.3, 15.9, 11.1]}}
        ]
ejemplo_plan4=plan=[
            {"id": 0, "funcion": "consumo_rango_dias", "desc": "Calcular consumo de la lavadora del 1 al 7 de abril de 2024", "dependencias": {"dispositivo": "AC", "dia_inicio": int(1), "dia_fin": int(7), "mes": int(4), "año": int(2024)}},
            {"id": 1, "funcion": "consumo_rango_dias", "desc": "Calcular consumo de la secadora del 1 al 7 de abril de 2024", "dependencias": {"dispositivo": "Ventilador", "dia_inicio": int(1), "dia_fin": int(7), "mes": int(4), "año": int(2024)}},
            {"id": 2, "funcion": "restar", "desc": "Restar el consumo de la secadora al de la lavadora", "dependencias": {"a": "@0", "b": "@1"}}
        ]

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

informe=worker(ejemplo_plan4,tools_catalogo,df)

print(informe)