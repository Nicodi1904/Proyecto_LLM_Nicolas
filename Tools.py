import pandas as pd
#Se definen las funciones propias de python que el agente usará
def sumar(a: int, b: int) -> int:
    print(f"[TOOL_USE] Ejecutando sumar({a}, {b})") #Mensaje para saber si la tool fue usada
    return a + b

def restar(a: int, b: int) -> int:
    print(f"[TOOL_USE] Ejecutando restar({a}, {b})") #Mensaje para saber si la tool fue usada
    return a - b
def dividir(a: float, b: float) -> float:
    print(f"[TOOL_USE] Ejecutando dividir({a}, {b})")  # Mensaje para saber si la tool fue usada
    if b == 0:
        raise ValueError("No se puede dividir por cero")
    return a / b
def multiplicar(a: float, b: float) -> float:
    print(f"[TOOL_USE] Ejecutando multiplicar({a}, {b})")  # Mensaje para saber si la tool fue usada
    return a * b


def calcular_min(valores: list[float]) -> float:
    print(f"[TOOL_USE] Ejecutando calcular_min({valores})")
    return min(valores)


def calcular_max(valores: list[float]) -> float:
    print(f"[TOOL_USE] Ejecutando calcular_max({valores})")
    return max(valores)


def calcular_promedio(valores: list[float]) -> float:
    print(f"[TOOL_USE] Ejecutando calcular_promedio({valores})")
    return sum(valores) / len(valores)

def consumo_rango_horas(df: pd.DataFrame, dispositivo: str ,hora_inicio: int, hora_fin: int, dia:int, mes:int, año:int ) -> float:
    print(f"[TOOL_USE] Ejecutando consumo_rango_horas(dispositivo={dispositivo}, hora_inicio={hora_inicio}, hora_fin={hora_fin}, dia={dia}, mes={mes}, año={año})")
    
    if dispositivo not in df.columns:
        raise ValueError(f"No existe la columna de dispositivo '{dispositivo}' en el DataFrame. Columnas disponibles: {list(df.columns)}")
    
    # Filtrar el dispositivo
    df_dispositivo = df[["TimeStamp", dispositivo]].copy()

    # Filtrar por día, mes y año
    df_dispositivo = df_dispositivo[
        (df_dispositivo["TimeStamp"].dt.day == dia) &
        (df_dispositivo["TimeStamp"].dt.month == mes) &
        (df_dispositivo["TimeStamp"].dt.year == año)
    ]

    # Filtrar por rango horario
    df_filtrado = df_dispositivo[
        (df_dispositivo["TimeStamp"].dt.hour >= hora_inicio) &
        (df_dispositivo["TimeStamp"].dt.hour < hora_fin)
    ]
    if df_filtrado.empty:
        print("No se encontraron registros para ese rango de horas.")
        return 0.0

    # Calcular el consumo total
    consumo_total = df_filtrado[dispositivo].sum()

    return consumo_total


def consumo_rango_dias(df: pd.DataFrame, dispositivo: str, dia_inicio: int, dia_fin: int, mes: int, año: int) -> float:
    print(f"[TOOL_USE] Ejecutando consumo_rango_dias(dispositivo={dispositivo}, dia_inicio={dia_inicio}, dia_fin={dia_fin}, mes={mes}, año={año})")
    
    if dispositivo not in df.columns:
        raise ValueError(f"No existe la columna de dispositivo '{dispositivo}' en el DataFrame. Columnas disponibles: {list(df.columns)}")

    # Filtrar el dispositivo
    df_dispositivo = df[["TimeStamp", dispositivo]].copy()

    # Filtrar por rango de días dentro del mismo mes y año
    df_filtrado = df_dispositivo[
        (df_dispositivo["TimeStamp"].dt.year == año) &
        (df_dispositivo["TimeStamp"].dt.month == mes) &
        (df_dispositivo["TimeStamp"].dt.day >= dia_inicio) &
        (df_dispositivo["TimeStamp"].dt.day <= dia_fin)
    ]

    if df_filtrado.empty:
        print("No se encontraron registros para ese rango de días.")
        return 0.0

    consumo_total = df_filtrado[dispositivo].sum()
    return consumo_total


def consumo_rango_meses(df: pd.DataFrame, dispositivo: str, mes_inicio: int, mes_fin: int, año: int) -> float:
    print(f"[TOOL_USE] Ejecutando consumo_rango_meses(dispositivo={dispositivo}, mes_inicio={mes_inicio}, mes_fin={mes_fin}, año={año})")
    
    if dispositivo not in df.columns:
        raise ValueError(f"No existe la columna de dispositivo '{dispositivo}' en el DataFrame. Columnas disponibles: {list(df.columns)}")

    # Filtrar el dispositivo
    df_dispositivo = df[["TimeStamp", dispositivo]].copy()

    # Filtrar por rango de meses dentro del mismo año
    df_filtrado = df_dispositivo[
        (df_dispositivo["TimeStamp"].dt.year == año) &
        (df_dispositivo["TimeStamp"].dt.month >= mes_inicio) &
        (df_dispositivo["TimeStamp"].dt.month <= mes_fin)
    ]

    if df_filtrado.empty:
        print("No se encontraron registros para ese rango de meses.")
        return 0.0

    consumo_total = df_filtrado[dispositivo].sum()
    return consumo_total
    

def plan_inviable(razon: str) -> str:
    print(f"[TOOL_USE] Plan marcado como inviable. Razón: {razon}")
    return f"Plan inviable: {razon}"


def faltan_datos(dato_faltante: str) -> str:
    print(f"[TOOL_USE] Faltan datos: {dato_faltante}")
    return f"Faltan datos: {dato_faltante}"

#######-------------------------------------------------------CATALOGO---------------------------------------------###########
tools_catalogo = {
    "sumar": {
        "descripcion": "Suma dos números y devuelve el resultado. Usar únicamente para realizar una suma simple.",
        "variables_entrada": ["a: float", "b: float"],
        "variables_salida": ["resultado: float"],
        "funcion": sumar
    },
    "restar": {
        "descripcion": "Resta el segundo número al primero y devuelve el resultado. Usar únicamente para realizar una resta simple.",
        "variables_entrada": ["a: float", "b: float"],
        "variables_salida": ["resultado: float"],
        "funcion": restar
    },
    "multiplicar": {
        "descripcion": "Multiplica dos números y devuelve el resultado.",
        "variables_entrada": ["a: float", "b: float"],
        "variables_salida": ["resultado: float"],
        "funcion": multiplicar
    },
    "dividir": {
        "descripcion": "Divide el primer número entre el segundo y devuelve el resultado. No usar si b=0.",
        "variables_entrada": ["a: float", "b: float"],
        "variables_salida": ["resultado: float"],
        "funcion": dividir
    },
    "calcular_min": {
        "descripcion": "Devuelve el valor mínimo de una lista de números.",
        "variables_entrada": ["valores: list[float]"],
        "variables_salida": ["minimo: float"],
        "funcion": calcular_min
    },
    "calcular_max": {
        "descripcion": "Devuelve el valor máximo de una lista de números.",
        "variables_entrada": ["valores: list[float]"],
        "variables_salida": ["maximo: float"],
        "funcion": calcular_max
    },
    "calcular_promedio": {
        "descripcion": "Calcula y devuelve el promedio de una lista de números.",
        "variables_entrada": ["valores: list[float]"],
        "variables_salida": ["promedio: float"],
        "funcion": calcular_promedio
    },
    "consumo_rango_horas": {
        "descripcion": "Calcula el consumo total de un dispositivo en un rango específico de horas de un día dado.",
        "variables_entrada": [
            "dispositivo: str",
            "hora_inicio: int",
            "hora_fin: int",
            "dia: int",
            "mes: int",
            "año: int"
        ],
        "variables_salida": ["consumo_total: float"],
        "funcion": consumo_rango_horas
    },
    "consumo_rango_dias": {
        "descripcion": "Calcula el consumo total de un dispositivo en un rango de días dentro de un mes y año.",
        "variables_entrada": [
            "dispositivo: str",
            "dia_inicio: int",
            "dia_fin: int",
            "mes: int",
            "año: int"
        ],
        "variables_salida": ["consumo_total: float"],
        "funcion": consumo_rango_dias
    },
    "consumo_rango_meses": {
        "descripcion": "Calcula el consumo total de un dispositivo en un rango de meses dentro de un año.",
        "variables_entrada": [
            "dispositivo: str",
            "mes_inicio: int",
            "mes_fin: int",
            "año: int"
        ],
        "variables_salida": ["consumo_total: float"],
        "funcion": consumo_rango_meses
    },
    "plan_inviable": {
        "descripcion": "Se usa cuando la consulta no puede resolverse con las herramientas disponibles o excede las capacidades del sistema.",
        "variables_entrada": ["razon: str"],
        "variables_salida": ["mensaje: str"],
        "funcion": plan_inviable
    },
    "faltan_datos": {
        "descripcion": "Se usa cuando no hay suficiente información en la entrada del usuario para ejecutar la consulta.",
        "variables_entrada": ["dato_faltante: str"],
        "variables_salida": ["mensaje: str"],
        "funcion": faltan_datos
    }
}
