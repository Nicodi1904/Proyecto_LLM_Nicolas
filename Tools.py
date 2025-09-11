import pandas as pd
#Se definen las funciones propias de python que el agente usará
def sumar(a: int, b: int) -> int:
    print(f"[TOOL_USE] Ejecutando sumar({a}, {b})") #Mensaje para saber si la tool fue usada
    return a + b

def restar(a: int, b: int) -> int:
    print(f"[TOOL_USE] Ejecutando restar({a}, {b})") #Mensaje para saber si la tool fue usada
    return a - b

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
    

#######-------------------------------------------------------CATALOGO---------------------------------------------###########
tools_catalogo = [
    {
        "nombre": "sumar",
        "descripcion": "Suma dos números enteros y devuelve el resultado. Usar cuando se necesite realizar una suma simple.",
        "variables_entrada": ["a: int", "b: int"],
        "variables_salida": ["resultado: int"],
        "funcion":sumar
    },
    {
        "nombre": "restar",
        "descripcion": "Resta el segundo número entero del primero y devuelve el resultado. Usar cuando se necesite una resta simple.",
        "variables_entrada": ["a: int", "b: int"],
        "variables_salida": ["resultado: int"],
        "funcion":restar
    },
    {
        "nombre": "calcular_min",
        "descripcion": "Devuelve el valor mínimo de una lista de números flotantes. Usar para obtener el valor más bajo en una serie de datos.",
        "variables_entrada": ["valores: list[float]"],
        "variables_salida": ["minimo: float"],
        "funcion":calcular_min
    },
    {
        "nombre": "calcular_max",
        "descripcion": "Devuelve el valor máximo de una lista de números flotantes. Usar para obtener el valor más alto en una serie de datos.",
        "variables_entrada": ["valores: list[float]"],
        "variables_salida": ["maximo: float"],
        "funcion":calcular_max
    },
    {
        "nombre": "calcular_promedio",
        "descripcion": "Calcula y devuelve el promedio de una lista de números flotantes. Usar cuando se requiera el valor medio.",
        "variables_entrada": ["valores: list[float]"],
        "variables_salida": ["promedio: float"],
        "funcion":calcular_promedio
    },
    {
        "nombre": "consumo_rango_horas",
        "descripcion": "Calcula el consumo total de un dispositivo en un rango específico de horas de un día en particular.",
        "variables_entrada": [
            "dispositivo: str",
            "hora_inicio: int",
            "hora_fin: int",
            "dia: int",
            "mes: int",
            "año: int"
        ],
        "variables_salida": ["consumo_total: float"],
        "funcion":consumo_rango_horas
    },
    {
        "nombre": "consumo_rango_dias",
        "descripcion": "Calcula el consumo total de un dispositivo en un rango de días dentro de un mismo mes y año.",
        "variables_entrada": [
            "dispositivo: str",
            "dia_inicio: int",
            "dia_fin: int",
            "mes: int",
            "año: int"
        ],
        "variables_salida": ["consumo_total: float"],
        "funcion":consumo_rango_dias
    },
    {
        "nombre": "consumo_rango_meses",
        "descripcion": "Calcula el consumo total de un dispositivo en un rango de meses dentro de un mismo año.",
        "variables_entrada": [
            "dispositivo: str",
            "mes_inicio: int",
            "mes_fin: int",
            "año: int"
        ],
        "variables_salida": ["consumo_total: float"],
        "funcion":consumo_rango_meses
    }
]
