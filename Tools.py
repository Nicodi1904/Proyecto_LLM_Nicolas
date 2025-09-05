import pandas as pd
#Se definen las funciones propias de python que el agente usará
def sumar(a: int, b: int) -> int:
    print(f"[TOOL_USE] Ejecutando sumar({a}, {b})") #Mensaje para saber si la tool fue usada
    return a + b

def restar(a: int, b: int) -> int:
    print(f"[TOOL_USE] Ejecutando restar({a}, {b})") #Mensaje para saber si la tool fue usada
    return a - b

def consumo_rango_horas(df: pd.DataFrame, dispositivo: str ,hora_inicio: int, hora_fin: int, dia:int, mes:int, año:int ) -> float:
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

