


import pandas as pd
import matplotlib.pyplot as plt

# ==============================
# 1. Cargar y preparar datos
# ==============================

def cargar_dataset(file_path : str) -> pd.DataFrame:
    """Carga el dataset limpio y devuelve un DataFrame."""
    df = pd.read_csv(file_path, delimiter=";")
    df["TimeStamp"] = pd.to_datetime(df["TimeStamp"], errors="coerce") #convierte en objetos datetime de pandas la columna de timestamp y en NaT (Not a Time) los valores sin registro de tiempo, gracias a errors="coerce"
    return df
def tratemiento1(df:pd.DataFrame)-> pd.DataFrame:
    df=df.copy()
    df.sort_values(by="TimeStamp", ascending=True, na_position="last") # Ordenar por tiempo 
    df = df.dropna(subset=["TimeStamp"]) #Retira los datos sin etiqueta de tiempo
    df = df.drop_duplicates(subset=["TimeStamp"]) # Eliminar duplicados de tiempo, sensor y valor al mismo tiempo
    return df

def consumo_por_hora(df: pd.DataFrame, dispositivo: str, dia:int, mes:int, año:int ,hora_inicio: int, hora_fin: int) -> float:
    """
    Calcula el consumo en kWh de un dispositivo en un rango de horas específico de un día.

    Parámetros:
    -----------
    df : pd.DataFrame
        Dataset con columnas ["TimeStamp", "Device", "Value"] o ["TimeStamp", dispositivo1, dispositivo2, ...].
    dispositivo : str
        Nombre del dispositivo (ej. "PC", "TV", "Ventilador").
    dia : int
        Día específico (ej. 15).
    mes : int
        Mes específico (ej. 2 para febrero).
    año : int
        Año específico (ej. 2024).
    hora_inicio : int
        Hora inicial en formato 24h (ej. 12 para 12:00).
    hora_fin : int
        Hora final en formato 24h (ej. 18 para 18:00).

    Retorna:
    --------
    float : Consumo total en kWh en el rango de horas.
    """
    if "TimeStamp" not in df.columns:
        raise ValueError("El DataFrame no contiene la columna 'TimeStamp'.")

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
    print(df_filtrado)
    # Calcular el consumo total
    consumo_total = df_filtrado[dispositivo].sum()

    return consumo_total






df=tratemiento1(cargar_dataset("Energy Consumption in KWh of a Typical House Sincelejo Colombia.csv"))
#consumo_por_hora(df)
consumo_pc_tarde = consumo_por_hora(df, dispositivo="PC",mes=1,año=2024,dia=2,hora_inicio= 6,hora_fin= 12)
print(consumo_pc_tarde)