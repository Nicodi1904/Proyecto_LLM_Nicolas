import pandas as pd
import matplotlib.pyplot as plt
from cargar_CSV import cargar_dataset_sinselejo
from Tools import consumo_rango_horas,consumo_rango_dias,consumo_rango_meses


data:pd.DataFrame=cargar_dataset_sinselejo("Energy Consumption in KWh of a Typical House Sincelejo Colombia.csv") 


def grafica_consumo_mes(df: pd.DataFrame, dispositivo: str, mes: int, año: int):
    """
    Genera una gráfica de consumo diario de un dispositivo para un mes específico.

    Parámetros:
    -----------
    df : pd.DataFrame
        Dataset con columnas ["TimeStamp", "Ventilador", "PC", "AC", "Lampara", "TV"].
    dispositivo : str
        Nombre de la columna del dispositivo a graficar.
    mes : int
        Mes (1-12) para filtrar los datos.
    año : int
        Año para filtrar los datos.

    Retorna:
    --------
    None (muestra la gráfica)
    """

    if dispositivo not in df.columns:
        raise ValueError(f"No existe la columna de dispositivo '{dispositivo}'. Columnas disponibles: {list(df.columns)}")

    # Filtrar por mes y año
    df_filtrado = df[(df["TimeStamp"].dt.month == mes) & (df["TimeStamp"].dt.year == año)]

    if df_filtrado.empty:
        print("No hay datos para ese mes y año.")
        return

    # Agrupar por día y sumar el consumo
    consumo_diario = df_filtrado.groupby(df_filtrado["TimeStamp"].dt.day)[dispositivo].sum()

    # Crear la gráfica
    plt.figure(figsize=(10, 5))
    plt.plot(consumo_diario.index, consumo_diario.values, marker='o', linestyle='-', color='blue')
    plt.title(f"Consumo diario de {dispositivo} - Mes {mes}/{año}")
    plt.xlabel("Día del mes")
    plt.ylabel("Consumo (kWh)")
    plt.xticks(range(1, 32))  # Mostrar todos los días del mes
    plt.grid(True)
    plt.show()

def grafica_consumo_dia(df: pd.DataFrame, dispositivo: str, dia: int, mes: int, año: int):
    """
    Genera una gráfica del consumo horario de un dispositivo para un día específico.

    Parámetros:
    -----------
    df : pd.DataFrame
        Dataset con columnas ["TimeStamp", "Ventilador", "PC", "AC", "Lampara", "TV"].
    dispositivo : str
        Nombre de la columna del dispositivo a graficar.
    dia : int
        Día del mes.
    mes : int
        Mes (1-12).
    año : int
        Año.

    Retorna:
    --------
    None (muestra la gráfica)
    """

    if dispositivo not in df.columns:
        raise ValueError(f"No existe la columna de dispositivo '{dispositivo}'. Columnas disponibles: {list(df.columns)}")

    # Filtrar por el día, mes y año seleccionados
    df_filtrado = df[
        (df["TimeStamp"].dt.day == dia) &
        (df["TimeStamp"].dt.month == mes) &
        (df["TimeStamp"].dt.year == año)
    ]

    if df_filtrado.empty:
        print("No hay datos para esa fecha.")
        return

    # Agrupar por hora y sumar el consumo
    consumo_horario = df_filtrado.groupby(df_filtrado["TimeStamp"].dt.hour)[dispositivo].sum()

    # Crear la gráfica
    plt.figure(figsize=(10, 5))
    plt.plot(consumo_horario.index, consumo_horario.values, marker='o', linestyle='-', color='green')
    plt.title(f"Consumo horario de {dispositivo} - {dia}/{mes}/{año}")
    plt.xlabel("Hora del día")
    plt.ylabel("Consumo (kWh)")
    plt.xticks(range(0, 24))  # 0 a 23 horas
    plt.grid(True)
    plt.show()


print(consumo_rango_horas(data,"AC",0,24,1,1,2024))
print(consumo_rango_dias(data,"AC",0,31,1,2024))
print(consumo_rango_meses(data,"AC",0,12,2024))



grafica_consumo_dia(data,"PC",14,2,2024)
grafica_consumo_dia(data,"Ventilador",14,2,2024)
grafica_consumo_dia(data,"AC",14,2,2024)
grafica_consumo_dia(data,"Lampara",14,2,2024)
grafica_consumo_dia(data,"TV",14,2,2024)