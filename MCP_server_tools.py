from mcp.server.fastmcp import FastMCP
import pandas as pd
from cargar_CSV import cargar_dataset_sinselejo
# Initialize FastMCP server
mcp = FastMCP("MCP_server_tools")

# -------------------------
# Cargar dataset global
# -------------------------
DATASET=cargar_dataset_sinselejo("Energy Consumption in KWh of a Typical House Sincelejo Colombia.csv") 

@mcp.tool()
def sumar(a: float, b: float) -> float:
    """Suma dos números y devuelve el resultado. Usar únicamente para realizar una suma simple."""
    print(f"[TOOL_USE] Ejecutando sumar({a}, {b})")
    return a + b

@mcp.tool()
def restar(a: float, b: float) -> float:
    """Resta el segundo número al primero y devuelve el resultado. Usar únicamente para realizar una resta simple."""
    print(f"[TOOL_USE] Ejecutando restar({a}, {b})")
    return a - b

@mcp.tool()
def multiplicar(a: float, b: float) -> float:
    """Multiplica dos números y devuelve el resultado."""
    print(f"[TOOL_USE] Ejecutando multiplicar({a}, {b})")
    return a * b

@mcp.tool()
def dividir(a: float, b: float) -> float:
    """Divide el primer número entre el segundo y devuelve el resultado. No usar si b=0."""
    print(f"[TOOL_USE] Ejecutando dividir({a}, {b})")
    if b == 0:
        raise ValueError("No se puede dividir por cero")
    return a / b

@mcp.tool()
def calcular_min(valores: list[float]) -> float:
    """Devuelve el valor mínimo de una lista de números."""
    print(f"[TOOL_USE] Ejecutando calcular_min({valores})")
    return min(valores)

@mcp.tool()
def calcular_max(valores: list[float]) -> float:
    """Devuelve el valor máximo de una lista de números."""
    print(f"[TOOL_USE] Ejecutando calcular_max({valores})")
    return max(valores)

@mcp.tool()
def calcular_promedio(valores: list[float]) -> float:
    """Calcula y devuelve el promedio de una lista de números."""
    print(f"[TOOL_USE] Ejecutando calcular_promedio({valores})")
    return sum(valores) / len(valores)

@mcp.tool()
def consumo_rango_horas(dispositivo: str, hora_inicio: int, hora_fin: int, dia:int, mes:int, año:int ) -> float:
    """Calcula el consumo total de un dispositivo en un rango específico de horas de un día dado."""
    df_dispositivo = DATASET[["TimeStamp", dispositivo]].copy()
    df_dispositivo = df_dispositivo[
        (df_dispositivo["TimeStamp"].dt.day == dia) &
        (df_dispositivo["TimeStamp"].dt.month == mes) &
        (df_dispositivo["TimeStamp"].dt.year == año)
    ]
    df_filtrado = df_dispositivo[
        (df_dispositivo["TimeStamp"].dt.hour >= hora_inicio) &
        (df_dispositivo["TimeStamp"].dt.hour < hora_fin)
    ]
    if df_filtrado.empty:
        print("No se encontraron registros para ese rango de horas.")
        return 0.0
    consumo_total = df_filtrado[dispositivo].sum()
    return consumo_total

@mcp.tool()
def consumo_rango_dias(dispositivo: str, dia_inicio: int, dia_fin: int, mes: int, año: int) -> float:
    """Calcula el consumo total de un dispositivo en un rango de días dentro de un mes y año."""
    df_dispositivo = DATASET[["TimeStamp", dispositivo]].copy()
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

@mcp.tool()
def consumo_rango_meses(dispositivo: str, mes_inicio: int, mes_fin: int, año: int) -> float:
    """Calcula el consumo total de un dispositivo en un rango de meses dentro de un año."""
    df_dispositivo = DATASET[["TimeStamp", dispositivo]].copy()
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

@mcp.tool()
def plan_inviable(razon: str) -> str:
    """Se usa cuando la consulta no puede resolverse con las herramientas disponibles o excede las capacidades del sistema."""
    print(f"[TOOL_USE] Plan marcado como inviable. Razón: {razon}")
    return f"Plan inviable: {razon}"

@mcp.tool()
def falta_informacion(datos_faltante: str) -> str:
    """Se usa cuando no hay suficiente información en la entrada del usuario para ejecutar la consulta."""
    print(f"[TOOL_USE] Faltan datos: {datos_faltante}")
    return f"Faltan datos: {datos_faltante}"

if __name__ == "__main__":
    mcp.run()
