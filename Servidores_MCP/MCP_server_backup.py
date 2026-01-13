import sys
import os

# Agregar directorio padre al path para importar módulos compartidos
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fastmcp import FastMCP
import pandas as pd
from PlayGround.cargar_CSV import cargar_dataset_sinselejo
# Initialize FastMCP server
mcp = FastMCP("MCP_server_tools")

# -------------------------
# Cargar dataset global
csv_path = os.path.join(os.path.dirname(__file__), "Energy Consumption in KWh of a Typical House Sincelejo Colombia.csv")
DATASET = cargar_dataset_sinselejo(csv_path)



#TOOLS
@mcp.tool(
    meta={
        "input_schema": {
            "type": "object",
            "properties": {
                "a": {"type": "number", "description": "Primer número a sumar."},
                "b": {"type": "number", "description": "Segundo número a sumar."}
            },
            "required": ["a", "b"]
        },
        "output_schema": {
            "type": "object",
            "properties": {"result": {"type": "number", "description": "Resultado de la suma."}},
            "x-fastmcp-wrap-result": True
        }
    }
)
def sumar(a: float, b: float) -> float:
    """Suma dos números y devuelve el resultado. Usar únicamente para realizar una suma simple."""
    print(f"[TOOL_USE] Ejecutando sumar({a}, {b})")
    return a + b

@mcp.tool(
    meta={
        "input_schema": {
            "type": "object",
            "properties": {
                "a": {"type": "number", "description": "Primer número a restar."},
                "b": {"type": "number", "description": "Segundo número a restar."}
            },
            "required": ["a", "b"]
        },
        "output_schema": {
            "type": "object",
            "properties": {"result": {"type": "number", "description": "Resultado de la resta."}},
            "x-fastmcp-wrap-result": True
        }
    }
)
def restar(a: float, b: float) -> float:
    """Resta el segundo número al primero y devuelve el resultado. Usar únicamente para realizar una resta simple."""
    print(f"[TOOL_USE] Ejecutando restar({a}, {b})")
    return a - b

@mcp.tool(
    meta={
        "input_schema": {
            "type": "object",
            "properties": {
                "a": {"type": "number", "description": "Primer número a multiplicar."},
                "b": {"type": "number", "description": "Segundo número a multiplicar."}
            },
            "required": ["a", "b"]
        },
        "output_schema": {
            "type": "object",
            "properties": {"result": {"type": "number", "description": "Producto de la multiplicación."}},
            "x-fastmcp-wrap-result": True
        }
    }
)
def multiplicar(a: float, b: float) -> float:
    """Multiplica dos números y devuelve el resultado."""
    print(f"[TOOL_USE] Ejecutando multiplicar({a}, {b})")
    return a * b

@mcp.tool(
    meta={
        "input_schema": {
            "type": "object",
            "properties": {
                "a": {"type": "number", "description": "Dividendo."},
                "b": {"type": "number", "description": "Divisor."}
            },
            "required": ["a", "b"]
        },
        "output_schema": {
            "type": "object",
            "properties": {"result": {"type": "number", "description": "Resultado de la división."}},
            "x-fastmcp-wrap-result": True
        }
    }
)
def dividir(a: float, b: float) -> float:
    """Divide el primer número entre el segundo y devuelve el resultado. No usar si b=0."""
    print(f"[TOOL_USE] Ejecutando dividir({a}, {b})")
    if b == 0:
        raise ValueError("No se puede dividir por cero")
    return a / b

@mcp.tool(
    meta={
        "input_schema": {
            "type": "object",
            "properties": {"valores": {"type": "array", "items": {"type": "number"}, "description": "Lista de números."}},
            "required": ["valores"]
        },
        "output_schema": {
            "type": "object",
            "properties": {"result": {"type": "number", "description": "Valor mínimo encontrado."}},
            "x-fastmcp-wrap-result": True
        }
    }
)
def calcular_min(valores: list[float]) -> float:
    """Devuelve el valor mínimo de una lista de números."""
    print(f"[TOOL_USE] Ejecutando calcular_min({valores})")
    return min(valores)

@mcp.tool(
    meta={
        "input_schema": {
            "type": "object",
            "properties": {"valores": {"type": "array", "items": {"type": "number"}, "description": "Lista de números."}},
            "required": ["valores"]
        },
        "output_schema": {
            "type": "object",
            "properties": {"result": {"type": "number", "description": "Valor máximo encontrado."}},
            "x-fastmcp-wrap-result": True
        }
    }
)
def calcular_max(valores: list[float]) -> float:
    """Devuelve el valor máximo de una lista de números."""
    print(f"[TOOL_USE] Ejecutando calcular_max({valores})")
    return max(valores)

@mcp.tool(
    meta={
        "input_schema": {
            "type": "object",
            "properties": {"valores": {"type": "array", "items": {"type": "number"}, "description": "Lista de números."}},
            "required": ["valores"]
        },
        "output_schema": {
            "type": "object",
            "properties": {"result": {"type": "number", "description": "Promedio de los valores."}},
            "x-fastmcp-wrap-result": True
        }
    }
)
def calcular_promedio(valores: list[float]) -> float:
    """Calcula y devuelve el promedio de una lista de números."""
    print(f"[TOOL_USE] Ejecutando calcular_promedio({valores})")
    return sum(valores) / len(valores)

@mcp.tool(
    meta={
        "input_schema": {
            "type": "object",
            "properties": {
                "dispositivo": {"type": "string"},
                "hora_inicio": {"type": "integer"},
                "hora_fin": {"type": "integer"},
                "dia": {"type": "integer"},
                "mes": {"type": "integer"},
                "anio": {"type": "integer"}
            },
            "required": ["dispositivo", "hora_inicio", "hora_fin", "dia", "mes", "anio"]
        },
        "output_schema": {
            "type": "object",
            "properties": {"result": {"type": "number", "description": "Consumo total en kWh."}},
            "x-fastmcp-wrap-result": True
        }
    }
)
def consumo_rango_horas(dispositivo: str, hora_inicio: int, hora_fin: int, dia:int, mes:int, anio:int ) -> float:
    """Calcula el consumo total de un dispositivo en un rango específico de horas de un día dado."""
    df_dispositivo = DATASET[["TimeStamp", dispositivo]].copy()
    df_dispositivo = df_dispositivo[
        (df_dispositivo["TimeStamp"].dt.day == dia) & 
        (df_dispositivo["TimeStamp"].dt.month == mes) & 
        (df_dispositivo["TimeStamp"].dt.year == anio)
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

@mcp.tool(
    meta={
        "input_schema": {
            "type": "object",
            "properties": {
                "dispositivo": {"type": "string"},
                "dia_inicio": {"type": "integer"},
                "dia_fin": {"type": "integer"},
                "mes": {"type": "integer"},
                "anio": {"type": "integer"}
            },
            "required": ["dispositivo", "dia_inicio", "dia_fin", "mes", "anio"]
        },
        "output_schema": {
            "type": "object",
            "properties": {"result": {"type": "number", "description": "Consumo total en kWh."}},
            "x-fastmcp-wrap-result": True
        }
    }
)
def consumo_rango_dias(dispositivo: str, dia_inicio: int, dia_fin: int, mes: int, anio: int) -> float:
    """Calcula el consumo total de un dispositivo en un rango de días dentro de un mes y año."""
    df_dispositivo = DATASET[["TimeStamp", dispositivo]].copy()
    df_filtrado = df_dispositivo[
        (df_dispositivo["TimeStamp"].dt.year == anio) & 
        (df_dispositivo["TimeStamp"].dt.month == mes) & 
        (df_dispositivo["TimeStamp"].dt.day >= dia_inicio) & 
        (df_dispositivo["TimeStamp"].dt.day <= dia_fin)
    ]
    if df_filtrado.empty:
        print("No se encontraron registros para ese rango de días.")
        return 0.0
    consumo_total = df_filtrado[dispositivo].sum()
    return consumo_total

@mcp.tool(
    meta={
        "input_schema": {
            "type": "object",
            "properties": {
                "dispositivo": {"type": "string"},
                "mes_inicio": {"type": "integer"},
                "mes_fin": {"type": "integer"},
                "anio": {"type": "integer"}
            },
            "required": ["dispositivo", "mes_inicio", "mes_fin", "anio"]
        },
        "output_schema": {
            "type": "object",
            "properties": {"result": {"type": "number", "description": "Consumo total en kWh."}},
            "x-fastmcp-wrap-result": True
        }
    }
)
def consumo_rango_meses(dispositivo: str, mes_inicio: int, mes_fin: int, anio: int) -> float:
    """Calcula el consumo total de un dispositivo en un rango de meses dentro de un año."""
    df_dispositivo = DATASET[["TimeStamp", dispositivo]].copy()
    df_filtrado = df_dispositivo[
        (df_dispositivo["TimeStamp"].dt.year == anio) & 
        (df_dispositivo["TimeStamp"].dt.month >= mes_inicio) & 
        (df_dispositivo["TimeStamp"].dt.month <= mes_fin)
    ]
    if df_filtrado.empty:
        print("No se encontraron registros para ese rango de meses.")
        return 0.0
    consumo_total = df_filtrado[dispositivo].sum()
    return consumo_total

@mcp.tool(
    meta={
        "input_schema": {
            "type": "object",
            "properties": {"razon": {"type": "string", "description": "Descripción del motivo por el cual el plan es inviable."}},
            "required": ["razon"]
        },
        "output_schema": {
            "type": "object",
            "properties": {"result": {"type": "string", "description": "Mensaje de plan inviable."}},
            "x-fastmcp-wrap-result": True
        }
    }
)
def plan_inviable(razon: str) -> str:
    """Se usa cuando la consulta no puede resolverse con las herramientas disponibles o excede las capacidades del sistema."""
    print(f"[TOOL_USE] Plan marcado como inviable. Razón: {razon}")
    return f"Plan inviable: {razon}"

@mcp.tool(
    meta={
        "input_schema": {
            "type": "object",
            "properties": {"datos_faltante": {"type": "string", "description": "Datos faltantes requeridos para la consulta."}},
            "required": ["datos_faltante"]
        },
        "output_schema": {
            "type": "object",
            "properties": {"result": {"type": "string", "description": "Mensaje informando los datos faltantes."}},
            "x-fastmcp-wrap-result": True
        }
    }
)
def falta_informacion(datos_faltante: str) -> str:
    """Se usa cuando no hay suficiente información en la entrada del usuario para ejecutar la consulta."""
    print(f"[TOOL_USE] Faltan datos: {datos_faltante}")
    return f"Faltan datos: {datos_faltante}"

# =====================================================
# Recursos de contexto del hogar inteligente Sincelejo
# =====================================================

@mcp.resource(
    name="dispositivos_hogar_sincelejo",
    description="Lista de dispositivos monitorizados en el hogar de Sincelejo (Colombia).",
    mime_type="application/json",
    uri="mcp://hogar/dispositivos"
)
def dispositivos_hogar_sincelejo():
    """Recurso informativo que describe los dispositivos conectados al sistema energético."""
    return {
        "dispositivos": [
            {"nombre": "Ventilador", "ubicacion": "Habitación principal", "tipo": "Electrodoméstico"},
            {"nombre": "PC", "ubicacion": "Estudio", "tipo": "Equipo electrónico"},
            {"nombre": "Aire acondicionado", "ubicacion": "Sala", "tipo": "Climatización"},
            {"nombre": "Lámpara", "ubicacion": "Sala", "tipo": "Iluminación"},
            {"nombre": "Televisor", "ubicacion": "Sala", "tipo": "Entretenimiento"}
        ]
    }

@mcp.resource(
    name="contexto_local_sincelejo",
    description="Contexto general del hogar ubicado en Sincelejo: ubicación, condiciones y tiempo local.",
    mime_type="application/json",
    uri="mcp://hogar/contexto_local"
)
def contexto_local_sincelejo():
    """Proporciona información contextual y temporal del entorno doméstico."""
    return {
        "ubicacion": {
            "ciudad": "Sincelejo",
            "pais": "Colombia",
            "zona_horaria": "America/Bogota"
        },
        "clima_actual": {
            "temperatura_promedio": "32°C",
            "humedad": "70%",
            "condiciones": "Soleado"
        },
        "tiempo_sistema": {
            "hora_local": "14:30",
            "fecha": "2025-11-09",
            "dia_semana": "Domingo"
        }
    }


@mcp.resource(
    name="familia_sincelejo",
    description="Información general sobre los habitantes del hogar, sin incluir datos personales sensibles.",
    mime_type="application/json",
    uri="mcp://hogar/familia"
)
def familia_sincelejo():
    """Describe de forma general la composición del hogar y hábitos de uso energético."""
    return {
        "miembros": [
            {"rol": "Padre", "edad_aprox": 40, "ocupacion": "Ingeniero"},
            {"rol": "Madre", "edad_aprox": 38, "ocupacion": "Docente"},
            {"rol": "Hijo", "edad_aprox": 14, "ocupacion": "Estudiante"},
            {"rol": "Hija", "edad_aprox": 8, "ocupacion": "Estudiante"}
        ],
        "habitos_generales": [
            "Uso frecuente del aire acondicionado en la tarde.",
            "Televisor encendido durante la noche.",
            "PC activo en horario laboral.",
            "Lámpara encendida al anochecer."
        ]
    }


if __name__ == "__main__":
    mcp.run(transport="sse") #Creo que por default se crea con los parámetros de red host='0.0.0.0', port=3000

