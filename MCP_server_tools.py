from fastmcp import FastMCP
import pandas as pd
from cargar_CSV import cargar_dataset_sinselejo

# Initialize FastMCP server
mcp = FastMCP("MCP_server_tools")

# -------------------------
# Cargar dataset global
# -------------------------
DATASET = cargar_dataset_sinselejo("Energy Consumption in KWh of a Typical House Sincelejo Colombia.csv")


#######################################################################################
common_math_meta = {
    "categoria": "matematica_auxiliar",
    "descripcion_funcional": (
        "Herramienta auxiliar de bajo nivel usada únicamente dentro de "
        "planes internos. Nunca debe emplearse para responder directamente "
        "consultas energéticas del usuario."
    ),
    "seleccion": {
        "usar_si": [
            "Otra herramienta del dominio energético ya filtró datos y requiere una operación matemática explícita"
        ],
        "no_usar_si": [
            "La intención del usuario involucra consumo energético",
            "La consulta puede resolverse con herramientas energéticas de rango-horas, rango-días o rango-meses",
            "La herramienta sería usada para interpretar la intención del usuario"
        ],
        "limitaciones": [
            "Es una herramienta de propósito general",
            "No contiene lógica energética",
        ],
        "proposito": "Permitir cálculos matemáticos simples dentro del planificador"
    }
}

#################################################################################################################################3

@mcp.tool(
    meta={
        **common_math_meta,
        "input_schema": {
            "type": "object",
            "properties": {
                "a": {"type": "number"},
                "b": {"type": "number"}
            },
            "required": ["a", "b"]
        },
        "output_schema": {
            "type": "object",
            "properties": {"result": {"type": "number"}},
            "x-fastmcp-wrap-result": True
        }
    }
)
def sumar(a: float, b: float) -> float:
    """Suma auxiliar."""
    print(f"[TOOL_USE] sumar({a}, {b})")
    return a + b


@mcp.tool(
    meta={
        **common_math_meta,
        "input_schema": {
            "type": "object",
            "properties": {
                "a": {"type": "number"},
                "b": {"type": "number"}
            },
            "required": ["a", "b"]
        },
        "output_schema": {
            "type": "object",
            "properties": {"result": {"type": "number"}},
            "x-fastmcp-wrap-result": True
        }
    }
)
def restar(a: float, b: float) -> float:
    """Resta auxiliar."""
    print(f"[TOOL_USE] restar({a}, {b})")
    return a - b


@mcp.tool(
    meta={
        **common_math_meta,
        "input_schema": {
            "type": "object",
            "properties": {
                "a": {"type": "number"},
                "b": {"type": "number"}
            },
            "required": ["a", "b"]
        },
        "output_schema": {
            "type": "object",
            "properties": {"result": {"type": "number"}},
            "x-fastmcp-wrap-result": True
        }
    }
)
def multiplicar(a: float, b: float) -> float:
    """Multiplicación auxiliar."""
    print(f"[TOOL_USE] multiplicar({a}, {b})")
    return a * b


@mcp.tool(
    meta={
        **common_math_meta,
        "input_schema": {
            "type": "object",
            "properties": {
                "a": {"type": "number"},
                "b": {"type": "number"}
            },
            "required": ["a", "b"]
        },
        "output_schema": {
            "type": "object",
            "properties": {"result": {"type": "number"}},
            "x-fastmcp-wrap-result": True
        }
    }
)
def dividir(a: float, b: float) -> float:
    """División auxiliar."""
    print(f"[TOOL_USE] dividir({a}, {b})")
    if b == 0:
        raise ValueError("No se puede dividir por cero")
    return a / b


@mcp.tool(
    meta={
        **common_math_meta,
        "input_schema": {
            "type": "object",
            "properties": {
                "valores": {"type": "array", "items": {"type": "number"}}
            },
            "required": ["valores"]
        },
        "output_schema": {
            "type": "object",
            "properties": {"result": {"type": "number"}},
            "x-fastmcp-wrap-result": True
        }
    }
)
def calcular_min(valores: list[float]) -> float:
    """Mínimo auxiliar."""
    print(f"[TOOL_USE] calcular_min({valores})")
    return min(valores)


@mcp.tool(
    meta={
        **common_math_meta,
        "input_schema": {
            "type": "object",
            "properties": {
                "valores": {"type": "array", "items": {"type": "number"}}
            },
            "required": ["valores"]
        },
        "output_schema": {
            "type": "object",
            "properties": {"result": {"type": "number"}},
            "x-fastmcp-wrap-result": True
        }
    }
)
def calcular_max(valores: list[float]) -> float:
    """Máximo auxiliar."""
    print(f"[TOOL_USE] calcular_max({valores})")
    return max(valores)


@mcp.tool(
    meta={
        **common_math_meta,
        "input_schema": {
            "type": "object",
            "properties": {
                "valores": {"type": "array", "items": {"type": "number"}}
            },
            "required": ["valores"]
        },
        "output_schema": {
            "type": "object",
            "properties": {"result": {"type": "number"}},
            "x-fastmcp-wrap-result": True
        }
    }
)
def calcular_promedio(valores: list[float]) -> float:
    """Promedio auxiliar."""
    print(f"[TOOL_USE] calcular_promedio({valores})")
    return sum(valores) / len(valores)


# =====================================================
# Herramientas del dominio energético
# =====================================================

def energy_meta(granularidad, usar_si, no_usar_si, limitaciones, proposito):
    return {
        "categoria": "energia",
        "granularidad": granularidad,
        "seleccion": {
            "usar_si": usar_si,
            "no_usar_si": no_usar_si,
            "limitaciones": limitaciones,
            "proposito": proposito
        }
    }


# ------------------- RANGO HORAS --------------------
@mcp.tool(
    meta={
        **energy_meta(
            granularidad="horaria",
            usar_si=[
                "El usuario pide consumo entre horas específicas",
                "La consulta corresponde a un único día",
                "La intención menciona horas explícitas"
            ],
            no_usar_si=[
                "No se mencionan horas",
                "El rango involucra más de un día",
                "El usuario pide consumo diario o mensual"
            ],
            limitaciones=[
                "La fecha debe existir en el dataset",
                "Devuelve 0 si no hay registros"
            ],
            proposito="Obtener el consumo energético exacto en una ventana horaria"
        ),
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
            "properties": {"result": {"type": "number"}},
            "x-fastmcp-wrap-result": True
        }
    }
)
def consumo_rango_horas(dispositivo: str, hora_inicio: int, hora_fin: int, dia: int, mes: int, anio: int) -> float:
    """Cálculo de consumo horario."""
    df = DATASET[["TimeStamp", dispositivo]].copy()
    df = df[
        (df["TimeStamp"].dt.day == dia) &
        (df["TimeStamp"].dt.month == mes) &
        (df["TimeStamp"].dt.year == anio)
    ]
    df = df[
        (df["TimeStamp"].dt.hour >= hora_inicio) &
        (df["TimeStamp"].dt.hour < hora_fin)
    ]
    if df.empty:
        return 0.0
    return df[dispositivo].sum()


# ------------------- RANGO DÍAS --------------------
@mcp.tool(
    meta={
        **energy_meta(
            granularidad="diaria",
            usar_si=[
                "El usuario pide consumo por días completos",
                "No se mencionan horas",
                "El rango está dentro de un mismo mes"
            ],
            no_usar_si=[
                "El rango cruza meses",
                "El usuario pide horas específicas",
                "El usuario pide consumo mensual"
            ],
            limitaciones=[
                "Días deben existir en el dataset",
                "Devuelve 0 si no hay registros"
            ],
            proposito="Obtener el consumo energético agregado por días completos"
        ),
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
            "properties": {"result": {"type": "number"}},
            "x-fastmcp-wrap-result": True
        }
    }
)
def consumo_rango_dias(dispositivo: str, dia_inicio: int, dia_fin: int, mes: int, anio: int) -> float:
    """Cálculo de consumo por días."""
    df = DATASET[["TimeStamp", dispositivo]].copy()
    df = df[
        (df["TimeStamp"].dt.year == anio) &
        (df["TimeStamp"].dt.month == mes) &
        (df["TimeStamp"].dt.day >= dia_inicio) &
        (df["TimeStamp"].dt.day <= dia_fin)
    ]
    if df.empty:
        return 0.0
    return df[dispositivo].sum()


# ------------------- RANGO MESES --------------------
@mcp.tool(
    meta={
        **energy_meta(
            granularidad="mensual",
            usar_si=[
                "El usuario pide consumo mensual o multimes",
                "No se requieren detalles diarios u horarios"
            ],
            no_usar_si=[
                "El rango cruza años",
                "El usuario pide días u horas"
            ],
            limitaciones=[
                "Meses deben existir en el dataset",
                "Devuelve 0 si no hay registros"
            ],
            proposito="Obtener el consumo energético agregado por meses completos"
        ),
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
            "properties": {"result": {"type": "number"}},
            "x-fastmcp-wrap-result": True
        }
    }
)
def consumo_rango_meses(dispositivo: str, mes_inicio: int, mes_fin: int, anio: int) -> float:
    """Cálculo de consumo mensual."""
    df = DATASET[["TimeStamp", dispositivo]].copy()
    df = df[
        (df["TimeStamp"].dt.year == anio) &
        (df["TimeStamp"].dt.month >= mes_inicio) &
        (df["TimeStamp"].dt.month <= mes_fin)
    ]
    if df.empty:
        return 0.0
    return df[dispositivo].sum()


# ------------------- TOOLS ESPECIALES --------------------

@mcp.tool(
    meta={
        "categoria": "control",
        "seleccion": {
            "usar_si": ["La consulta no puede resolverse con ninguna herramienta disponible"],
            "no_usar_si": [],
            "limitaciones": ["Debe incluir una razón clara"],
            "proposito": "Marcar explícitamente un plan como inviable"
        },
        "input_schema": {
            "type": "object",
            "properties": {
                "razon": {"type": "string"}
            },
            "required": ["razon"]
        },
        "output_schema": {
            "type": "object",
            "properties": {"result": {"type": "string"}},
            "x-fastmcp-wrap-result": True
        }
    }
)
def plan_inviable(razon: str) -> str:
    """Marca un plan como inviable."""
    print(f"[TOOL_USE] Plan inviable: {razon}")
    return f"Plan inviable: {razon}"


@mcp.tool(
    meta={
        "categoria": "control",
        "seleccion": {
            "usar_si": ["Faltan datos esenciales para ejecutar la consulta"],
            "no_usar_si": [],
            "limitaciones": [],
            "proposito": "Informar al usuario o al sistema que faltan parámetros obligatorios"
        },
        "input_schema": {
            "type": "object",
            "properties": {
                "datos_faltante": {"type": "string"}
            },
            "required": ["datos_faltante"]
        },
        "output_schema": {
            "type": "object",
            "properties": {"result": {"type": "string"}},
            "x-fastmcp-wrap-result": True
        }
    }
)
def falta_informacion(datos_faltante: str) -> str:
    """Indica que falta información."""
    print(f"[TOOL_USE] Faltan datos: {datos_faltante}")
    return f"Faltan datos: {datos_faltante}"


# =====================================================
# Recursos del hogar inteligente
# =====================================================

@mcp.resource(
    name="dispositivos_hogar_sincelejo",
    description="Lista de dispositivos monitorizados.",
    mime_type="application/json",
    uri="mcp://hogar/dispositivos"
)
def dispositivos_hogar_sincelejo():
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
    description="Contexto general del hogar ubicado en Sincelejo.",
    mime_type="application/json",
    uri="mcp://hogar/contexto_local"
)
def contexto_local_sincelejo():
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
    description="Información general sobre los habitantes del hogar.",
    mime_type="application/json",
    uri="mcp://hogar/familia"
)
def familia_sincelejo():
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
    mcp.run(transport="sse")
