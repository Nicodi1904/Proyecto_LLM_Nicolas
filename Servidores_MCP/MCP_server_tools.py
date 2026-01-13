from fastmcp import FastMCP
import pandas as pd
from PlayGround.cargar_CSV import cargar_dataset_sinselejo
import sys
import os

# Agregar directorio padre al path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Initialize FastMCP server
mcp = FastMCP("MCP_server_tools")

# -------------------------
# Cargar dataset global
# -------------------------
csv_path = os.path.join(os.path.dirname(__file__), "Energy Consumption in KWh of a Typical House Sincelejo Colombia.csv")
DATASET = cargar_dataset_sinselejo(csv_path)



def meta_energy(proposito: str, usar_si: list, no_usar: list):
    return { 
        "usar_si": usar_si,
        "no_usar_si": no_usar   # exactamente una condición crítica
    }


#Herramientas energéticas
@mcp.tool(
    meta={
        **meta_energy(
            proposito="Calcula el consumo energético dentro de un rango de horas de un día específico.",
            usar_si=[
                "El usuario pide consumo entre dos horas dentro del mismo día"
            ],
            no_usar="La consulta cubre más de un día"
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
            "properties": {"result": {"type": "number"}}
        }
    }
)
def consumo_rango_horas(dispositivo: str, hora_inicio: int, hora_fin: int, dia: int, mes: int, anio: int) -> float:
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


@mcp.tool(
    meta={
        **meta_energy(
            proposito="Calcula el consumo energético de uno o varios días dentro del mismo mes.",
            usar_si=[
                "El usuario pide consumo de un día específico o un rango de días dentro del mismo mes"
            ],
            no_usar="El rango solicitado cruza meses distintos"
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
            "properties": {"result": {"type": "number"}}
        }
    }
)
def consumo_rango_dias(dispositivo: str, dia_inicio: int, dia_fin: int, mes: int, anio: int) -> float:
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


@mcp.tool(
    meta={
        **meta_energy(
            proposito="Calcula el consumo energético acumulado entre uno o varios meses del mismo año.",
            usar_si=[
                "El usuario pide consumo mensual o de varios meses dentro del mismo año"
            ],
            no_usar="El usuario pide un rango que cruza dos años distintos"
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
            "properties": {"result": {"type": "number"}}
        }
    }
)
def consumo_rango_meses(dispositivo: str, mes_inicio: int, mes_fin: int, anio: int) -> float:
    df = DATASET[["TimeStamp", dispositivo]].copy()
    df = df[
        (df["TimeStamp"].dt.year == anio) &
        (df["TimeStamp"].dt.month >= mes_inicio) &
        (df["TimeStamp"].dt.month <= mes_fin)
    ]
    if df.empty:
        return 0.0
    return df[dispositivo].sum()


# =====================================================
# 📌 HERRAMIENTAS DE CONTROL
# =====================================================

@mcp.tool(
    meta={
        **meta_energy(
            proposito="Indica que la solicitud del usuario no puede resolverse con ninguna herramienta disponible.",
            usar_si=[
                "No existe ninguna herramienta aplicable a la intención detectada"
            ],
            no_usar="Existe una herramienta del dominio energético o matemático capaz de resolver la intención"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "razon": {"type": "string"}
            },
            "required": ["razon"]
        },
        "output_schema": {
            "type": "object",
            "properties": {"result": {"type": "string"}}
        }
    }
)
def plan_inviable(razon: str) -> str:
    print(f"[TOOL_USE] Plan inviable: {razon}")
    return f"Plan inviable: {razon}"


@mcp.tool(
    meta={
        **meta_energy(
            proposito="Indica que falta un dato obligatorio para ejecutar una acción del plan.",
            usar_si=[
                "Falta un parámetro necesario para ejecutar una herramienta"
            ],
            no_usar="La intención puede resolverse con los datos disponibles"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "datos_faltante": {"type": "string"}
            },
            "required": ["datos_faltante"]
        },
        "output_schema": {
            "type": "object",
            "properties": {"result": {"type": "string"}}
        }
    }
)
def falta_informacion(datos_faltante: str) -> str:
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
