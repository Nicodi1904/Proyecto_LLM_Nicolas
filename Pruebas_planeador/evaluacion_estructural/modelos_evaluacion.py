from pydantic import BaseModel, Field, field_validator, RootModel
from typing import List, Optional, Union, Dict, Any, Literal
import re

# Patrón para validar referencias a IDs de acciones (ej: @1.1)
ID_PATTERN = re.compile(r"^@\d+\.\d+$")

def is_reference(v: Any) -> bool:
    return isinstance(v, str) and ID_PATTERN.match(v)

# --- Modelos para Inputs de Herramientas ---

class ObtenerConsumoInput(BaseModel):
    dispositivos: List[str]
    fecha_inicio: str  # ISO 8601
    fecha_fin: str     # ISO 8601
    granularidad: Optional[Literal["hora", "dia", "mes", "total"]] = "total"

class ObjetivoComparacion(BaseModel):
    dispositivo: str
    fecha_inicio: str
    fecha_fin: str

class AnalizarComparacionInput(BaseModel):
    objetivo_a: Union[ObjetivoComparacion, str] # Permite referencia @N.M
    objetivo_b: Union[ObjetivoComparacion, str]

    @field_validator("objetivo_a", "objetivo_b")
    @classmethod
    def validate_reference(cls, v):
        if isinstance(v, str) and not is_reference(v):
            raise ValueError(f"La referencia '{v}' no tiene un formato válido (@N.M)")
        return v

class DetectarAnomaliasInput(BaseModel):
    dispositivo: str
    fecha_inicio: str
    fecha_fin: str
    sensibilidad: Optional[float] = 3.0

class AnalizarTendenciaInput(BaseModel):
    dispositivo: str
    fecha_inicio: str
    fecha_fin: str

class AccionImposibleInput(BaseModel):
    solicitud: str
    justificacion: str

# --- Modelo de Acción General ---

class Accion(BaseModel):
    id: str = Field(pattern=r"^@\d+\.\d+$")
    server_id: str
    tool: str
    inputs: Dict[str, Any] # Lo validaremos dinámicamente según la herramienta
    descripcion: str

    @field_validator("tool")
    @classmethod
    def validate_tool_name(cls, v):
        valid_tools = ["obtener_consumo", "analizar_comparacion", "detectar_anomalias", "analizar_tendencia", "accion_imposible"]
        if v not in valid_tools:
            raise ValueError(f"Herramienta desconocida: {v}")
        return v

class PlanAcciones(BaseModel):
    plan: List[Accion]

def validar_inputs_herramienta(tool: str, inputs: dict):
    """Valida los inputs específicos de cada herramienta usando Pydantic."""
    if tool == "obtener_consumo":
        ObtenerConsumoInput(**inputs)
    elif tool == "analizar_comparacion":
        AnalizarComparacionInput(**inputs)
    elif tool == "detectar_anomalias":
        DetectarAnomaliasInput(**inputs)
    elif tool == "analizar_tendencia":
        AnalizarTendenciaInput(**inputs)
    elif tool == "accion_imposible":
        AccionImposibleInput(**inputs)
    else:
        raise ValueError(f"No hay validador para la herramienta: {tool}")
