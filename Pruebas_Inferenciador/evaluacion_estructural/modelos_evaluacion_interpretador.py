"""
Modelos Pydantic para la Evaluación Estructural Determinista del Interpretador.
Replica el contrato de salida definido en la Signature del InterpretadorAgente.
"""
from pydantic import BaseModel, Field, field_validator
from typing import Dict
import re

# Patrón de clave: @N (N entero positivo, consecutivo)
KEY_PATTERN = re.compile(r"^@\d+$")

# Escenarios válidos según escenarios.json del sistema
ESCENARIOS_VALIDOS = {
    "consumo_basico",
    "comparacion_consumos",
    "deteccion_anomalias_tendencias",
    "prediccion_estimacion",
    "entrada_inadmisible",
}


class SolicitudItem(BaseModel):
    """
    Representa un ítem individual dentro del diccionario de solicitudes_categorizadas.
    Cada ítem debe tener:
      - solicitud: string no vacío con la solicitud detallada y autocontenida.
      - escenario: string que debe ser uno de los escenarios válidos del sistema.
    """
    solicitud: str = Field(min_length=1)
    escenario: str

    @field_validator("escenario")
    @classmethod
    def validar_escenario(cls, v: str) -> str:
        if v not in ESCENARIOS_VALIDOS:
            raise ValueError(
                f"Escenario desconocido: '{v}'. Válidos: {sorted(ESCENARIOS_VALIDOS)}"
            )
        return v


class SolicitudesCategorizadas(BaseModel):
    """
    Modelo raíz del output del Interpretador.
    Las claves deben seguir el patrón '@N' (N entero positivo).
    """
    solicitudes: Dict[str, SolicitudItem]

    @field_validator("solicitudes")
    @classmethod
    def validar_claves(cls, v: dict) -> dict:
        for key in v:
            if not KEY_PATTERN.match(key):
                raise ValueError(
                    f"Clave '{key}' no sigue el formato '@N' (ej. '@1', '@2')."
                )
        return v


def validar_solicitudes_categorizadas(data: dict) -> dict:
    """
    Punto de entrada principal para la validación.
    Recibe el diccionario de solicitudes (ya parseado) y retorna
    {'valido': bool, 'errores': list[str]}.
    """
    errores = []

    if not isinstance(data, dict):
        return {
            "valido": False,
            "errores": ["La salida debe ser un diccionario (dict)."],
        }

    if not data:
        return {
            "valido": False,
            "errores": ["El diccionario de solicitudes está vacío."],
        }

    # Validar claves con regex manualmente primero para mensajes precisos
    for key, val in data.items():
        if not KEY_PATTERN.match(str(key)):
            errores.append(
                f"Clave '{key}' no sigue el formato '@N'."
            )

        if not isinstance(val, dict):
            errores.append(f"El valor de '{key}' debe ser un diccionario.")
            continue

        # Validar con Pydantic el item completo
        try:
            SolicitudItem(**val)
        except Exception as e:
            if hasattr(e, "errors"):
                for err in e.errors():
                    loc = " -> ".join(str(x) for x in err["loc"])
                    errores.append(f"'{key}' -> {loc}: {err['msg']}")
            else:
                errores.append(f"'{key}': {str(e)}")

    return {
        "valido": len(errores) == 0,
        "errores": errores,
    }
