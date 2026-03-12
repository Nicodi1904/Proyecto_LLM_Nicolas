from dataclasses import dataclass, field
from typing import Any, Optional

@dataclass
class RecursosSalida:
    Consulta: str = ""
    Respuesta: str = ""
    Recursos_img: list = field(default_factory=list)
