from dataclasses import dataclass

@dataclass
class QueryRequest:
    pregunta: str
    fecha: str
    hora: str
    modelo: str
    referencias_horarias: dict  # { 'madrugada': '08:00 AM', ... }
    few_shots: bool
    widget: bool
