from dataclasses import dataclass, field
from typing import Any, Optional

@dataclass
class RecursosSalida:
    Consulta: str = ""
    # El diccionario crudo devuelto por el Presentador
    respuesta_presentador: dict = field(default_factory=dict)
    
    # Salidas del Interpretador y Worker 1
    solicitudes_categorizadas: dict = field(default_factory=dict)
    reporte_worker1: dict = field(default_factory=dict)
    
    # Salidas del Planeador y Worker 2
    plan_acciones: list = field(default_factory=list)
    reporte_worker2: dict = field(default_factory=dict)
    
    # Salidas del Cliente MCP 
    reporte_ejecucion_llm: dict = field(default_factory=dict)
    reporte_ejecucion_worker3: dict = field(default_factory=dict)
    
    # El diccionario de gráficas Plotly devuelto por Worker 3
    graficas_worker3: dict = field(default_factory=dict)
    
    # Notas y comentarios de los agentes
    notas_inferenciador: str = ""
    notas_planeador: str = ""
    
    # Texto de error global
    error_msg: str = ""
