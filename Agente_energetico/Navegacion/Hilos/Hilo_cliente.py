import sys
import os
import asyncio
from PySide6.QtCore import QThread, Signal

# Asegurar que el sistema reconozca las rutas (Tesis-MAS-LLM)
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from Agente_energetico.Sistema_MCP.mcp_client import MCP_Client
from Agente_energetico.Navegacion.Recursos_compartidos.Recursos_salida import RecursosSalida

class HiloCliente(QThread):
    terminado = Signal()
    error = Signal(str)

    def __init__(self, recursos_salida: RecursosSalida):
        super().__init__()
        self.recursos_salida = recursos_salida

    def run(self):
        try:
            # 1. Obtener el plan del paso anterior
            plan_acciones = self.recursos_salida.plan_acciones
            if not plan_acciones:
                raise ValueError("No hay plan de acciones provisto por el Planeador.")

            # 2. Inicializar Cliente MCP y Worker 3
            # Nota: Asumimos localhost:8000/sse por defecto según la prueba de mcp_client.py
            # TODO: En un futuro, el server_url idealmente se tomaría del variables de entorno o config.
            server_url = "http://localhost:8000/sse"
            cliente_mcp = MCP_Client()

            # 3. Ejecutar el plan (Requiere asyncio ya que ejecutar_plan es async)
            reporte_crudo = asyncio.run(cliente_mcp.ejecutar_plan(server_url, plan_acciones))

            if not reporte_crudo:
                 raise ValueError("El reporte de ejecución está vacío. Falló la conexión al servidor MCP.")
            
            # Revisar si hay errores fatales dentro de los reportes (ej error de timeout)
            for req_id, acciones in reporte_crudo.items():
                 for accion in acciones:
                      if accion.get("error"):
                           # Nota: Algunas herramientas fallidas pueden no matar todo el flujo,
                           # pero por seguridad lo sumamos al error global si es crítico.
                           pass

            # 4. Arreglar y dividir los reportes
            reporte_w3, reporte_llm = cliente_mcp.arreglar_reporte(reporte_crudo)

            # 5. Almacenar los resultados en Recursos_salida
            # Se guardan el reporte_llm (truncado para el Presentador) 
            # y el reporte_w3 (intacto para el Worker 3 que correrá en el Hilo_presentador)
            self.recursos_salida.reporte_ejecucion_llm = reporte_llm
            self.recursos_salida.reporte_ejecucion_worker3 = reporte_w3

            # 6. Disparar señal de éxito
            self.terminado.emit()

        except Exception as e:
            self.recursos_salida.error_msg = str(e)
            self.error.emit(str(e))
