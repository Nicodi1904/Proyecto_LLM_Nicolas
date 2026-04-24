import sys
import os
from PySide6.QtCore import QThread, Signal

# Asegurar que el sistema reconozca las rutas (Tesis-MAS-LLM)
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from Agente_energetico.Sistema_salida.Presentador.presentador import PresentadorAgente
from Agente_energetico.Sistema_salida.Worker3.worker3 import Worker3
from Agente_energetico.Navegacion.Recursos_compartidos.Recursos_salida import RecursosSalida

class HiloPresentador(QThread):
    terminado = Signal()
    error = Signal(str)

    def __init__(self, recursos_salida: RecursosSalida):
        super().__init__()
        self.recursos_salida = recursos_salida

    def run(self):
        try:
            # 1. Recuperar insumos
            notas_inferenciador = self.recursos_salida.notas_inferenciador
            notas_planeador = self.recursos_salida.notas_planeador
            reporte_llm = self.recursos_salida.reporte_ejecucion_llm
            reporte_worker3 = self.recursos_salida.reporte_ejecucion_worker3
            
            if not reporte_llm and not reporte_worker3:
                raise ValueError("No se encontraron reportes de ejecución MCP para procesar.")

            # 2. Generar Gráficas (Worker 3) a partir de los datos crudos
            # Lo hacemos aquí por requerimiento del diseño de flujo de datos
            worker3 = Worker3()
            graficas = worker3.generar_graficas(reporte_worker3)
            self.recursos_salida.graficas_worker3 = graficas

            import dspy
            print(f"🔌 [Hilo Presentador] Inicializando dspy.LM para {self.modelo_final}...")
            llm_activo = dspy.LM(self.modelo_final, api_base=self.base_final, api_key=self.api_key, num_ctx=8192, max_retries=1, timeout=120)

            with dspy.context(lm=llm_activo):
                # 3. Inicializar Agente Presentador
                # Carga automáticamente el formato_respuesta.json en su __init__
                agente = PresentadorAgente()

                # 4. Sintetizar el Informe de Texto
                resultado = agente(
                    notas_inferenciador=notas_inferenciador,
                    notas_planeador=notas_planeador,
                    Informe_LLM=reporte_llm
                )

                # 5. Formatear la respuesta en un diccionario
                respuesta_formateada = {
                    "Resumen_op": getattr(resultado, "Resumen_op", "Resumen no disponible."),
                    "Resultados_op": getattr(resultado, "Resultados_op", "Resultados no disponibles."),
                    "analisis": getattr(resultado, "analisis", "Análisis no disponible."),
                    "sugerencia": getattr(resultado, "sugerencia", "Sugerencias no disponibles.")
                }

                self.recursos_salida.respuesta_presentador = respuesta_formateada

            # 6. Disparar señal de éxito
            self.terminado.emit()

        except Exception as e:
            self.recursos_salida.error_msg = str(e)
            self.error.emit(str(e))
