import sys
import os
import json
from PySide6.QtCore import QThread, Signal

# Asegurar que el sistema reconozca las rutas (Tesis-MAS-LLM)
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from Agente_energetico.Sistema_entrada.Planeador.planeador import PlaneadorAgente
from Agente_energetico.Navegacion.Recursos_compartidos.Recursos_entrada import QueryRequest
from Agente_energetico.Navegacion.Recursos_compartidos.Recursos_salida import RecursosSalida

class HiloPlaneador(QThread):
    terminado = Signal()
    error = Signal(str)

    def __init__(self, recursos_entrada: QueryRequest, recursos_salida: RecursosSalida):
        super().__init__()
        self.recursos_entrada = recursos_entrada
        self.recursos_salida = recursos_salida
        self.usar_few_shots = False
        
        # Parámetros para inicializar LLM en el hilo
        self.modelo_final = ""
        self.base_final = ""
        self.api_key = ""

    def run(self):
        try:
            import dspy
            print(f"🔌 [Hilo Planeador] Inicializando dspy.LM para {self.modelo_final}...")
            llm_activo = dspy.LM(self.modelo_final, api_base=self.base_final, api_key=self.api_key, num_ctx=8192, max_retries=1, timeout=120)

            with dspy.context(lm=llm_activo):
                # 1. Crear el agente Planeador
                agente = PlaneadorAgente()
                
                if self.usar_few_shots:
                    agente.entrenar_con_fewshots()
                if not agente.system_summary:
                    raise ValueError("No se pudo cargar el resumen del sistema (system_summary.json).")

                # 2. Obtener insumos
                solicitudes = self.recursos_salida.solicitudes_categorizadas
                if not solicitudes:
                    raise ValueError("No hay solicitudes categorizadas provistas por el Interpretador.")

                contexto_temporal = {
                    "fecha_actual": f"{self.recursos_entrada.fecha}T{self.recursos_entrada.hora}"
                }
                
                preferencias_horarias = self.recursos_entrada.referencias_horarias
                if not preferencias_horarias:
                    preferencias_horarias = {}

                print(f"\n🧠 [Hilo Planeador] EJECUTANDO AGENTE CON:")
                import json
                print(f" - Solicitudes Categorizadas:\n{json.dumps(solicitudes, indent=2, ensure_ascii=False)}")
                print(f" - Contexto Temporal: {contexto_temporal}")
                print(f" - Preferencias Horarias:\n{json.dumps(preferencias_horarias, indent=2, ensure_ascii=False)}")

                # 3. Ejecutar el agente 
                resultado = agente(
                    solicitudes_categorizadas=solicitudes, 
                    temporal_context=contexto_temporal,
                    temporal_preferences=preferencias_horarias
                )

                print(f"✅ [Hilo Planeador] RESPUESTA DEL AGENTE:")
                print(f"Plan de Acciones: {getattr(resultado, 'plan_acciones', 'N/A')}")
                print(f"Notas: {getattr(resultado, 'notas', 'Sin notas')}\n")

                print("\n🔍 [Hilo Planeador] HISTORIAL DSPy (Último Prompt y Respuesta Cruda):")
                dspy.inspect_history(n=1)
                print("\n" + "="*50 + "\n")

                # Validar con Worker 2
                reporte_w2 = agente.worker2(resultado)
                self.recursos_salida.reporte_worker2 = reporte_w2

                if not reporte_w2.get("valido", False):
                    errores_formateados = "; ".join(reporte_w2.get("errores", []))
                    raise ValueError(f"Worker 2 encontró errores en el plan: {errores_formateados}")

                # 4. Asignar las salidas válidas a los recursos de salida
                self.recursos_salida.plan_acciones = getattr(resultado, "plan_acciones", [])
                self.recursos_salida.notas_planeador = getattr(resultado, "notas", "")

            # Fuera del contexto (no se necesita LM para emitir señales)
            self.terminado.emit()

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.recursos_salida.error_msg = str(e)
            self.error.emit(str(e))
