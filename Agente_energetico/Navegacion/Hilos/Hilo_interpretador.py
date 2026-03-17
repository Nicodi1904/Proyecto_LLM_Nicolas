import sys
import os
from PySide6.QtCore import QThread, Signal

# Asegurar que el sistema reconozca las rutas (Tesis-MAS-LLM)
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from Agente_energetico.Sistema_entrada.Interpretador.interpretador import InterpretadorAgente
from Agente_energetico.Navegacion.Recursos_compartidos.Recursos_entrada import QueryRequest
from Agente_energetico.Navegacion.Recursos_compartidos.Recursos_salida import RecursosSalida

class HiloInterpretador(QThread):
    # Señales para comunicarse con la UI / Orquestador
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
            # 0. Configurar dspy en segundo plano para evitar congelar la GUI
            import dspy
            print(f"🔌 [Hilo Interpretador] Inicializando dspy.LM para {self.modelo_final}...")
            llm_activo = dspy.LM(self.modelo_final, api_base=self.base_final, api_key=self.api_key, num_ctx=8192)

            # Usar dspy.context para seguridad de hilos (Thread-Local)
            with dspy.context(lm=llm_activo):
                print(f"🔌 [Hilo Interpretador] Contexto LM aplicado.")
                
                # 1. Crear el agente interpretador (Carga escenarios automáticamente)
                agente = InterpretadorAgente()
                
                if self.usar_few_shots:
                    agente.entrenar_con_fewshots()
                
                if not agente.escenarios:
                    raise ValueError("No se pudieron cargar los escenarios. Revisa escenarios.json.")

                # 2. Obtener el prompt del usuario
                prompt = self.recursos_entrada.Mensaje_usuario
                if not prompt:
                    raise ValueError("El mensaje del usuario está vacío.")

                print(f"\n🧠 [Hilo Interpretador] EJECUTANDO AGENTE CON PROMPT: '{prompt}'")

                # 3. Ejecutar el agente
                resultado = agente(prompt)
                
                print(f"✅ [Hilo Interpretador] RESPUESTA DEL AGENTE:")
                # Imprimir directamente la representación (más seguro)
                print(f"Solicitudes: {getattr(resultado, 'solicitudes_categorizadas', 'N/A')}")
                print(f"Notas: {getattr(resultado, 'notas', 'Sin notas')}\n")
                
                # Pasar resultado al worker1 para validación
                reporte_w1 = agente.worker1(resultado)
                self.recursos_salida.reporte_worker1 = reporte_w1
                
                # Guardar en RecursosSalida
                if getattr(resultado, "solicitudes_categorizadas", None):
                    self.recursos_salida.solicitudes_categorizadas = resultado.solicitudes_categorizadas
                    self.recursos_salida.notas_inferenciador = getattr(resultado, "notas", "")
                else:
                    raise ValueError("El agente no devolvió solicitudes categorizadas válidas.")

            # Fuera del contexto (no se necesita LM para emitir señales)
            if reporte_w1.get("valido", False):
                self.terminado.emit()
            else:
                errores = "\n".join(reporte_w1.get("errores", []))
                self.error.emit(f"Error de Validación (Worker 1):\n{errores}")

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.recursos_salida.error_msg = str(e)
            self.error.emit(str(e))
