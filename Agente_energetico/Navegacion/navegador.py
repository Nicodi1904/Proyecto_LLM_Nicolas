import sys
import os
import json
from PySide6.QtWidgets import QApplication, QWidget, QStackedWidget, QVBoxLayout, QMessageBox
from PySide6.QtCore import Qt, QTimer, QPoint
import sqlite3

# Asegurar visibilidad de los módulos de Interfaz_usuario
script_dir = os.path.dirname(os.path.abspath(__file__))
# Ahora que Navegacion está en Agente_energetico, el parent_dir es Agente_energetico
agente_energetico_dir = os.path.dirname(script_dir)
interfaz_dir = os.path.join(agente_energetico_dir, "Interfaz_usuario")
sys.path.append(interfaz_dir)

from QueryWindow.query_window import Query_Window
from ResultWindow.result_window import Result_Window
from Recursos_compartidos.Recursos_entrada import QueryRequest
from Recursos_compartidos.Recursos_salida import RecursosSalida
from QueryWindow.DbWindow.db_window import DbWindow
from FloatingWidget.floating_widget import FloatingWidget
from cript import encriptar_clave, desencriptar_clave

class Navigator(QWidget):
    """
    Controlador central de la aplicación.
    Utiliza QStackedWidget para alternar entre QueryWindow y ResultWindow.
    """
    def __init__(self):
        super().__init__()
        self.ultima_consulta = None
        self.recursos_salida = None
        self.db_path = r'C:\sqlite_tesis\Base_datos_tesis\Hogar_Sincelejo.db'
        
        # Guardaremos referencias a los hilos para evitar que el recolector de basura los destruya
        self.hilo_interpretador = None
        self.hilo_planeador = None
        self.hilo_cliente = None
        self.hilo_presentador = None
        self.paso_actual_mas = 0 # Rastreador del hilo en ejecución
        self.mensaje_espera_actual = "" # Almacena el texto de carga de fondo
        self.settings_path = os.path.join(script_dir, "settings.json")
        self.floating_widget = None

        self._inicializar_tabla_modelos()
        self._configurar_ventana()
        self._setup_ui()
        self._cargar_modelos_dropdown()
        self._conectar_senales()
        self._cargar_settings()

    def _configurar_ventana(self):
        self.setWindowTitle("Agente Energético - MAS")
        self.resize(1000, 600)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setWindowFlags(Qt.Window)

    def _setup_ui(self):
        self.layout_principal = QVBoxLayout(self)
        self.layout_principal.setContentsMargins(0, 0, 0, 0)

        # Contenedor de vistas
        self.stacked_widget = QStackedWidget()
        
        # Instanciar ventanas
        self.query_window = Query_Window()
        self.result_window = Result_Window()

        # Añadir al stack
        self.stacked_widget.addWidget(self.query_window)   # Índice 0
        self.stacked_widget.addWidget(self.result_window)  # Índice 1

        self.layout_principal.addWidget(self.stacked_widget)

    def _conectar_senales(self):
        # Cuando se dispara una consulta en QueryWindow, cambiar a ResultWindow
        self.query_window.consulta_disparada.connect(self._navegar_a_resultados)
        # Cuando se solicita volver desde ResultWindow
        self.result_window.solicitud_regreso.connect(self._navegar_a_consulta)
        # Click en los pasos del LeftMenu
        self.result_window.left_menu.step_clicked.connect(self._cambiar_vista_paso)
        # Botón "+" para añadir modelo
        self.query_window.status_info.btn_add_modelo.clicked.connect(self._abrir_ventana_db)
        # Botones de gestión
        self.query_window.status_info.editar_modelo_solicitado.connect(self._editar_modelo_actual)
        self.query_window.status_info.eliminar_modelo_solicitado.connect(self._eliminar_modelo_actual)
        
        # Switch de Widget
        self.query_window.right_bar.sw_widget.toggled.connect(self._toggle_modo_widget)

    def _inicializar_tabla_modelos(self):
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS Modelos_lenguaje (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    Model TEXT UNIQUE NOT NULL,
                    Api_base TEXT NOT NULL,
                    Encripted_ApiKey TEXT
                )
            """)
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"Error inicializando tabla Modelos_lenguaje: {e}")

    def _cargar_modelos_dropdown(self):
        try:
            self.query_window.status_info.combo_modelos.clear()
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT Model FROM Modelos_lenguaje")
            modelos = cursor.fetchall()
            conn.close()

            if modelos:
                modelos_nombre = [m[0] for m in modelos]
                self.query_window.status_info.combo_modelos.addItems(modelos_nombre)
            else:
                 # Fallback por si está vacía
                 self.query_window.status_info.combo_modelos.addItem("Sin modelos")
        except Exception as e:
             print(f"Error cargando modelos al dropdown: {e}")

    def _abrir_ventana_db(self):
        ventana = DbWindow(self)
        ventana.guardar_presionado.connect(self._guardar_nuevo_modelo)
        ventana.exec() # Corre en modo Dialog bloqueante

    def _editar_modelo_actual(self):
        nombre_modelo = self.query_window.status_info.get_selected_model()
        if not nombre_modelo or nombre_modelo == "Sin modelos":
             QMessageBox.warning(self, "Advertencia", "No hay un modelo seleccionado para editar.")
             return
             
        # Cargar datos de la DB
        api_base = ""
        api_key_desencriptada = ""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT Api_base, Encripted_ApiKey FROM Modelos_lenguaje WHERE Model=?", (nombre_modelo,))
            res = cursor.fetchone()
            conn.close()
            
            if res:
                api_base = res[0]
                if res[1]:
                    api_key_desencriptada = desencriptar_clave(res[1])
        except Exception as e:
            print(f"Error cargando modelo para editar: {e}")

        ventana = DbWindow(self, model=nombre_modelo, base=api_base, key=api_key_desencriptada)
        ventana.guardar_presionado.connect(self._guardar_nuevo_modelo)
        ventana.exec()

    def _eliminar_modelo_actual(self):
        nombre_modelo = self.query_window.status_info.get_selected_model()
        if not nombre_modelo or nombre_modelo == "Sin modelos":
             QMessageBox.warning(self, "Advertencia", "No hay un modelo seleccionado para eliminar.")
             return

        respuesta = QMessageBox.question(
            self, 
            "Eliminar Modelo", 
            f"¿Estás seguro de que deseas eliminar el modelo '{nombre_modelo}'?",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if respuesta == QMessageBox.Yes:
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                cursor.execute("DELETE FROM Modelos_lenguaje WHERE Model=?", (nombre_modelo,))
                conn.commit()
                conn.close()
                
                QMessageBox.information(self, "Éxito", f"Modelo '{nombre_modelo}' eliminado correctamente.")
                self._cargar_modelos_dropdown()
            except Exception as e:
                print(f"Error al eliminar modelo: {e}")
                QMessageBox.critical(self, "Error", f"No se pudo eliminar el modelo.\nDetalle: {e}")

    def _guardar_nuevo_modelo(self, model, base, key):
        try:
            # 1. Encriptar Key usando cript.py
            key_encriptada = encriptar_clave(key)

            # 2. Guardar en SQLite
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO Modelos_lenguaje (Model, Api_base, Encripted_ApiKey)
                VALUES (?, ?, ?)
            """, (model, base, key_encriptada))
            conn.commit()
            conn.close()
            
            QMessageBox.information(self, "Éxito", f"Modelo '{model}' guardado correctamente en la base de datos.")

            # 3. Recargar el dropdown para que se vea reflejado el cambio
            self._cargar_modelos_dropdown()
            # Seleccionar el recién agregado
            index = self.query_window.status_info.combo_modelos.findText(model)
            if index >= 0:
                 self.query_window.status_info.combo_modelos.setCurrentIndex(index)

        except Exception as e:
             print(f"Error al guardar modelo en DB: {e}")
             QMessageBox.critical(self, "Error", f"No se pudo guardar el modelo.\nDetalle: {e}")

    def _navegar_a_resultados(self, consulta):
        try:
            # Capturar datos integrales desde la ventana de consulta
            datos = self.query_window.get_full_query_data()
            
            # VALIDACIÓN: Si no hay un modelo válido seleccionado, no dejar avanzar
            modelo_elegido = datos.get("modelo")
            if not modelo_elegido or modelo_elegido == "Sin modelos":
                print("Consulta de procesamiento cancelada: No se ha seleccionado un modelo de lenguaje válido.")
                self.query_window.limpiar_interfaz()
                QMessageBox.warning(self, "Advertencia", "No se ha configurado un modelo de lenguaje válido. Por favor, añada o seleccione uno.")
                return
            
            # Guardar en recursos compartidos
            self.ultima_consulta = QueryRequest(
                Mensaje_usuario=datos["pregunta"],
                fecha=datos["fecha"],
                hora=datos["hora"],
                modelo=datos["modelo"],
                referencias_horarias=datos["referencias_horarias"],
                few_shots=datos["few_shots"],
                widget=datos["widget"]
            )
            
            # Pasar datos a la ventana de resultados y cambiar vista temporal
            self.mensaje_espera_actual = "Iniciando análisis inteligente..."
            self.result_window.mostrar_datos(consulta, self.mensaje_espera_actual)
            self.stacked_widget.setCurrentIndex(1)
            
            # Iniciar la cadena de ejecución asíncrona delegándola para evitar bloquear el renderizado
            QTimer.singleShot(100, self._iniciar_procesamiento_en_cadena)

        except Exception as e:
            print(f"Error en _navegar_a_resultados: {e}")
            QMessageBox.critical(self, "Error", f"Error crítico al procesar navegación:\n{e}")

    def _iniciar_procesamiento_en_cadena(self):
        # 0. Importaciones Diferidas (Lazy Loading)
        from Hilos.Hilo_interpretador import HiloInterpretador
        from Hilos.Hilo_planeador import HiloPlaneador
        from Hilos.Hilo_cliente import HiloCliente
        from Hilos.Hilo_presentador import HiloPresentador
        
        # 1. Recuperar Modelo desde DB y Desencriptar
        nombre_modelo = self.ultima_consulta.modelo
        api_base = "http://localhost:11434"
        api_key = ""
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT Api_base, Encripted_ApiKey FROM Modelos_lenguaje WHERE Model=?", (nombre_modelo,))
            resultado = cursor.fetchone()
            conn.close()
            
            if resultado:
                api_base = resultado[0]
                encrypted_key = resultado[1]
                
                # Desencriptar usando cript.py
                api_key = desencriptar_clave(encrypted_key)
            
        except Exception as e:
            print(f"Error recuperando credenciales de la DB: {e}")
            # Fallback continuará con lo que tenga

        # 2. Determinar variables de ruteo del modelo (fallback para locales)
        if "/" in nombre_modelo:
            modelo_final = nombre_modelo
            base_final = api_base if api_base else None
        else:
            modelo_final = f'ollama_chat/{nombre_modelo}'
            base_final = api_base if api_base else "http://localhost:11434"

        # 3. Reiniciar recursos de salida y estados visuales
        self.recursos_salida = RecursosSalida()
        self.paso_actual_mas = 0
        for i in range(4):
            self.result_window.left_menu.set_step_status(i, 0)
        self.result_window.left_menu.set_active_step(3) # Dejar enfocado el último por visual de carga
        
        print("\n" + "="*50)
        print("📥 RECURSOS DE ENTRADA ENVIADOS (navegador.py)")
        print(f"Mensaje: {self.ultima_consulta.Mensaje_usuario}")
        print(f"Fecha: {self.ultima_consulta.fecha} | Hora: {self.ultima_consulta.hora}")
        print(f"Modelo: {self.ultima_consulta.modelo}")
        print(f"Few-Shots: {self.ultima_consulta.few_shots}")
        print(f"Referencias Horarias:\n{json.dumps(self.ultima_consulta.referencias_horarias, indent=2)}")
        print("="*50 + "\n")
        
        # 1. Instanciar Hilos
        self.hilo_interpretador = HiloInterpretador(self.ultima_consulta, self.recursos_salida)
        # Pasar credenciales para inicializar el LLM de forma asíncrona
        self.hilo_interpretador.modelo_final = modelo_final
        self.hilo_interpretador.base_final = base_final
        self.hilo_interpretador.api_key = api_key
        
        self.hilo_planeador = HiloPlaneador(self.ultima_consulta, self.recursos_salida)
        self.hilo_planeador.modelo_final = modelo_final
        self.hilo_planeador.base_final = base_final
        self.hilo_planeador.api_key = api_key
        
        self.hilo_cliente = HiloCliente(self.recursos_salida)
        self.hilo_presentador = HiloPresentador(self.recursos_salida)
        self.hilo_presentador.modelo_final = modelo_final
        self.hilo_presentador.base_final = base_final
        self.hilo_presentador.api_key = api_key
        
        # Articular lógica de Few-Shots
        if self.ultima_consulta.few_shots:
            self.hilo_interpretador.usar_few_shots = True
            self.hilo_planeador.usar_few_shots = True
        
        # 2. Conectar Señales de Flujo (Éxito)
        self.hilo_interpretador.terminado.connect(self._al_terminar_interpretador)
        self.hilo_planeador.terminado.connect(self._al_terminar_planeador)
        self.hilo_cliente.terminado.connect(self._al_terminar_cliente)
        self.hilo_presentador.terminado.connect(self._al_terminar_presentador)
        
        # 3. Conectar Señales de Error
        self.hilo_interpretador.error.connect(self._manejar_error_hilo)
        self.hilo_planeador.error.connect(self._manejar_error_hilo)
        self.hilo_cliente.error.connect(self._manejar_error_hilo)
        self.hilo_presentador.error.connect(self._manejar_error_hilo)
        
        # 4. Arrancar la cadena (Empieza Interpretador)
        self.hilo_interpretador.start()

    # --- Callbacks de Transición de Hilos ---
    def _al_terminar_interpretador(self):
        print("\n=== [1] SALIDA DEL INTERPRETADOR (Inferencia) ===")
        print(self.recursos_salida.solicitudes_categorizadas)
        self.result_window.left_menu.set_step_status(0, 1) # Verde
        self.paso_actual_mas = 1
        self.mensaje_espera_actual = "Comprendiendo su solicitud... ☑\nDiseñando plan de acciones energéticas..."
        self.result_window.mostrar_datos(self.ultima_consulta.Mensaje_usuario, self.mensaje_espera_actual)
        self.hilo_planeador.start()
        
    def _al_terminar_planeador(self):
        print("\n=== [2] SALIDA DEL PLANEADOR (Worker 1) ===")
        print(self.recursos_salida.plan_acciones)
        self.result_window.left_menu.set_step_status(1, 1) # Verde
        self.paso_actual_mas = 2
        self.mensaje_espera_actual = "Plan diseñado... ☑\nConectando a bases de datos MCP para recuperar información..."
        self.result_window.mostrar_datos(self.ultima_consulta.Mensaje_usuario, self.mensaje_espera_actual)
        self.hilo_cliente.start()
        
    def _al_terminar_cliente(self):
        print("\n=== [3] SALIDA DEL CLIENTE MCP (Worker 2) ===")
        print(self.recursos_salida.reporte_ejecucion_worker3)
        self.result_window.left_menu.set_step_status(2, 1) # Verde
        self.paso_actual_mas = 3
        self.mensaje_espera_actual = "Información recuperada... ☑\nGenerando gráficas y redactando informe final..."
        self.result_window.mostrar_datos(self.ultima_consulta.Mensaje_usuario, self.mensaje_espera_actual)
        self.hilo_presentador.start()

    def _al_terminar_presentador(self):
        respuesta = self.recursos_salida.respuesta_presentador
        graficas = self.recursos_salida.graficas_worker3
        
        print("\n=== [4] SALIDA DEL PRESENTADOR (Informe Final) ===")
        print(respuesta)
        
        self.result_window.left_menu.set_step_status(3, 1) # Verde
        self.result_window.left_menu.set_active_step(3)

        
        self.result_window.mostrar_datos(self.ultima_consulta.Mensaje_usuario, respuesta)
        self.result_window.resources_panel.display_graphs(graficas)

    def _cambiar_vista_paso(self, index):
        """Disparado por el LeftMenu para visualizar auditoría de cada hilo."""
        if not self.recursos_salida: return
        self.result_window.left_menu.set_active_step(index)
        
        if index == 0:
            # Interpretador
            self.result_window.response_panel.set_title("Razonamiento del modelo")
            self.result_window.response_panel.set_response_text(self.recursos_salida.notas_inferenciador)
            
            self.result_window.resources_panel.set_title("Solicitudes categorizadas")
            self.result_window.resources_panel.display_text(json.dumps(self.recursos_salida.solicitudes_categorizadas, indent=2, ensure_ascii=False))
        
        elif index == 1:
            # Planeador
            self.result_window.response_panel.set_title("Razonamiento del modelo")
            self.result_window.response_panel.set_response_text(self.recursos_salida.notas_planeador)
            
            self.result_window.resources_panel.set_title("Plan de acciones")
            self.result_window.resources_panel.display_text(json.dumps(self.recursos_salida.plan_acciones, indent=2, ensure_ascii=False))
            
        elif index == 2:
            # Cliente MCP
            self.result_window.response_panel.set_title("Acciones Llamadas")
            self.result_window.resources_panel.set_title("Resultados del Servidor")
            
            llamadas = {}
            resultados = {}
            for req_id, acciones in self.recursos_salida.reporte_ejecucion_worker3.items():
                llamadas[req_id] = []
                resultados[req_id] = []
                for acc in acciones:
                    # Copiar acción sin los resultados o gráficos para mostrar las puras llamadas (inputs)
                    llamada = {k: v for k, v in acc.items() if k not in ["resultado", "figura", "error"]}
                    llamadas[req_id].append(llamada)
                    
                    # Extraer puramente la salida del servidor
                    res = acc.get("resultado", acc.get("error", "Sin resultado"))
                    resultados[req_id].append({acc.get("accion_id", "accion"): res})
            
            self.result_window.response_panel.set_response_text(json.dumps(llamadas, indent=2, ensure_ascii=False))
            self.result_window.resources_panel.display_text(json.dumps(resultados, indent=2, ensure_ascii=False))
            
        elif index == 3:
            # Presentador (Vista predeterminada)
            self.result_window.response_panel.set_title("RESPUESTA")
            self.result_window.resources_panel.set_title("RECURSOS Y DATOS")
            
            if self.recursos_salida.respuesta_presentador:
                self.result_window.response_panel.set_response_text(self.recursos_salida.respuesta_presentador)
                self.result_window.resources_panel.display_graphs(self.recursos_salida.graficas_worker3)
            else:
                self.result_window.response_panel.set_response_text(self.mensaje_espera_actual)
                self.result_window.resources_panel.display_graphs({})

    def _manejar_error_hilo(self, mensaje_error):
        self.result_window.left_menu.set_step_status(self.paso_actual_mas, 3) # Rojo al paso que falló
        texto_error = (
            f"El agente ha detectado un problema grave de ejecución.\n\n"
            f"Posibles causas referidas al Modelo o Comunicación:\n"
            f"1. El modelo '{self.ultima_consulta.modelo}' no existe localmente (Ollama).\n"
            f"2. Hubo un error de conexión con la Base URL (O el puerto provisto).\n"
            f"3. La Clave API es inexistente, inválida o ha superado su límite.\n\n"
            f"Detalle técnico emitido internamente:\n"
            f"{mensaje_error}"
        )
        self.result_window.mostrar_datos(self.ultima_consulta.Mensaje_usuario, texto_error)

    def _navegar_a_consulta(self):
        # Descartar paquete de recursos compartidos y limpiar interfaz
        self.ultima_consulta = None
        self.query_window.limpiar_interfaz()
        self.stacked_widget.setCurrentIndex(0)

    # --- Gestión de Configuración y Widget ---
    def _cargar_settings(self):
        if os.path.exists(self.settings_path):
            try:
                with open(self.settings_path, 'r') as f:
                    settings = json.load(f)
                
                # Aplicar estado del widget
                active = settings.get("widget_active", False)
                pos = settings.get("widget_pos", [100, 100])
                
                if active:
                    # Sincronizar el switch visualmente sin disparar señal recursiva
                    self.query_window.right_bar.sw_widget.blockSignals(True)
                    self.query_window.right_bar.sw_widget.setChecked(True)
                    self.query_window.right_bar.sw_widget.blockSignals(False)
                    # Forzar entrada a modo widget
                    QTimer.singleShot(500, lambda: self._toggle_modo_widget(True))
            except Exception as e:
                print(f"Error cargando settings: {e}")

    def _guardar_settings(self):
        settings = {
            "widget_active": self.query_window.right_bar.sw_widget.isChecked(),
            "widget_pos": [self.floating_widget.x(), self.floating_widget.y()] if self.floating_widget else [100, 100]
        }
        try:
            with open(self.settings_path, 'w') as f:
                json.dump(settings, f, indent=4)
        except Exception as e:
            print(f"Error guardando settings: {e}")

    def _toggle_modo_widget(self, active):
        if active:
            # Leer posición guardada
            pos_x, pos_y = 100, 100
            if os.path.exists(self.settings_path):
                with open(self.settings_path, 'r') as f:
                    s = json.load(f)
                    pos_x, pos_y = s.get("widget_pos", [100, 100])

            self.hide()
            if not self.floating_widget:
                self.floating_widget = FloatingWidget(QPoint(pos_x, pos_y))
                self.floating_widget.solicitar_restauracion.connect(self._restaurar_desde_widget)
            self.floating_widget.show()
        else:
            if self.floating_widget:
                self.floating_widget.hide()
            self.show()
        
        self._guardar_settings()

    def _restaurar_desde_widget(self):
        # Desactivar switch (esto disparará _toggle_modo_widget(False) automáticamente)
        self.query_window.right_bar.sw_widget.setChecked(False)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    navigator = Navigator()
    navigator.show()
    sys.exit(app.exec())
