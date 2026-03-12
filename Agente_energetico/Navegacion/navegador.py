import sys
import os
from PySide6.QtWidgets import QApplication, QWidget, QStackedWidget, QVBoxLayout
from PySide6.QtCore import Qt
from datetime import datetime

# Asegurar visibilidad de los módulos de Interfaz_usuario
script_dir = os.path.dirname(os.path.abspath(__file__))
# Ahora que Navegacion está en Agente_energetico, el parent_dir es Agente_energetico
agente_energetico_dir = os.path.dirname(script_dir)
interfaz_dir = os.path.join(agente_energetico_dir, "Interfaz_usuario")
sys.path.append(interfaz_dir)

from QueryWindow.query_window import Query_Window
from ResultWindow.result_window import Result_Window
from Temas import Tema
from Recursos_compartidos.Recursos_entrada import QueryRequest

class Navigator(QWidget):
    """
    Controlador central de la aplicación.
    Utiliza QStackedWidget para alternar entre QueryWindow y ResultWindow.
    """
    def __init__(self):
        super().__init__()
        self.ultima_consulta = None
        self._configurar_ventana()
        self._setup_ui()
        self._conectar_senales()

    def _configurar_ventana(self):
        self.setWindowTitle("Agente Energético - MAS")
        self.resize(1000, 600)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.Window)

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

    def _navegar_a_resultados(self, consulta):
        # Capturar datos integrales desde la ventana de consulta
        datos = self.query_window.get_full_query_data()
        
        # Guardar en recursos compartidos
        self.ultima_consulta = QueryRequest(
            pregunta=datos["pregunta"],
            fecha=datos["fecha"],
            hora=datos["hora"],
            modelo=datos["modelo"],
            referencias_horarias=datos["referencias_horarias"],
            few_shots=datos["few_shots"],
            widget=datos["widget"]
        )
        
        # Pasar datos a la ventana de resultados y cambiar vista
        self.result_window.mostrar_datos(consulta, f"Procesando respuesta del agente con el modelo {datos['modelo']}...")
        self.stacked_widget.setCurrentIndex(1)

    def _navegar_a_consulta(self):
        # Descartar paquete de recursos compartidos y limpiar interfaz
        self.ultima_consulta = None
        self.query_window.limpiar_interfaz()
        self.stacked_widget.setCurrentIndex(0)

    # Permitir arrastrar la ventana
    def mousePressEvent(self, event):
        # Si se hace clic en un widget interactivo, no iniciar arrastre
        child = self.childAt(event.position().toPoint())
        if isinstance(child, (QPushButton, QLineEdit, QComboBox, QLabel)) and child.cursor().shape() == Qt.PointingHandCursor:
            return
        # Caso especial para QLineEdit y QComboBox que no siempre tienen PointingHandCursor
        if isinstance(child, (QLineEdit, QComboBox)):
            return

        if event.button() == Qt.LeftButton:
            self.drag_pos = event.globalPosition().toPoint()

    def mouseMoveEvent(self, event):
        if event.buttons() == Qt.LeftButton:
            self.move(self.pos() + event.globalPosition().toPoint() - self.drag_pos)
            self.drag_pos = event.globalPosition().toPoint()
            event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    navigator = Navigator()
    navigator.show()
    sys.exit(app.exec())
