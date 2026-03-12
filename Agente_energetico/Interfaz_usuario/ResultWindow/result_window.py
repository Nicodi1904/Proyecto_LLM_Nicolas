import sys
import os
from PySide6.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPainter

# Visibilidad de Temas
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.append(parent_dir)
sys.path.append(script_dir)
from Temas import Tema

# Componentes locales
from Components.top_banner import TopBanner
from Components.response_panel import ResponsePanel
from Components.resources_panel import ResourcesPanel
from Components.left_menu import LeftMenu

class Result_Window(QWidget):
    """
    Ventana de Resultados modular e independiente.
    Implementa su propio efecto cristal.
    """
    solicitud_regreso = Signal()

    def __init__(self):
        super().__init__()
        self._configurar_ventana()
        self._construir_interfaz()

    def _configurar_ventana(self):
        self.setWindowTitle("Resultados Energéticos - MAS")
        self.resize(1000, 600)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.Window)

    def paintEvent(self, event):
        """Dibuja el fondo de cristal redondeado directamente aquí."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setBrush(Tema.CRISTAL_BG)
        painter.setPen(Qt.NoPen)
        painter.drawRoundedRect(self.rect(), 20, 20)

    def _construir_interfaz(self):
        layout_maestro = QVBoxLayout(self)
        layout_maestro.setContentsMargins(10, 10, 10, 10)
        layout_maestro.setSpacing(10)

        self.top_banner = TopBanner()
        self.top_banner.regresar_presionado.connect(self.solicitud_regreso.emit)
        layout_maestro.addWidget(self.top_banner)

        layout_cuerpo = QHBoxLayout()
        layout_cuerpo.setSpacing(10)

        self.left_menu = LeftMenu()
        self.response_panel = ResponsePanel()
        self.resources_panel = ResourcesPanel()

        layout_cuerpo.addWidget(self.left_menu)
        layout_cuerpo.addWidget(self.response_panel, 3)
        layout_cuerpo.addWidget(self.resources_panel, 3)

        layout_maestro.addLayout(layout_cuerpo)

    def mostrar_datos(self, consulta, respuesta):
        self.top_banner.set_query_text(consulta)
        self.response_panel.set_response_text(respuesta)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = Result_Window()
    window.mostrar_datos("¿Estado de la red?", "Operación normal.")
    window.show()
    sys.exit(app.exec())
