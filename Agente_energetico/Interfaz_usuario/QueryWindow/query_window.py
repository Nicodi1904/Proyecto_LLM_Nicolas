import sys
import os
from PySide6.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPainter, QFont

# Asegurar visibilidad de Temas
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.append(parent_dir)
sys.path.append(script_dir)
from Temas import Tema

# Importar componentes locales
from Components.head_bar import HeadBar
from Components.input_text_bar import InputTextBar
from Components.right_bar import RightBar
from Components.status_info import StatusInfo

class Query_Window(QWidget):
    """
    Ventana de Consulta modular e independiente.
    Maneja el efecto cristal y la organización de componentes.
    """
    consulta_disparada = Signal(str)

    def __init__(self):
        super().__init__()
        self._configurar_ventana()
        self._construir_interfaz()

    def _configurar_ventana(self):
        self.setWindowTitle("Consulta Energética - MAS")
        self.resize(1000, 600)
        # Se elimina FramelessWindowHint para permitir maximizar/minimizar/cerrar estándar
        # Se comenta TranslucentBackground para que el marco de Windows sea visible
        # self.setAttribute(Qt.WA_TranslucentBackground)
        self.setWindowFlags(Qt.Window)

    def paintEvent(self, event):
        """Dibuja el fondo de cristal redondeado directamente aquí."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setBrush(Tema.CRISTAL_BG)
        painter.setPen(Qt.NoPen)
        painter.drawRoundedRect(self.rect(), 20, 20)

    def _construir_interfaz(self):
        layout_maestro = QHBoxLayout(self)
        layout_maestro.setContentsMargins(0, 0, 0, 0)
        layout_maestro.setSpacing(0)

        # ZONA IZQUIERDA
        panel_izquierdo = QWidget()
        layout_izquierdo = QVBoxLayout(panel_izquierdo)
        layout_izquierdo.setContentsMargins(0, 0, 0, 0)
        layout_izquierdo.setSpacing(0)

        self.head_bar = HeadBar()
        
        self.cuerpo_central = QWidget()
        layout_centro = QVBoxLayout(self.cuerpo_central)
        
        self.input_bar = InputTextBar()
        self.input_bar.consultar_presionado.connect(self._al_hacer_click_consultar)
        
        self.status_info = StatusInfo()
        
        self.lbl_estado = QLabel("")
        self.lbl_estado.setStyleSheet("color: white; font-style: italic;")
        self.lbl_estado.setAlignment(Qt.AlignCenter)

        layout_centro.addStretch(1)
        layout_centro.addWidget(self.input_bar)
        layout_centro.addWidget(self.status_info)
        layout_centro.addWidget(self.lbl_estado, 0, Qt.AlignCenter)
        layout_centro.addStretch(2)

        layout_izquierdo.addWidget(self.head_bar)
        layout_izquierdo.addWidget(self.cuerpo_central)

        # ZONA DERECHA
        self.right_bar = RightBar()

        layout_maestro.addWidget(panel_izquierdo, 1)
        layout_maestro.addWidget(self.right_bar)

    def _al_hacer_click_consultar(self):
        texto = self.input_bar.text().strip()
        modelo = self.status_info.get_selected_model()
        
        # VALIDACIÓN: Si no hay un modelo válido seleccionado, advertir y abortar
        if not modelo or modelo == "Sin modelos":
            self.lbl_estado.setText("Por favor, agrega o selecciona un modelo válido antes de consultar.")
            return

        if texto:
            self.lbl_estado.setText(f"Enviando consulta...")
            self.input_bar.titulo.setText("Procesando datos de consulta, dame un momento...")
            self.consulta_disparada.emit(texto)
        else:
            self.lbl_estado.setText("Por favor, escribe algo.")

    def get_full_query_data(self):
        """
        Recopila toda la información de los componentes hijo para armar el QueryRequest.
        """
        tiempo = self.head_bar.get_time_info()
        config_derecha = self.right_bar.get_config_data()
        
        return {
            "pregunta": self.input_bar.text().strip(),
            "fecha": tiempo["fecha"],
            "hora": tiempo["hora"],
            "modelo": self.status_info.get_selected_model(),
            "referencias_horarias": config_derecha["referencias"],
            "few_shots": config_derecha["few_shots"],
            "widget": config_derecha["widget"]
        }

    def limpiar_interfaz(self):
        self.input_bar.limpiar()
        self.lbl_estado.setText("")
        self.input_bar.titulo.setText("¿En qué puedo ayudarte?")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = Query_Window()
    window.show()
    sys.exit(app.exec())
