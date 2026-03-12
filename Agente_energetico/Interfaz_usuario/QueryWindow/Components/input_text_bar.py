import sys
import os
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QLineEdit, QPushButton
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont

# Asegurar visibilidad de Temas
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from Temas import Tema

class InputTextBar(QWidget):
    """
    Componente central de la ventana de consulta.
    Contiene el título de ayuda, el campo de entrada y el botón de consulta.
    """
    consultar_presionado = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignCenter)
        layout.setSpacing(20)

        # Título arriba de la barra
        self.titulo = QLabel("¿En qué puedo ayudarte?")
        self.titulo.setStyleSheet(f"color: {Tema.TEXTO_SUAVE}; font-size: 26px; font-weight: bold; background: transparent;")
        self.titulo.setAlignment(Qt.AlignCenter)
        
        # Campo de entrada
        self.input_field = QLineEdit()
        self.input_field.setPlaceholderText("Describe tu consulta energética...")
        self.input_field.setMinimumHeight(60)
        self.input_field.setFixedWidth(500)
        self.input_field.setStyleSheet("""
            QLineEdit {
                background-color: rgba(255, 255, 255, 20);
                border: 2px solid rgba(255, 255, 255, 50);
                border-radius: 15px;
                color: white;
                font-size: 18px;
                padding: 10px;
            }
            QLineEdit:focus { border: 2px solid rgba(255, 150, 100, 150); }
        """)

        # Botón de consulta
        self.btn_consultar = QPushButton("Consultar")
        self.btn_consultar.setFixedSize(200, 50)
        self.btn_consultar.setCursor(Qt.PointingHandCursor)
        self.btn_consultar.setFont(QFont(*Tema.FUENTE_BOTON))
        self.btn_consultar.setStyleSheet(f"""
            QPushButton {{ 
                background-color: {Tema.AMARILLO}; 
                color: #2D1E14; 
                border-radius: 10px; 
            }}
            QPushButton:hover {{ background-color: {Tema.AMARILLO_HOVER}; }}
        """)
        self.btn_consultar.clicked.connect(self.consultar_presionado.emit)

        layout.addWidget(self.titulo)
        layout.addWidget(self.input_field)
        layout.addSpacing(10)
        layout.addWidget(self.btn_consultar, 0, Qt.AlignCenter)

    def text(self):
        return self.input_field.text()
    
    def limpiar(self):
        self.input_field.clear()
