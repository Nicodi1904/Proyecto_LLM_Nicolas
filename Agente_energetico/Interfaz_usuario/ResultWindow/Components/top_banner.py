import sys
import os
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QPushButton
from PySide6.QtGui import QFont
from PySide6.QtCore import Qt, Signal

# Visibilidad de Temas
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from Temas import Tema

class TopBanner(QFrame):
    """
    Banner superior de resultados.
    Lógica de visualización y botón de regreso incluidos localmente.
    """
    regresar_presionado = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        self.setFixedHeight(80)
        self.setStyleSheet("background: transparent; border: none;")
        
        layout = QHBoxLayout(self)
        layout.setContentsMargins(20, 0, 20, 0)

        # Caja de consulta
        self.caja_consulta = QFrame()
        self.caja_consulta.setFixedHeight(50)
        self.caja_consulta.setStyleSheet("background-color: rgba(255, 255, 255, 15); border: 1px solid rgba(255, 255, 255, 30); border-radius: 15px;")
        
        layout_caja = QHBoxLayout(self.caja_consulta)
        self.lbl_consulta = QLabel("CONSULTA: ...")
        self.lbl_consulta.setStyleSheet("color: white; font-weight: bold; font-size: 14px; border: none; background: transparent;")
        layout_caja.addWidget(self.lbl_consulta)
        
        # Botón Volver (Ahora Hacer otra consulta)
        self.btn_volver = QPushButton("Hacer otra consulta")
        self.btn_volver.setFixedSize(220, 40)
        self.btn_volver.setFont(QFont(*Tema.FUENTE_BOTON))
        self.btn_volver.setCursor(Qt.PointingHandCursor)
        self.btn_volver.setStyleSheet(f"""
            QPushButton {{ 
                background-color: rgba(255, 255, 255, 20); 
                color: {Tema.AMARILLO}; 
                border: 1px solid {Tema.MARCO_WHITE}; 
                border-radius: 10px; 
            }}
            QPushButton:hover {{ background-color: rgba(255, 255, 255, 40); }}
        """)
        self.btn_volver.clicked.connect(self.regresar_presionado.emit)

        layout.addWidget(self.caja_consulta, 1)
        layout.addWidget(self.btn_volver)

    def set_query_text(self, texto):
        self.lbl_consulta.setText(f"CONSULTA: {texto}")
