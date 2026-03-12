import sys
import os
from PySide6.QtWidgets import QFrame, QVBoxLayout, QPushButton
from PySide6.QtGui import QFont

# Visibilidad de Temas
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from Temas import Tema

class LeftMenu(QFrame):
    """
    Menú lateral de navegación.
    Botones estilizados localmente para independencia.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        self.setFixedWidth(100)
        self.setStyleSheet("background-color: rgba(255, 255, 255, 15); border: 1px solid rgba(255, 255, 255, 30); border-radius: 15px;")
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 20, 5, 20)
        layout.setSpacing(15)

        layout.addStretch()
