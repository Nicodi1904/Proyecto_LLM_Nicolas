import sys
import os
from PySide6.QtWidgets import QFrame, QVBoxLayout, QLabel
from PySide6.QtCore import Qt

# Visibilidad de Temas
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from Temas import Tema

class ResourcesPanel(QFrame):
    """
    Panel lateral de recursos.
    Estilizado directamente para ser independiente.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        self.setStyleSheet("background-color: rgba(255, 255, 255, 15); border: 1px solid rgba(255, 255, 255, 30); border-radius: 15px;")
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(15, 15, 15, 15)

        self.titulo = QLabel("RECURSOS Y DATOS")
        self.titulo.setStyleSheet("color: white; font-weight: bold; font-size: 22px; border: none; background: transparent;")
        
        self.scroll_area = QFrame()
        self.scroll_area.setStyleSheet("background: rgba(255, 255, 255, 5); border-radius: 10px;")
        
        layout.addWidget(self.titulo)
        layout.addWidget(self.scroll_area, 1)
