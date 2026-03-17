import sys
import os
from PySide6.QtWidgets import QFrame, QVBoxLayout, QPushButton
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QPainter, QPen, QBrush

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
        self.setFixedWidth(50)
        self.setStyleSheet("background-color: rgba(255, 255, 255, 15); border: 1px solid rgba(255, 255, 255, 30); border-radius: 15px;")
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 20, 5, 20)
        layout.setSpacing(15)

        layout.addStretch()

    def paintEvent(self, event):
        super().paintEvent(event)
        
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        center_x = self.width() // 2
        margin_y = 60
        start_y = margin_y
        end_y = self.height() - margin_y
        
        if end_y > start_y:
            radio = 6
            espaciado = (end_y - start_y) / 3
            
            # Dibujar línea punteada en segmentos (para no tapar los círculos vacíos)
            pen = QPen(Qt.white, 2, Qt.DotLine)
            painter.setPen(pen)
            
            margen_linea = radio + 2 # Espacio para salir del círculo
            for i in range(3):
                y1 = int(start_y + i * espaciado) + margen_linea
                y2 = int(start_y + (i + 1) * espaciado) - margen_linea
                if y2 > y1:
                    painter.drawLine(center_x, y1, center_x, y2)
            
            from PySide6.QtGui import QColor
            color_rosquilla = QColor(Tema.AMARILLO)
            
            # Dibujar los 4 círculos como rosquillas
            # Usamos un Pen grueso y sin Brush (fondo transparente)
            grosor_rosquilla = 3
            pen_rosquilla = QPen(color_rosquilla, grosor_rosquilla)
            painter.setPen(pen_rosquilla)
            painter.setBrush(Qt.NoBrush)
            
            for i in range(4):
                cy = int(start_y + i * espaciado)
                painter.drawEllipse(center_x - radio, cy - radio, radio * 2, radio * 2)
