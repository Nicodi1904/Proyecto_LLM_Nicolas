import sys
import os
from PySide6.QtWidgets import QFrame, QVBoxLayout, QPushButton
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont, QPainter, QPen, QBrush, QColor

# Visibilidad de Temas
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from Temas import Tema

class LeftMenu(QFrame):
    """
    Menú lateral de navegación interactiva para los pasos multi-agente.
    """
    step_clicked = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.estados_pasos = [0, 0, 0, 0] # 0: Pendiente, 1: Exito(Verde), 2: Warning(Amarillo), 3: Error(Rojo)
        self.paso_activo = 3 # Por defecto mostrar el paso final
        self._setup_ui()

    def set_step_status(self, step_idx, status_code):
        if 0 <= step_idx < 4:
            self.estados_pasos[step_idx] = status_code
            self.update()

    def set_active_step(self, step_idx):
        if 0 <= step_idx < 4:
            self.paso_activo = step_idx
            self.update()

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
            
            # Dibujar línea punteada en segmentos
            pen = QPen(Qt.white, 2, Qt.DotLine)
            painter.setPen(pen)
            
            margen_linea = radio + 2
            for i in range(3):
                y1 = int(start_y + i * espaciado) + margen_linea
                y2 = int(start_y + (i + 1) * espaciado) - margen_linea
                if y2 > y1:
                    painter.drawLine(center_x, y1, center_x, y2)
            
            # Colores según el estado:
            # 0: Pendiente (Gris translúcido)
            # 1: Procesado (Verde)
            # 2: Reparado / Warning (Amarillo)
            # 3: Error Crítico (Rojo)
            colores = [
                QColor(255, 255, 255, 60), # 0
                QColor("#4cd137"),         # 1
                QColor(Tema.AMARILLO),     # 2
                QColor("#e84118")          # 3
            ]
            
            # Dibujar los 4 círculos
            for i in range(4):
                cy = int(start_y + i * espaciado)
                estado = self.estados_pasos[i]
                color_actual = colores[estado]
                
                # Si el paso está seleccionado, lo pintamos sólido. Si no, rosquilla o sólido suave.
                if self.paso_activo == i:
                    pen_borde = QPen(color_actual, 2)
                    painter.setPen(pen_borde)
                    painter.setBrush(color_actual)
                    painter.drawEllipse(center_x - radio - 2, cy - radio - 2, (radio + 2) * 2, (radio + 2) * 2)
                else:
                    grosor_rosquilla = 3
                    pen_rosquilla = QPen(color_actual, grosor_rosquilla)
                    painter.setPen(pen_rosquilla)
                    painter.setBrush(Qt.NoBrush)
                    painter.drawEllipse(center_x - radio, cy - radio, radio * 2, radio * 2)

    def mousePressEvent(self, event):
        """Detecta si el clic ocurrió cerca de alguno de los 4 círculos."""
        center_x = self.width() // 2
        margin_y = 60
        start_y = margin_y
        end_y = self.height() - margin_y
        espaciado = (end_y - start_y) / 3
        radio = 10 # Radio de tolerancia para el clic

        y_pos = event.pos().y()
        for i in range(4):
            cy = int(start_y + i * espaciado)
            if abs(y_pos - cy) <= radio * 2: # Área clickeable generosa
                self.step_clicked.emit(i)
                break
        
        super().mousePressEvent(event)
