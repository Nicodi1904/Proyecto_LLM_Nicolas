import os
from PySide6.QtWidgets import QWidget, QLabel, QVBoxLayout
from PySide6.QtCore import Qt, Signal, QPoint
from PySide6.QtGui import QPixmap, QPainter, QColor

class FloatingWidget(QWidget):
    solicitar_restauracion = Signal()

    def __init__(self, pos_inicial=QPoint(100, 100)):
        super().__init__()
        self.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground)
        
        self.setFixedSize(80, 80)
        self.move(pos_inicial)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Cargar icono
        self.lbl_icon = QLabel()
        self.lbl_icon.setScaledContents(True)
        icon_path = os.path.join(os.path.dirname(__file__), "icon.png")
        if os.path.exists(icon_path):
            pixmap = QPixmap(icon_path)
            self.lbl_icon.setPixmap(pixmap)
        else:
            self.lbl_icon.setText("⚡")
            self.lbl_icon.setStyleSheet("color: #00FFFF; font-size: 40px;")
        
        layout.addWidget(self.lbl_icon)
        
        self.oldPos = self.pos()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.oldPos = event.globalPos()

    def mouseMoveEvent(self, event):
        if event.buttons() == Qt.LeftButton:
            delta = QPoint(event.globalPos() - self.oldPos)
            self.move(self.x() + delta.x(), self.y() + delta.y())
            self.oldPos = event.globalPos()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            # Si no se movió mucho, es un click
            if abs(event.globalPos().x() - self.oldPos.x()) < 5:
                self.solicitar_restauracion.emit()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        # Dibujar un círculo sutil de fondo (cristal)
        painter.setBrush(QColor(255, 255, 255, 20))
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(0, 0, self.width(), self.height())
