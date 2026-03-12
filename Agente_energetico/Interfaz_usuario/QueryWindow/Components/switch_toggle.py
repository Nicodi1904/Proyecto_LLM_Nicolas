from PySide6.QtWidgets import QAbstractButton
from PySide6.QtCore import Qt, QPropertyAnimation, QRect, Property, Signal, QEasingCurve
from PySide6.QtGui import QPainter, QColor

class SwitchToggle(QAbstractButton):
    """
    Interruptor deslizante personalizado con animaciones.
    """
    def __init__(self, parent=None, width=45, bg_color="#777777", circle_color="#FFFFFF", active_color="#FFB300"):
        super().__init__(parent)
        self.setCheckable(True)
        self.setFixedSize(width, 24)
        
        self._bg_color = bg_color
        self._circle_color = circle_color
        self._active_color = active_color
        
        self._circle_position = 3
        self.animation = QPropertyAnimation(self, b"circle_position")
        self.animation.setDuration(200)
        self.animation.setEasingCurve(QEasingCurve.InOutCubic)
        
        self.toggled.connect(self._start_animation)

    @Property(float)
    def circle_position(self):
        return self._circle_position

    @circle_position.setter
    def circle_position(self, pos):
        self._circle_position = pos
        self.update()

    def _start_animation(self, checked):
        self.animation.stop()
        if checked:
            self.animation.setEndValue(self.width() - 21)
        else:
            self.animation.setEndValue(3)
        self.animation.start()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        
        # Color de fondo según estado
        bg = QColor(self._active_color) if self.isChecked() else QColor(self._bg_color)
        p.setBrush(bg)
        p.setPen(Qt.NoPen)
        
        # Dibujar carril (fondo redondeado)
        p.drawRoundedRect(0, 0, self.width(), self.height(), self.height() / 2, self.height() / 2)
        
        # Dibujar círculo deslizante
        p.setBrush(QColor(self._circle_color))
        p.drawEllipse(self._circle_position, 3, 18, 18)
        
        p.end()
