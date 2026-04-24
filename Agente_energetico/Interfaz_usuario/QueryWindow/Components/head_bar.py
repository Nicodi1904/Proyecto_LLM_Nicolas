import sys
import os
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel
from PySide6.QtCore import Qt, QTimer, QDateTime, QLocale

# Asegurar visibilidad de los módulos en la raíz (Interfaz_usuario)
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from Temas import Tema

class HeadBar(QFrame):
    """
    Componente superior de la ventana.
    Muestra la fecha actual a la izquierda y la hora del sistema a la derecha.
    Ocupa aproximadamente 1/4 (o lo definido en el layout de la ventana) de la altura.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()
        self._iniciar_reloj()

    def _setup_ui(self):
        """Configura el diseño visual y las etiquetas de fecha/hora."""
        self.setFixedHeight(150) # Altura relativa
        self.setStyleSheet("background: transparent; border: none;")
        
        layout = QHBoxLayout(self)
        layout.setContentsMargins(40, 20, 40, 20)

        # Etiqueta de Fecha (Izquierda)
        self.lbl_fecha = QLabel()
        self.lbl_fecha.setStyleSheet(f"color: {Tema.TEXTO_SUAVE}; font-size: 18px; font-weight: bold;")
        
        # Etiqueta de Hora (Derecha)
        self.lbl_hora = QLabel()
        self.lbl_hora.setStyleSheet(f"color: white; font-size: 24px; font-weight: bold;")

        layout.addWidget(self.lbl_fecha, 0, Qt.AlignLeft | Qt.AlignTop)
        layout.addStretch()
        layout.addWidget(self.lbl_hora, 0, Qt.AlignRight | Qt.AlignTop)

    def _iniciar_reloj(self):
        """Crea un temporizador que actualiza la hora cada segundo."""
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._actualizar_tiempo)
        self.timer.start(1000)
        self._actualizar_tiempo() # Llamada inicial

    def _actualizar_tiempo(self):
        """Obtiene la fecha y hora actual del sistema y las muestra."""
        ahora = QDateTime.currentDateTime()
        locale = QLocale(QLocale.Spanish, QLocale.Colombia)
        self.lbl_fecha.setText(locale.toString(ahora, "dddd d /  MMMM / yyyy").upper())
        self.lbl_hora.setText(ahora.toString("hh:mm:ss"))
    def get_time_info(self):
        """Retorna un diccionario con la fecha y hora actuales mostradas."""
        return {
            "fecha": self.lbl_fecha.text(),
            "hora": self.lbl_hora.text()
        }
