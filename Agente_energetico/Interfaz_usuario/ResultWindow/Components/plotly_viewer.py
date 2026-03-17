from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtWidgets import QSizePolicy
from PySide6.QtCore import Qt
import plotly.io as pio

class PlotlyViewer(QWebEngineView):
    """
    Componente personalizado para renderizar gráficos de Plotly
    nativamente dentro de PySide6.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        # Establecer una altura mínima para garantizar que el gráfico se vea bien
        self.setMinimumHeight(400)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        
        # Hacer que el fondo del visor web sea totalmente transparente
        self.page().setBackgroundColor(Qt.transparent)
        self.setStyleSheet("background: transparent; border: none;")

    def set_figure(self, fig):
        """
        Toma un objeto go.Figure, lo convierte a HTML crudo con
        Plotly.js y lo inyecta en el visor web.
        """
        # include_plotlyjs='cdn' es importante para que el archivo no pese 3MB de texto crudo
        # al inyectar todas las dependencias javascript de plotly locales.
        # full_html=True devuelve el documento completo estructurado.
        html_crudo = pio.to_html(fig, include_plotlyjs='cdn', full_html=True)
        self.setHtml(html_crudo)
