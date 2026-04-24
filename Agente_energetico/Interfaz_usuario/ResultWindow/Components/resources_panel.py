import sys
import os
from PySide6.QtWidgets import QFrame, QVBoxLayout, QLabel, QScrollArea, QWidget
from PySide6.QtCore import Qt

from .plotly_viewer import PlotlyViewer

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
        
        # Área de Scroll Real
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setStyleSheet("""
            QScrollArea {
                background: rgba(255, 255, 255, 5); 
                border-radius: 10px;
                border: none;
            }
            QScrollBar:vertical {
                border: none;
                background: rgba(0, 0, 0, 50);
                width: 10px;
                border-radius: 5px;
            }
            QScrollBar::handle:vertical {
                background: rgba(255, 255, 255, 100);
                min-height: 20px;
                border-radius: 5px;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                border: none;
                background: none;
            }
        """)

        # Contenedor interno para apilar gráficos
        self.contenedor_graficos = QWidget()
        self.contenedor_graficos.setStyleSheet("background: transparent;")
        self.layout_graficos = QVBoxLayout(self.contenedor_graficos)
        self.layout_graficos.setContentsMargins(5, 5, 5, 5)
        self.layout_graficos.setSpacing(15)
        self.layout_graficos.setAlignment(Qt.AlignTop)
        
        # Enlazar contenedor al scroll
        self.scroll_area.setWidget(self.contenedor_graficos)
        
        layout.addWidget(self.titulo)
        layout.addWidget(self.scroll_area, 1)

    def set_title(self, titulo_texto):
        self.titulo.setText(str(titulo_texto).upper())

    def display_text(self, texto):
        """Muestra texto en crudo/JSON en lugar de gráficas."""
        for i in reversed(range(self.layout_graficos.count())): 
            item = self.layout_graficos.itemAt(i)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
            else:
                self.layout_graficos.removeItem(item)
                
        lbl = QLabel(str(texto))
        lbl.setStyleSheet("color: rgba(255, 255, 255, 220); font-size: 14px; background: transparent; font-family: Consolas, monospace;")
        lbl.setWordWrap(True)
        lbl.setTextInteractionFlags(Qt.TextSelectableByMouse)
        lbl.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.layout_graficos.addWidget(lbl)

    def display_graphs(self, reporte_worker3: dict):
        """
        Recibe el diccionario de gráficas generadas por Worker3 y las muestra
        en el panel derecho de la interfaz.
        """
        # Limpiar contenedor previo por si se hacen búsquedas consecutivas
        for i in reversed(range(self.layout_graficos.count())): 
            widget = self.layout_graficos.itemAt(i).widget()
            if widget is not None:
                widget.setParent(None)

        graficas_agregadas = 0

        for req_id, acciones in reporte_worker3.items():
            for accion in acciones:
                if "figura" in accion:
                    # Crear contenedor para título y gráfico
                    marco = QFrame()
                    marco.setStyleSheet("background: rgba(0, 0, 0, 40); border-radius: 8px;")
                    l_marco = QVBoxLayout(marco)
                    
                    titulo = QLabel(f"Resultados para: {accion.get('descripcion', 'Operación')}")
                    titulo.setStyleSheet("color: white; font-weight: bold; font-size: 14px; background: transparent;")
                    titulo.setAlignment(Qt.AlignCenter)
                    titulo.setWordWrap(True)
                    
                    visor = PlotlyViewer()
                    visor.set_figure(accion["figura"])
                    
                    l_marco.addWidget(titulo)
                    l_marco.addWidget(visor)
                    
                    self.layout_graficos.addWidget(marco)
                    graficas_agregadas += 1

        if graficas_agregadas == 0:
            lbl_vacia = QLabel("No hay recursos gráficos para mostrar en esta solicitud.")
            lbl_vacia.setStyleSheet("color: rgba(255,255,255,150); font-style: italic;")
            lbl_vacia.setAlignment(Qt.AlignCenter)
            self.layout_graficos.addWidget(lbl_vacia)
