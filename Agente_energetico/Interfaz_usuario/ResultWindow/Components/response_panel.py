import sys
import os
from PySide6.QtWidgets import QFrame, QVBoxLayout, QLabel, QTextEdit
from PySide6.QtCore import Qt

# Visibilidad de Temas
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from Temas import Tema

class ResponsePanel(QFrame):
    """
    Panel de visualización de respuesta.
    Sustituye a CajaCristal definiendo su propia estética.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        self.setStyleSheet("background-color: rgba(255, 255, 255, 15); border: 1px solid rgba(255, 255, 255, 30); border-radius: 15px;")
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(15, 15, 15, 15)

        self.titulo = QLabel("RESPUESTA")
        self.titulo.setStyleSheet("color: white; font-weight: bold; font-size: 22px; border: none; background: transparent;")
        
        self.txt_respuesta = QTextEdit()
        self.txt_respuesta.setReadOnly(True)
        self.txt_respuesta.setStyleSheet("""
            QTextEdit {
                background: transparent;
                border: none;
                color: white;
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

        layout.addWidget(self.titulo)
        layout.addWidget(self.txt_respuesta)

    def set_response_text(self, respuesta):
        if isinstance(respuesta, dict):
            html_content = ""
            
            # Función auxiliar para agregar contenido como párrafos
            def add_section(contenido):
                nonlocal html_content
                if contenido:
                    # Reemplazamos saltos de línea con <br>
                    html_texto = str(contenido).replace('\\n', '<br>')
                    html_content += f'<p style="color: white; margin-top: 0px; margin-bottom: 20px; font-size: 16px; line-height: 1.5;">{html_texto}</p>'

            add_section(respuesta.get("Resumen_op"))
            add_section(respuesta.get("Resultados_op"))
            add_section(respuesta.get("analisis"))
            add_section(respuesta.get("sugerencia"))

            self.txt_respuesta.setHtml(html_content)
        else:
            self.txt_respuesta.setPlainText(str(respuesta))
