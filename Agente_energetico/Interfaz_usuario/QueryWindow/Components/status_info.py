from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QPushButton
from PySide6.QtCore import Qt, Signal

class StatusInfo(QWidget):
    """
    Componente que muestra información del modelo y servidores conectados.
    Ubicación sugerida: Debajo de la input_text_bar.
    """
    editar_modelo_solicitado = Signal()
    eliminar_modelo_solicitado = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        layout_principal = QVBoxLayout(self)
        layout_principal.setContentsMargins(0, 5, 0, 5)
        layout_principal.setSpacing(8)
        
        # Estilo común para los textos de estado
        estilo_texto = "color: #BBBBBB; font-size: 17px; background: transparent; border: none; font-weight: bold;"
        
        # 1. Etiqueta "Modelo actual:"
        self.lbl_modelo = QLabel("Modelo actual:")
        self.lbl_modelo.setStyleSheet(estilo_texto)
        self.lbl_modelo.setAlignment(Qt.AlignCenter)
        
        # 2. Fila de Selección de Modelo (+ Botón Añadir)
        fila_modelo = QHBoxLayout()
        fila_modelo.setAlignment(Qt.AlignCenter)
        fila_modelo.setSpacing(10)
        
        self.combo_modelos = QComboBox()
        self.combo_modelos.addItems(["DeepSeek-R1-8b"])
        self.combo_modelos.setFixedWidth(160)
        self.combo_modelos.setStyleSheet("""
            QComboBox {
                background-color: rgba(255, 255, 255, 15);
                border: 1px solid rgba(255, 255, 255, 30);
                border-radius: 6px;
                color: white;
                font-size: 15px;
                padding: 4px 10px;
            }
            QComboBox::drop-down { border: none; }
            QComboBox QAbstractItemView {
                background-color: #2D1E14;
                color: white;
                selection-background-color: rgba(255, 179, 0, 100);
            }
        """)
        
        self.btn_add_modelo = QPushButton("+")
        self.btn_add_modelo.setFixedSize(32, 32)
        self.btn_add_modelo.setCursor(Qt.PointingHandCursor)
        self.btn_add_modelo.setStyleSheet("""
            QPushButton {
                background-color: rgba(255, 255, 255, 15);
                border: 1px solid rgba(255, 255, 255, 30);
                border-radius: 16px;
                color: #FFB300;
                font-size: 20px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: rgba(255, 255, 255, 25);
            }
        """)
        
        self.btn_editar = QPushButton("✏️")
        self.btn_editar.setFixedSize(32, 32)
        self.btn_editar.setCursor(Qt.PointingHandCursor)
        self.btn_editar.setToolTip("Editar modelo actual")
        self.btn_editar.setStyleSheet(self.btn_add_modelo.styleSheet().replace("font-size: 20px;", "font-size: 14px;"))
        self.btn_editar.clicked.connect(self.editar_modelo_solicitado.emit)

        self.btn_eliminar = QPushButton("🗑️")
        self.btn_eliminar.setFixedSize(32, 32)
        self.btn_eliminar.setCursor(Qt.PointingHandCursor)
        self.btn_eliminar.setToolTip("Eliminar modelo actual")
        self.btn_eliminar.setStyleSheet(self.btn_add_modelo.styleSheet().replace("font-size: 20px;", "font-size: 14px; color: #ff5555;"))
        self.btn_eliminar.clicked.connect(self.eliminar_modelo_solicitado.emit)

        fila_modelo.addWidget(self.combo_modelos)
        fila_modelo.addWidget(self.btn_add_modelo)
        fila_modelo.addWidget(self.btn_editar)
        fila_modelo.addWidget(self.btn_eliminar)
        
        layout_principal.addWidget(self.lbl_modelo)
        layout_principal.addLayout(fila_modelo)
    def get_selected_model(self):
        """Retorna el modelo actualmente seleccionado en el combo box."""
        return self.combo_modelos.currentText()
