import sys
import os
from PySide6.QtWidgets import QFrame, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QWidget, QGridLayout, QLineEdit, QComboBox
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont, QIntValidator

# Asegurar visibilidad de los módulos en la raíz (Interfaz_usuario)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from Temas import Tema
from QueryWindow.Components.switch_toggle import SwitchToggle

class RightBar(QFrame):
    """
    Panel lateral derecho interactivo.
    
    Diseño solicitado:
    - El engranaje debe ir JUSTO ARRIBA de la flechita.
    - La flecha debe estar en el borde izquierdo centrado del menú.
    - El título 'CONFIGURACIÓN' debe estar en la parte superior del menú centrado.
    """
    toggled = Signal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.expandida = False
        self._setup_ui()

    def _setup_ui(self):
        """Configura el layout horizontal con la columna de control y el área de menú."""
        self.setFixedWidth(60)
        self.setStyleSheet(f"""
            QFrame {{
                background-color: rgba(255, 255, 255, 12); 
                border-left: 1px solid {Tema.MARCO_WHITE};
                border-radius: 0px; 
            }}
            QPushButton {{ 
                border: none; 
                background: transparent; 
                color: white;
            }}
        """)
        
        self.layout_maestro = QHBoxLayout(self)
        self.layout_maestro.setContentsMargins(0, 0, 0, 0)
        self.layout_maestro.setSpacing(0)

        # ── 1. COLUMNA DE CONTROL (Engranaje y Flecha) ──────────────────────
        self.columna_control = QWidget()
        self.columna_control.setFixedWidth(60)
        layout_control = QVBoxLayout(self.columna_control)
        layout_control.setContentsMargins(0, 0, 0, 0)
        layout_control.setSpacing(0) # Sin espacio entre engranaje y flecha

        # Botón Flecha (Toggle / Centrada verticalmente)
        self.btn_toggle = QPushButton("◀")
        self.btn_toggle.setFixedSize(60, 60)
        self.btn_toggle.setCursor(Qt.PointingHandCursor)
        self.btn_toggle.setStyleSheet(f"color: {Tema.AMARILLO}; font-size: 22px; font-weight: bold; border: none; background: transparent;")
        self.btn_toggle.clicked.connect(self._toggle_menu)

        # Para centrar la Flecha verticalmente:
        layout_control.addStretch(1)
        layout_control.addWidget(self.btn_toggle, 0, Qt.AlignCenter)
        layout_control.addStretch(1)

        # ── 2. ÁREA DE MENÚ (Expandible) ────────────────────────────────────
        self.area_menu = QWidget()
        self.area_menu.setVisible(False)
        layout_menu = QVBoxLayout(self.area_menu)
        layout_menu.setContentsMargins(15, 30, 15, 10) 
        layout_menu.setSpacing(10)

        # Título centrado en la parte superior
        self.lbl_titulo_menu = QLabel("CONFIGURACIÓN")
        self.lbl_titulo_menu.setStyleSheet("color: white; font-weight: bold; font-size: 24px; background: transparent;")
        self.lbl_titulo_menu.setAlignment(Qt.AlignCenter)
        
        # Sub-títulos de categorías con íconos
        estilo_subtitulo = "color: #CCCCCC; font-size: 20px; font-weight: bold; margin-top: 20px; background: transparent;"
        
        self.lbl_ref_horarias = QLabel("☀️🌙 Referencias horarias")
        self.lbl_ref_horarias.setStyleSheet(estilo_subtitulo)
        
        # Contenedor para las filas horarias
        self.grid_horarios = QGridLayout()
        self.grid_horarios.setSpacing(10) # Espaciado aumentado para fuentes grandes
        self.grid_horarios.setContentsMargins(0, 5, 0, 5)
        self._set_up_filas_horarias()

        self.lbl_optimizadores = QLabel("🔧 Optimizadores")
        self.lbl_optimizadores.setStyleSheet(estilo_subtitulo)
        
        # Fila Few-Shots
        self.layout_few_shots = QHBoxLayout()
        self.lbl_few_shots = QLabel("Few-Shots")
        self.lbl_few_shots.setStyleSheet("color: #BBBBBB; font-size: 17px; background: transparent;")
        self.sw_few_shots = SwitchToggle(active_color=Tema.AMARILLO, width=50)
        self.layout_few_shots.addWidget(self.lbl_few_shots)
        self.layout_few_shots.addStretch()
        self.layout_few_shots.addWidget(self.sw_few_shots)

        self.lbl_personalizacion = QLabel("🧩 Personalización")
        self.lbl_personalizacion.setStyleSheet(estilo_subtitulo)
        
        # Fila Widget
        self.layout_widget_sw = QHBoxLayout()
        self.lbl_widget_text = QLabel("Widget")
        self.lbl_widget_text.setStyleSheet("color: #BBBBBB; font-size: 17px; background: transparent;")
        self.sw_widget = SwitchToggle(active_color=Tema.AMARILLO, width=50)
        self.layout_widget_sw.addWidget(self.lbl_widget_text)
        self.layout_widget_sw.addStretch()
        self.layout_widget_sw.addWidget(self.sw_widget)

        layout_menu.addWidget(self.lbl_titulo_menu)
        layout_menu.addSpacing(15)
        layout_menu.addWidget(self.lbl_ref_horarias)
        layout_menu.addLayout(self.grid_horarios)
        layout_menu.addSpacing(15)
        layout_menu.addWidget(self.lbl_optimizadores)
        layout_menu.addLayout(self.layout_few_shots)
        layout_menu.addSpacing(15)
        layout_menu.addWidget(self.lbl_personalizacion)
        layout_menu.addLayout(self.layout_widget_sw)
        layout_menu.addStretch(1)

        # Ensamblar
        self.layout_maestro.addWidget(self.columna_control)
        self.layout_maestro.addWidget(self.area_menu, 1)

    def _set_up_filas_horarias(self):
        """Crea las filas de configuración de tiempo detalladas por el usuario."""
        referencias = [
            ("madrugada", "#A0C4FF", "12", "00", "AM", "05", "59", "AM"),
            ("mañana", "#FFD6A5", "06", "00", "AM", "11", "59", "AM"),
            ("tarde", "#FDFFB6", "12", "00", "PM", "03", "59", "PM"),
            ("media tarde", "#CAFFBF", "04", "00", "PM", "06", "59", "PM"),
            ("noche", "#9BF6FF", "07", "00", "PM", "09", "59", "PM"),
            ("media noche", "#BDB2FF", "10", "00", "PM", "11", "59", "PM")
        ]

        # Estilo para los inputs de tiempo
        estilo_input = """
            QLineEdit {
                background-color: rgba(255, 255, 255, 10);
                border: 1px solid rgba(255, 255, 255, 30);
                border-radius: 5px;
                color: white;
                font-size: 15px;
                padding: 2px;
            }
        """
        estilo_combo = """
            QComboBox {
                background-color: rgba(255, 255, 255, 10);
                border: 1px solid rgba(255, 255, 255, 30);
                border-radius: 5px;
                color: white;
                font-size: 15px;
                padding: 2px;
            }
            QComboBox QAbstractItemView {
                background-color: #2D1E14;
                color: white;
                selection-background-color: rgba(255, 255, 255, 40);
            }
            QComboBox::drop-down { border: none; }
        """
        estilo_label_limpio = "background: transparent; border: none; color: #BBBBBB;"

        # Diccionario para guardar referencias a los widgets de entrada
        self.inputs_horarios = {}

        for i, (nombre, color, h_i, m_i, ampm_i, h_f, m_f, ampm_f) in enumerate(referencias):
            # Etiqueta de la referencia
            lbl = QLabel(nombre)
            lbl.setStyleSheet(f"color: {color}; font-size: 17px; font-weight: demi-bold; background: transparent; border: none;")
            
            # ── 1. INICIO (Desde) ──
            edit_h_i = QLineEdit(h_i)
            edit_h_i.setFixedWidth(30) 
            edit_h_i.setAlignment(Qt.AlignCenter)
            edit_h_i.setStyleSheet(estilo_input)
            edit_h_i.setMaxLength(2)
            edit_h_i.setValidator(QIntValidator(1, 12))
            
            lbl_sep_i = QLabel(":")
            lbl_sep_i.setStyleSheet(f"font-weight: bold; font-size: 17px; {estilo_label_limpio}")
            lbl_sep_i.setFixedWidth(8)
            
            edit_m_i = QLineEdit(m_i)
            edit_m_i.setFixedWidth(30) 
            edit_m_i.setAlignment(Qt.AlignCenter)
            edit_m_i.setStyleSheet(estilo_input)
            edit_m_i.setMaxLength(2)
            edit_m_i.setValidator(QIntValidator(0, 59))
            
            combo_i = QComboBox()
            combo_i.addItems(["AM", "PM"])
            combo_i.setCurrentText(ampm_i)
            combo_i.setFixedWidth(47) # Ajustado para que no se corte el texto
            combo_i.setStyleSheet(estilo_combo)

            # ── Separador Rango ──
            lbl_rango = QLabel("-")
            lbl_rango.setStyleSheet(f"font-weight: bold; font-size: 17px; {estilo_label_limpio}")
            lbl_rango.setFixedWidth(12)

            # ── 2. FIN (Hasta) ──
            edit_h_f = QLineEdit(h_f)
            edit_h_f.setFixedWidth(30) 
            edit_h_f.setAlignment(Qt.AlignCenter)
            edit_h_f.setStyleSheet(estilo_input)
            edit_h_f.setMaxLength(2)
            edit_h_f.setValidator(QIntValidator(1, 12))
            
            lbl_sep_f = QLabel(":")
            lbl_sep_f.setStyleSheet(f"font-weight: bold; font-size: 17px; {estilo_label_limpio}")
            lbl_sep_f.setFixedWidth(8)
            
            edit_m_f = QLineEdit(m_f)
            edit_m_f.setFixedWidth(30) 
            edit_m_f.setAlignment(Qt.AlignCenter)
            edit_m_f.setStyleSheet(estilo_input)
            edit_m_f.setMaxLength(2)
            edit_m_f.setValidator(QIntValidator(0, 59))
            
            combo_f = QComboBox()
            combo_f.addItems(["AM", "PM"])
            combo_f.setCurrentText(ampm_f)
            combo_f.setFixedWidth(47) # Ajustado
            combo_f.setStyleSheet(estilo_combo)

            # Guardar referencias en tupla ampliada (6 elementos)
            self.inputs_horarios[nombre] = (edit_h_i, edit_m_i, combo_i, edit_h_f, edit_m_f, combo_f)

            # --- CORRECCIÓN DE TAMAÑOS PARA AJUSTAR AL ESPACIO ---
            lbl.setFixedWidth(105) 
            lbl.setStyleSheet(f"color: {color}; font-size: 16px; font-weight: demi-bold; background: transparent;")
            
            from PySide6.QtWidgets import QHBoxLayout, QWidget
            
            l_ini = QHBoxLayout()
            l_ini.setContentsMargins(0, 0, 0, 0)
            l_ini.setSpacing(2)
            l_ini.addWidget(edit_h_i)
            l_ini.addWidget(lbl_sep_i)
            l_ini.addWidget(edit_m_i)
            l_ini.addWidget(combo_i)
            
            w_ini = QWidget()
            w_ini.setLayout(l_ini)
            w_ini.setFixedWidth(125) # Devuelto a 125 para alojar combobox más ancho

            l_fin = QHBoxLayout()
            l_fin.setContentsMargins(0, 0, 0, 0)
            l_fin.setSpacing(2)
            l_fin.addWidget(edit_h_f)
            l_fin.addWidget(lbl_sep_f)
            l_fin.addWidget(edit_m_f)
            l_fin.addWidget(combo_f)
            
            w_fin = QWidget()
            w_fin.setLayout(l_fin)
            w_fin.setFixedWidth(125) # Devuelto a 125

            lbl_rango.setAlignment(Qt.AlignCenter)
            lbl_rango.setFixedWidth(15)

            # Agregar al grid
            self.grid_horarios.addWidget(lbl, i, 0)
            self.grid_horarios.addWidget(w_ini, i, 1)
            self.grid_horarios.addWidget(lbl_rango, i, 2)
            self.grid_horarios.addWidget(w_fin, i, 3)

    def get_config_data(self):
        """Retorna un diccionario con todas las configuraciones creadas en formato rango."""
        horarios = {}
        for nombre, (h_i, m_i, ampm_i, h_f, m_f, ampm_f) in self.inputs_horarios.items():
            horarios[nombre] = f"{h_i.text()}:{m_i.text()} {ampm_i.currentText()} - {h_f.text()}:{m_f.text()} {ampm_f.currentText()}"
        
        return {
            "referencias": horarios,
            "few_shots": self.sw_few_shots.isChecked(),
            "widget": self.sw_widget.isChecked()
        }

    def _toggle_menu(self):
        """Alterna el estado del menú lateral."""
        self.expandida = not self.expandida
        
        if self.expandida:
            self.setFixedWidth(460) # Aumentado de 380 a 460 para evitar solapamientos
            self.btn_toggle.setText("▶")
            self.area_menu.setVisible(True)
        else:
            self.setFixedWidth(60)
            self.btn_toggle.setText("◀")
            self.area_menu.setVisible(False)
        
        self.toggled.emit(self.expandida)
