import sys
import os
from PySide6.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, QFrame
from PySide6.QtCore import Qt, Signal

# Visibilidad de Temas
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from Temas import Tema

class DbWindow(QDialog):
    """
    Ventana modal emergente para agregar nuevos modelos de lenguaje a la Base de Datos.
    Utiliza QDialog para bloquear la ventana QueryWindow mientras está abierta.
    """
    guardar_presionado = Signal(str, str, str)

    def __init__(self, parent=None, model="", base="", key=""):
        super().__init__(parent)
        self.model_original = model
        self.base_original = base
        self.key_original = key
        self._configurar_ventana()
        self._construir_interfaz()
        
        if model:
             self.input_model.setText(model)
             # En edición, no dejar cambiar el nombre para evitar romper la PK Unique
             self.input_model.setReadOnly(True)
             self.input_model.setStyleSheet(self.input_model.styleSheet() + "color: #888888;")
        if base:
             self.input_base.setText(base)
        if key:
             self.input_key.setText(key)

    def _configurar_ventana(self):
        self.setWindowTitle("Agregar Modelo - MAS")
        self.resize(450, 400)
        self.setWindowFlags(Qt.Dialog | Qt.WindowMinimizeButtonHint | Qt.WindowMaximizeButtonHint | Qt.WindowCloseButtonHint)
        self.setAttribute(Qt.WA_TranslucentBackground)

    def _construir_interfaz(self):
        layout_maestro = QVBoxLayout(self)
        layout_maestro.setContentsMargins(0, 0, 0, 0)

        # Marco de Cristal para efecto de profundidad
        self.marco_cristal = QFrame()
        self.marco_cristal.setStyleSheet(f"""
            background-color: rgba(15, 15, 20, 220); 
            border: 1px solid rgba(255, 255, 255, 40); 
            border-radius: 20px;
        """)
        layout_maestro.addWidget(self.marco_cristal)

        layout_interno = QVBoxLayout(self.marco_cristal)
        layout_interno.setContentsMargins(25, 20, 25, 25)
        layout_interno.setSpacing(15)

        # --- Cabecera ---
        cabecera = QHBoxLayout()
        titulo = QLabel("EDITAR CREDENCIAL" if self.model_original else "NUEVA CREDENCIAL")
        titulo.setStyleSheet("color: white; font-weight: bold; font-size: 18px; font-family: 'Outfit', sans-serif;")
        cabecera.addWidget(titulo, alignment=Qt.AlignCenter)
        layout_interno.addLayout(cabecera)

        # --- Entradas de Datos ---
        def crear_label_input(texto_label, placeholder, obscure=False):
            contenedor = QVBoxLayout()
            contenedor.setSpacing(4)
            label = QLabel(texto_label)
            label.setStyleSheet("color: #b0b0b0; font-size: 13px; font-weight: bold;")
            
            input_ext = QLineEdit()
            input_ext.setPlaceholderText(placeholder)
            input_ext.setStyleSheet(f"""
                QLineEdit {{
                    background-color: rgba(255, 255, 255, 15);
                    border: 1px solid rgba(255, 255, 255, 25);
                    border-radius: 8px;
                    color: white;
                    padding: 8px 12px;
                    font-size: 14px;
                }}
                QLineEdit:focus {{
                    border: 1px solid {Tema.AMARILLO};
                }}
            """)
            if obscure:
                 input_ext.setEchoMode(QLineEdit.Password)
                 
            contenedor.addWidget(label)
            contenedor.addWidget(input_ext)
            layout_interno.addLayout(contenedor)
            return input_ext

        self.input_model = crear_label_input("Nombre del Modelo", "Ej: llama3.1:latest o deepseek-r1:8b")
        self.input_base = crear_label_input("Api Base (URL)", "Ej: http://localhost:11434")
        self.input_key = crear_label_input("Api Key (Opcional)", "Digite su credencial...", obscure=True)

        # --- Mensaje de Error Interno ---
        self.lbl_error = QLabel("")
        self.lbl_error.setStyleSheet("color: #ff5555; font-size: 12px;")
        layout_interno.addWidget(self.lbl_error)

        # --- Botones ---
        layout_botones = QHBoxLayout()
        layout_botones.setSpacing(15)

        self.btn_cancelar = QPushButton("Cancelar")
        self.btn_cancelar.setStyleSheet("""
            QPushButton {
                background-color: rgba(255, 255, 255, 10);
                border: 1px solid rgba(255, 255, 255, 30);
                border-radius: 8px;
                color: white;
                padding: 10px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: rgba(255, 255, 255, 20);
            }
        """)
        self.btn_cancelar.clicked.connect(self.reject)

        self.btn_guardar = QPushButton("Guardar")
        self.btn_guardar.setStyleSheet(f"""
            QPushButton {{
                background: linear-gradient(135deg, #00d2ff 0%, #3a7bd5 100%);
                background-color: {Tema.AMARILLO};
                border: none;
                border-radius: 8px;
                color: black;
                padding: 10px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: {Tema.AMARILLO_HOVER};
            }}
        """)
        self.btn_guardar.clicked.connect(self._validar_y_emitir)

        layout_botones.addWidget(self.btn_cancelar)
        layout_botones.addWidget(self.btn_guardar)
        layout_interno.addLayout(layout_botones)

    def _validar_y_emitir(self):
        model = self.input_model.text().strip()
        base = self.input_base.text().strip()
        key = self.input_key.text().strip()

        if not model:
            self.lbl_error.setText("El nombre del modelo es obligatorio.")
            return

        # Solo validar HTTP si el usuario escribió algo en Api_base
        if base and not (base.startswith("http://") or base.startswith("https://")):
             self.lbl_error.setText("La Api Base debe comenzar con http:// o https://")
             return

        self.lbl_error.setText("")
        self.guardar_presionado.emit(model, base, key)
        self.accept() # Cierra el Dialog indicando éxito

    def keyPressEvent(self, event):
         # Capturar tecla Enter / Intro para que ejecute Guardar en vez de cerrar el popup
         from PySide6.QtCore import Qt
         if event.key() in (Qt.Key_Enter, Qt.Key_Return):
              self._validar_y_emitir()
         else:
              super().keyPressEvent(event)
