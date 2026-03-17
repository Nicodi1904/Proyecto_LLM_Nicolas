import sys
import os
from PySide6.QtWidgets import QApplication

# Agregar ruta raíz de Tesis-MAS-LLM
sys.path.append(r'c:\Users\Owner\Documents\2025-2\Tesis-MAS-LLM')

from Agente_energetico.Interfaz_usuario.QueryWindow.DbWindow.db_window import DbWindow

app = QApplication(sys.argv)
try:
    ventana = DbWindow()
    ventana.show()
    print("Ventana creada y mostrada correctamente en script de prueba.")
except Exception as e:
    print(f"ERROR AL CREAR VENTANA: {e}")
    import traceback
    traceback.print_exc()

sys.exit(app.exec())
