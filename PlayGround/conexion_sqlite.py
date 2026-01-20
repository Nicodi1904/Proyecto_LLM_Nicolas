import sqlite3
import os

def obtener_columnas_dispositivos(db_path):
    """
    Se conecta a la base de datos, verifica la conexión y devuelve los nombres
    de las columnas (dispositivos) ignorando la columna de tiempo (ID 0).
    """
    if not os.path.exists(db_path):
        print(f"Error: No se logró establecer conexión. El archivo no existe en: {db_path}")
        return None

    try:
        # Intentar conectar
        mi_conexion = sqlite3.connect(db_path)
        mi_cursor = mi_conexion.cursor()
        
        # Obtener lista de tablas
        mi_cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tablas = mi_cursor.fetchall()
        
        if not tablas:
            print("Conexión establecida, pero la base de datos está vacía (sin tablas).")
            mi_conexion.close()
            return []

        columnas_finales = []
        
        for (nombre_tabla,) in tablas:
            # Obtener información de las columnas
            mi_cursor.execute(f'PRAGMA table_info("{nombre_tabla}")')
            info_columnas = mi_cursor.fetchall()
            
            # Filtrar: solo nombres de columnas donde ID > 0
            nombres = [col[1] for col in info_columnas if col[0] != 0]
            columnas_finales.extend(nombres)
            
        mi_conexion.close()
        return columnas_finales

    except Exception as e:
        print(f"Error crítico: No se logró establecer conexión con la base de datos. Detalles: {e}")
        return None

# --- Ejemplo de uso ---
if __name__ == "__main__":
    ruta = r'C:\sqlite_tesis\Base_datos_tesis\Hogar_Sincelejo.db'
    dispositivos = obtener_columnas_dispositivos(ruta)
    
    if dispositivos is not None:
        print("Conexión exitosa.")
        print("Dispositivos detectados (columnas > 0):")
        print(dispositivos)
