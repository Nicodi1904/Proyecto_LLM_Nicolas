import sqlite3

db_path = r'C:\sqlite_tesis\Base_datos_tesis\Hogar_Sincelejo.db'

print(f"Inspeccionando tabla en: {db_path}")

try:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Obtener el comando CREATE TABLE original que creó el usuario
    cursor.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='Modelos_lenguaje'")
    sql = cursor.fetchone()
    print("\n--- SQL ORIGINAL DE CREACIÓN ---")
    print(sql[0] if sql else "Tabla no encontrada.")
    
    # Obtener info de tabla
    cursor.execute("PRAGMA table_info(Modelos_lenguaje)")
    info = cursor.fetchall()
    print("\n--- COLUMNAS Y CONSTRAINTS (cid, name, type, notnull, dflt_value, pk) ---")
    for col in info:
        print(col)
        
    conn.close()

except Exception as e:
    print(f"Error inspeccionando tabla: {e}")
