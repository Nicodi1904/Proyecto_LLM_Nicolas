import sys
import os
import sqlite3

# Agregar ruta raíz de Tesis-MAS-LLM
sys.path.append(r'c:\Users\Owner\Documents\2025-2\Tesis-MAS-LLM')

from Agente_energetico.Navegacion.cript import encriptar_clave

db_path = r'C:\sqlite_tesis\Base_datos_tesis\Hogar_Sincelejo.db'

print(f"Probando escritura en: {db_path}")

try:
    # 1. Probar encriptar
    key_encriptada = encriptar_clave("test_api_key_123")
    print(f"Clave encriptada con éxito: {key_encriptada}")

    # 2. Probar inserción
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Crear tabla por si las moscas
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS Modelos_lenguaje (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            Model TEXT UNIQUE NOT NULL,
            Api_base TEXT NOT NULL,
            Encripted_ApiKey TEXT
        )
    """)
    
    cursor.execute("""
        INSERT OR REPLACE INTO Modelos_lenguaje (Model, Api_base, Encripted_ApiKey)
        VALUES (?, ?, ?)
    """, ("Modo_Prueba_AI", "http://localhost:11434", key_encriptada))
    
    conn.commit()
    conn.close()
    print("¡Inserción de PRUEBA exitosa sin bloqueos!")

    # 3. Leer qué hay en la base de datos ahora
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM Modelos_lenguaje")
    rows = cursor.fetchall()
    conn.close()
    
    print("\n--- CONTENIDO ACTUAL DE LA BASE DE DATOS ---")
    for r in rows:
        print(r)

except Exception as e:
    print(f"\nERROR DETECTADO: {e}")
    import traceback
    traceback.print_exc()
