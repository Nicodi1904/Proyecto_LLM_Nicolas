import pandas as pd

def limpiar_dataset_csv(file_path: str) -> str:
    """
    Limpia un dataset de consumo energético:
    - Convierte la columna 'PC' de texto (coma decimal) a float con punto decimal.
    - Convierte la columna 'TimeStamp' a datetime.
    - Sobrescribe el archivo en la misma ubicación.
    """
    # Cargar archivo con delimitador ;
    df = pd.read_csv(file_path, delimiter=";")
    
    # Normalizar columna PC (reemplazar coma por punto y convertir a float)
    df["PC"] = df["PC"].astype(str).str.replace(",", ".").astype(float)
    
    # Convertir TimeStamp a datetime
    df["TimeStamp"] = pd.to_datetime(df["TimeStamp"], errors="coerce", dayfirst=True)
    
    # Guardar en la misma ubicación (sobreescribir el original)
    df.to_csv(file_path, sep=";", index=False)
    
    return file_path

# Ejemplo de uso:
ruta = "Energy Consumption in KWh of a Typical House Sincelejo Colombia.csv"
print("Archivo limpio guardado en:", limpiar_dataset_csv(ruta))
