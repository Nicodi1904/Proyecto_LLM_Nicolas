import pandas as pd


def cargar_CSV(file_path : str) -> pd.DataFrame: 
    """Carga el dataset limpio y devuelve un DataFrame."""
    df = pd.read_csv(file_path, delimiter=";")
    return df

def Pre_procesamiento_sincelejo(df:pd.DataFrame):
    
    #Convierte la columan de TimeStamp en objetos especiales de pandas para el TimeStamp
    df["TimeStamp"] = pd.to_datetime(df["TimeStamp"], errors="coerce")
    df = df.dropna(subset=["TimeStamp"])  # elimina filas con TimeStamp inválido

    #Elimina las columnas 6 y 7 que aparecen al cargar el dataset (Columnas basura no oficiales)
    df = df.drop(columns=["Unnamed: 6", "Unnamed: 7"])

    return df

def cargar_dataset_sinselejo(file_path : str) -> pd.DataFrame:
    
    df=Pre_procesamiento_sincelejo(cargar_CSV(file_path))

    return df








