#Le pasamos a la función de carga la ruta del archivo y luego la leemos

#Ella obtiene 
import pandas as pd

def cargar_desde_json(ruta_archivo):
    df = pd.read_json(ruta_archivo)
    return df
def buscarConsumo(datos=None,fecha=None,dispositivo=None):
        #Otra forma de buscar la fila
        """
        numfilas=datos.shape[0] 
        dealer1=datos.iloc
        for i in range(0,numfilas):
            if dealer1[i]["fecha"]==fecha and dealer1[i]["dispositivo"]==dispositivo:
                dealer=i
                break 
            """
        #Devuelve una lista de las filas que se cumplen y no cumplen la condición
        filtro = (datos["fecha"] == fecha) & (datos["dispositivo"] == dispositivo)
         # Si no hay coincidencias
        if filtro.sum() == 0:
            return None
        fila=datos.index[filtro][0]
        consumo=datos.iloc[fila]['consumo_kwh']
        return consumo
# Ejecución de prueba
if __name__ == "__main__": #Esto hace que solo se ejecute si se compila desde este archivo, si se compila desde algún otro esta parte no se compilará
    archivo = "consumo.json"
    df = cargar_desde_json(archivo)
    print("Datos desde JSON:")
    print(df.shape,"\n")
    #print(df.iloc[0]["dispositivo"])
    #print(df.size) (4, 3) 
    
    fecha="2025-08-02"
    dispositivo="Aire acondicionado"
    mask = (df["fecha"] == fecha) & (df["dispositivo"] == dispositivo)
    #print(mask)
   
    print (buscarConsumo(df,fecha,dispositivo))