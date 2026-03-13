from dataclasses import dataclass

@dataclass
class QueryRequest:
    #instanciales
    Mensaje_usuario: str #Mensaje que ingresa el usuario en la ventana de texto
    fecha: str #Fecha al momento de consultar
    hora: str #Hora al momento de consultar
    modelo: str #Modelo establecido al momento de la consulta 




    #Longevas
    referencias_horarias: dict  # { 'madrugada': '08:00 AM', ... }, diccionario que contiene las referencias horarias preferidas del usuario
    widget: bool #Indica si el usuario quiere utilizar el widget o no (Debe queda guardado tras cerrar el programa)
    few_shots: bool #Indica si se usara optimización few shots o no


