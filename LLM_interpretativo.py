import time

start = time.time()

# Código que quieres medir
for i in range(1000000):
    pass

import dspy

# 1️⃣ Conectar DSPy con Ollama (modelo llama3.1)
lm = dspy.LM(
    model="ollama_chat/llama3.1",
    api_base="http://localhost:11434",
    api_key=""
)
dspy.configure(lm=lm)

class InterpretarConsulta(dspy.Signature):
    """
    Analiza el mensaje del usuario para determinar su intención
    y extraer datos clave de la consulta.
    """

    # Entrada
    mensaje_usuario = dspy.InputField(
        desc="Texto crudo que escribió el usuario (o transcripción)"
    )
    idioma = dspy.InputField(
        desc="Código ISO 639-1 del idioma del mensaje (ej. 'es')",
        default="es"
    )
    timestamp = dspy.InputField(
        desc="Fecha/hora del mensaje en ISO-8601 (opcional)",
        default=None
    )
    historial_reducido = dspy.InputField(
        desc="Historial conversacional reducido (últimos N intercambios)",
        default=None
    )

    # Salida (entregable al razonador)
    entidades = dspy.OutputField(
        desc=(
            "Lista de entidades detectadas. "
            "Formato: lista de dicts. "
            "Keys obligatorias: type (tipo de entidad), text (texto original), "
            "normalized (valor normalizado), canonical_id (ID canónico si aplica, string o null), "
            "span (posición en texto), confidence (0.0–1.0), metadata (dict con info adicional). "
            "Ejemplo: [{'type':'dispositivo','text':'Nevera','normalized':'nevera',"
            "'canonical_id':'dev_123','span':{'start':3,'end':9},'confidence':0.98,"
            "'metadata':{'color':'blanco'}}]"
        ),
        default=[]
    )








############################################################################

# 3️⃣ Crear el predictor
interpretar = dspy.Predict(InterpretarConsulta)
mensaje_usuario= "Hola llama cómo estás, quería preguntarme una cosa: ¿Cuánto consumió el aire acondicionado el 19 de abril?"
# 4️⃣ Ejemplo de prueba
if __name__ == "__main__":
    prueba = interpretar(mensaje_usuario=mensaje_usuario)
    print("------------Mensaje del usuario----------\n",mensaje_usuario,"\n---------Respuesta_LLM---------")
    print("Intención detectada:", prueba.intencion)
    print("Entidades encontradas:", prueba.entidades)


end = time.time()
print("Tiempo transcurrido:", end - start, "segundos")

#############################################################################

raw_parse = dspy.OutputField(desc="Parseo intermedio sin normalizar (útil para trazabilidad)", default=None)
parse_version = dspy.OutputField(desc="Versión del parser/prompt usado", default=None)

