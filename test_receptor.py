import dspy
from test_dump import system_summary


class Receptor(dspy.Signature):
    """
    Detecta y estructura en lenguaje natural las intenciones explícitas e implícitas 
    del usuario, evitando generar datos inventados y manteniendo toda la información 
    tal como el usuario la expresó.
    """

    # ============================
    # ENTRADAS
    # ============================
    pregunta: str = dspy.InputField(
        desc="Mensaje original del usuario sin modificaciones."
    )

    feedback: dict = dspy.InputField(
        default={},
        desc="Contexto previo relevante (historial, preferencias, restricciones, dispositivos conocidos)."
    )

    system_summary: dict = dspy.InputField(
        default={},
        desc="Conjunto de herramientas disponibles para el sistema."
    )

    # ============================
    # SALIDAS
    # ============================
    intenciones_principales: list = dspy.OutputField(
        desc=(
            "Lista de intenciones principales expresadas en lenguaje natural. "
            "Una intención principal puede ser simple o compuesta. "
            "Si el usuario solicita varios datos o pasos interrelacionados como parte del mismo propósito "
            "(por ejemplo, obtener consumos y luego compararlos), todo debe considerarse una única "
            "intención principal compuesta. "
            "Cada intención principal debe describir claramente la acción deseada tal como fue escrita, "
            "incluyendo dispositivos, eventos o expresiones temporales, sin interpretar, resumir ni modificar "
            "la estructura temporal original usada por el usuario. "
            "NO convertir fechas ni horas, NO normalizar texto, NO inventar detalles y NO usar formatos tipo JSON."
        )
    )

    intenciones_secundarias: list = dspy.OutputField(
        desc=(
            "Lista en lenguaje natural de intenciones adicionales expresadas explícitamente por el usuario. "
            "Una intención secundaria debe: "
            "(1) ser una acción literal formulada por el usuario, "
            "(2) NO ser necesaria para completar la intención principal, "
            "(3) NO ser parte del objetivo principal del mensaje. "
            "NO incluir narraciones, comentarios personales, emociones, humor, exageraciones o descripciones "
            "que no expresen una acción concreta. "
            "NO se permiten inferencias, suposiciones ni interpretaciones. "
            "Si el usuario no escribió literalmente otra ACCIÓN independiente, la lista debe estar vacía."
        )
    )






    razonamiento: str = dspy.OutputField(
    desc=(
    "Explicación clara del proceso de interpretación. Debe justificar por qué cada acción "
    "fue clasificada como principal o secundaria, señalando si el usuario expresó o no un verbo rector. "
    "Debe identificar qué información temporal o contextual falta, sin inventar nada. "
    "Debe diferenciar entre información presente en feedback (que NO debe marcarse como faltante) "
    "y datos realmente ausentes en el mensaje del usuario."
)

    )


    confianza: float = dspy.OutputField(
        desc="Nivel de certeza (0 a 1) sobre la interpretación final."
    )



# Configurar LLM base
llama_31 = dspy.LM('ollama_chat/llama3.1:latest', api_base='http://localhost:11434', api_key='')
dspy.configure(lm=llama_31)


# Crear el predictor semántico
receptor = dspy.Predict(Receptor)


formato_feedback = {
    "historial": [
        {
            "id_interaccion": int,  # número incremental o timestamp
            "intencion_prev": str,  # resumen semántico de la intención principal detectada
            "entidades": list,  # lista de entidades involucradas, ej. ["dispositivo_A", "dispositivo_B"]
            "resultado": str,  # {"exitoso", "fallido", "parcial", "invalido"}
            "observacion": str,  # descripción general del evento, sin ejemplos específicos
            "acciones_recomendadas": list,  # sugerencias automáticas o de otros SLMs
        }
    ],

    "restricciones": [
        # condiciones acumuladas a nivel de sistema o usuario
        {"tipo": "tecnica", "descripcion": "Evitar servidores en mantenimiento"},
        {"tipo": "logica", "descripcion": "No repetir consultas idénticas consecutivas"},
        {"tipo": "usuario", "descripcion": "No usar predicción con menos de 3 datos históricos"}
    ],

    "preferencias_usuario": {
        "formato_respuesta": "grafica" or "texto" or "mixto",
        "nivel_detalle": "bajo" or "medio" or "alto",
        "idioma": "es" or "en",
        "unidades": "kWh",
        "modo_interaccion": "texto" or "voz"
    },

    "dispositivos": [
        {"nombre": "nombre del dispositivo", "ubicacion": "Ubicación del dispositivo", "tipo": "tipo de dispositivo"},
    
    ]
    
}

""" feedback={"historial":None,
          "restricciones":None,
          "preferencias_usuario":"el usuario quiere que lo llamen con el nombre de cuchurrumin",
          "dispositivos de los que se tiene informacion": [
        {
            "nombre": "Aire acondicionado",
            "ubicacion": "Sala principal",
            "tipo": "Electrodoméstico de climatización"
        },
        {
            "nombre": "Televisor",
            "ubicacion": "Sala principal",
            "tipo": "Dispositivo de entretenimiento"
        },
        {
            "nombre": "Ventilador",
            "ubicacion": "Habitación principal",
            "tipo": "Dispositivo de ventilación"
        },
        {
            "nombre": "Lámpara",
            "ubicacion": "Habitación principal",
            "tipo": "Dispositivo de iluminación"
        },
        {
            "nombre": "PC",
            "ubicacion": "Estudio",
            "tipo": "Equipo electrónico"
        }
    ]
        } """



pregunta=("Necesito saber cuánto consumió mi nevera ayer por la noche, "
    "y también cuánto consumió mi lavadora el sábado pasado en la mañana. "
    "Además quiero que me digas si entre esos dos días cuál gastó más energía. "
    "Ah, y por cierto, mientras miraba esos consumos se me descargó el celular "
    "y me dio mucha pereza pararme a buscar el cargador, pero igual quiero la comparación."
    "Papá también pidió que le dijeras cuánto fue el consumo de todos los dispositivos en el año 2024, quiero ver gráficas de todo lo que se pueda")

#Llamada al LLM
resultado = receptor(
    pregunta = pregunta,

    feedback=None,
    system_summary=system_summary,
)


intenciones_principales=resultado.intenciones_principales
intenciones_secundarias=resultado.intenciones_secundarias
razonamiento=resultado.razonamiento
confianza=resultado.confianza



print("==================Mensaje enviado por el usuario==================")
print(pregunta)
print("=====================")


print("=== COOKED RESULTADO ===")
print('las intenciones principales detectadas fueron:\n',intenciones_principales)
print('las intenciones_secundarias detectadas fueron:\n',intenciones_secundarias)
print('el razonamiento realizado fue:\n',razonamiento)
print('la confianza es:\n',confianza)
