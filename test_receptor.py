import dspy
from test_dump import system_summary


class Receptor(dspy.Signature):
    """
    Evalúa la intención principal y las posibles intenciones secundarias del usuario,
    considerando el contexto conversacional, el estado del sistema y las capacidades disponibles.
    """

    # ------------------------
    # ENTRADAS
    # ------------------------

    pregunta: str = dspy.InputField(
        desc="Mensaje original del usuario en lenguaje natural."
    )

    feedback: dict = dspy.InputField(
        default={},
        desc=(
            "Información contextual acumulada de interacciones previas o de otros SLMs. "
            "Incluye observaciones, restricciones o aclaraciones históricas relevantes."
        )
    )

    system_summary: dict = dspy.InputField(
        default={},
        desc=(
            "Resumen global del sistema: herramientas y funciones disponibles, dominios cubiertos, "
            "limitaciones generales y descripciones breves de cada capacidad."
        )
    )


    # ------------------------
    # SALIDAS
    
    intenciones_principales: list = dspy.OutputField(
    desc=(
        "Lista de intenciones principales detectadas (cada elemento es una cadena). "
        "Cada intención debe formularse como la pregunta exacta que el sistema debe responder; "
        "es decir, debe dejar claro: (1) la acción solicitada o métrica requerida (qué), "
        "(2) las entidades o dispositivos involucrados (qué objetos), "
        "(3) el alcance temporal o condición espacial si aplica (cuándo / rango), "
        "(4) la forma de agregación o comparación requerida (cómo medir o comparar), "
        "y (5) el formato de salida preferido si fue explícito (texto, tabla, gráfico). "
    )
)

    intenciones_secundarias: list = dspy.OutputField(
    desc=(
        "Lista de intenciones complementarias o implícitas derivadas del mensaje del usuario. "
        "Estas intenciones no son el objetivo principal, pero aportan contexto, precisión o valor agregado "
        "a la solicitud original. Cada elemento debe representar una posible subpregunta o acción auxiliar "
        "que el sistema podría ejecutar para mejorar la respuesta final, validar datos o ampliar el análisis. "
        "Incluye, por ejemplo, peticiones comparativas, validaciones, condiciones no explícitas o posibles "
        "interpretaciones alternativas del enunciado del usuario. "
    )
)
    planeacion: str = dspy.OutputField(
    desc=(
        "Descripción en lenguaje natural del plan de acción propuesto para resolver la solicitud del usuario. "
        "Debe detallar los pasos o estrategias a seguir, indicando qué información se requiere, qué operaciones "
        "se ejecutarán (por ejemplo, cálculos, comparaciones o consultas), y en qué orden lógico se desarrollarán. "
        "Debe ser comprensible, coherente y reflejar razonamiento causal —no solo una lista de tareas."
    )
)


    razonamiento: str = dspy.OutputField(
        desc=(
            "Análisis estructurado sobre la viabilidad de cumplir las intenciones detectadas. "
            "Debe incluir: (1) si es posible cumplirlas con las herramientas disponibles, "
            "(2) qué información falta o limita la ejecución, y "
            "(3) sugerencias para resolver esas limitaciones."
            )

    )

    confianza: float = dspy.OutputField(
        desc=(
            "Nivel de certeza (0 a 1) en la interpretación general de la intención del usuario "
            "y la viabilidad de procesarla con el sistema actual."
        )
    )


# Configurar LLM base
llama_31 = dspy.LM('ollama_chat/llama3.1', api_base='http://localhost:11434', api_key='')
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

feedback={"historial":None,
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
        }
#Llamada al LLM
resultado = receptor(
    pregunta="cuánto fue el consumo de mi aire acondicionado el 5 de marzo y el del PC el 8 de octubre, cuál de ellos consume menos? YABA DABA DUUUUUUUUUUUUUUUUUU" \
    "Ayer me comí un helado pero se me cayó, entonces quiero saber cuál entre mi televisor o mi computador consumio más de las 8 a las 3, es de vital importancia saber esto, no hay nada más importante en el planeta.",
    feedback=feedback,
    system_summary=system_summary,
)



print("=== RAW RESULTADO ===")
print(resultado)


intenciones_principales=resultado.intenciones_principales
intenciones_secundarias=resultado.intenciones_secundarias
planeacion=resultado.planeacion
razonamiento=resultado.razonamiento
confianza=resultado.confianza




print("=== COOKED RESULTADO ===")
print('las intenciones principales detectadas fueron:\n',intenciones_principales)
print('las intenciones_secundarias detectadas fueron:\n',intenciones_secundarias)
print('las intenciones_secundarias detectadas fueron:\n',planeacion)
print('el razonamiento realizado fue:\n',razonamiento)
print('la confianza es:\n',confianza)
