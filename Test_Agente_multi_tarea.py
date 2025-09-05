import dspy

#Se definen las funciones propias de python que el agente usará
def sumar(a: int, b: int) -> int:
    print(f"[TOOL_USE] Ejecutando sumar({a}, {b})") #Mensaje para saber si la tool fue usada
    return a + b

def restar(a: int, b: int) -> int:
    print(f"[TOOL_USE] Ejecutando restar({a}, {b})") #Mensaje para saber si la tool fue usada
    return a - b

# Para cada función se crea un módulo de DSPy que actúa como wrappers (Para que DSPY las detecte) 
class SumarModule(dspy.Module):
    def __call__(self, a: int, b: int) -> int:
        return sumar(a, b)

class RestarModule(dspy.Module):
    def __call__(self, a: int, b: int) -> int:
        return restar(a, b)


#Cargar LLM
lm = dspy.LM('ollama_chat/llama3.1', api_base='http://localhost:11434', api_key='')
dspy.configure(lm=lm)

#Se define la signature que el LLM deberá armar para resolver la solicitud
class Interprete(dspy.Signature):
    """
    El modelo debe identificar todas las operaciones matemáticas
    (sumas o restas) dentro de la pregunta.
    """
    pregunta: str = dspy.InputField()
    operaciones: list[dict] = dspy.OutputField(
        desc="Lista de diccionarios con las indicaciones a realizar, cada diccionario traerá: 'tipo', 'reasoning', variables necesarias. Ejemplo de respuesta:  [{'tipo': 'resta', 'reasoning': el usuario solicitó una resta ,'a': 10, 'b': 3}, ...] "
    )
    respuesta: str = dspy.OutputField(desc="Respuesta final para el usuario")


#  Módulo principal 
class Agente(dspy.Module):
    #Se declaran variables y funciones propias que el agente puede tener
    def __init__(self):
        super().__init__()
        self.sumar = SumarModule()
        self.restar = RestarModule()
        self.planificador = dspy.ChainOfThought(Interprete)

    #Se declara el "Accionador" que llamará las tools creadas
    def __call__(self, pregunta: str):
        # El LLM genera un plan usando la signature
        plan = self.planificador(pregunta=pregunta)
        print(plan)

        # Si el modelo detecta operaciones
        if plan.operaciones and isinstance(plan.operaciones, list): #isinstance sirve para corroborar que una variable sea de cierto tipo
            resultados = []
            
            for operacion in plan.operaciones:

                if operacion["tipo"] == "suma":
                    resultado = self.sumar(operacion["a"], operacion["b"])
                    resultados.append(f"{operacion['a']} + {operacion['b']} = {resultado}")

                elif operacion["tipo"] == "resta":
                    resultado = self.restar(operacion["a"], operacion["b"])
                    resultados.append(f"{operacion['a']} - {operacion['b']} = {resultado}")

            # Devolver resultados combinados
            return " | ".join(resultados)

        # Si no hay operaciones, devolver la respuesta normal
        return plan.respuesta


#  Ejemplo de uso 
agente = Agente()

print(agente("¿Cuál es la capital de Francia?"),'\n\n-----------------------')
print(agente("Sabes? siempre me he preguntado cuál podría ser la respuesta a múltiples expresiones matemáticas, quisiera saber cuánto es 40008-78, podrías ayudarme? y también 2-1"),'\n\n-----------------------')
print(agente("Sabes? siempre me he preguntado cuál podría ser la respuesta a múltiples expresiones matemáticas, quisiera saber cuánto es 40008-78, podrías ayudarme? y también 2+1"),'\n\n-----------------------')
print(agente("Sabes? siempre me he preguntado cuál podría ser la respuesta a múltiples expresiones matemáticas, quisiera saber cuánto es 40008+78, podrías ayudarme? y también 2+1"),'\n\n-----------------------')
print(agente("Sabes? siempre me he preguntado cuál podría ser la respuesta a múltiples expresiones matemáticas, quisiera saber cuánto es 40008+78, podrías ayudarme? y también 2-1"),'\n\n-----------------------')
