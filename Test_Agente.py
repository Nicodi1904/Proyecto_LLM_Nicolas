import dspy

#Se definen las funciones propias de python que el agente usará
def sumar(a: int, b: int) -> int:
    print(f"[TOOL_USE] Ejecutando sumar({a}, {b})")
    return a + b

def restar(a: int, b: int) -> int:
    print(f"[TOOL_USE]] Ejecutando restar({a}, {b})")
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

# se define el plan que hará el LLM
class OperacionSignature(dspy.Signature):
    """
    El modelo debe decidir si la pregunta requiere una suma o resta,
    e identificar los números involucrados.
    Si no es una operación matemática, simplemente debe responder.
    """
    pregunta: str = dspy.InputField()
    operacion: str = dspy.OutputField(desc="Puede ser 'suma', 'resta' o 'ninguna'")
    a: int = dspy.OutputField(desc="Primer número si aplica, o 0")
    b: int = dspy.OutputField(desc="Segundo número si aplica, o 0")
    respuesta: str = dspy.OutputField(desc="Respuesta final para el usuario")


#  Módulo principal 
class Agente(dspy.Module):
    def __init__(self):
        super().__init__()
        self.sumar = SumarModule()
        self.restar = RestarModule()
        self.planificador = dspy.ChainOfThought(OperacionSignature)

    def __call__(self, pregunta: str):
        # El LLM genera un plan
        plan = self.planificador(pregunta=pregunta)
        print(plan)

        if plan.operacion == "suma":
            resultado = self.sumar(plan.a, plan.b)
            return f"{plan.a} + {plan.b} = {resultado}"
        
        elif plan.operacion == "resta":
            resultado = self.restar(plan.a, plan.b)
            return f"{plan.a} - {plan.b} = {resultado}"

        else:
            return plan.respuesta


#  Ejemplo de uso 
agente = Agente()

print(agente("¿Cuál es la capital de Francia?"))
print(agente("¿Qué es 10 menos 3 y 4-2?"))
