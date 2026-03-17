import dspy
import time

print("Iniciando dspy.LM test...")
start = time.time()

try:
    # Simular modelo falso y URL que no responde
    llm = dspy.LM('ollama_chat/modelo-falso-que-no-existe', api_base='http://localhost:9999', api_key='')
    print(f"Instancia creada en {time.time() - start:.2f} segundos.")
    
    # Intentar configurar dspy
    dspy.settings.configure(lm=llm)
    print("Configuración exitosa.")

except Exception as e:
    print(f"Error: {e}")

print(f"Fin del test. Tiempo total: {time.time() - start:.2f} segundos.")
