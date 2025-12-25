import asyncio
from fastmcp import Client

# Server address - assuming default or same as reference
SERVER_URL = "http://127.0.0.1:8000/sse"

async def extraer_metadatos_server():
    client = Client(SERVER_URL)
    print(f"Conectando con el servidor en {SERVER_URL}...")
    try:
        async with client:
            await client.ping()
            tools = await client.list_tools()
        print("Metadatos obtenidos exitosamente.")
        return tools
    except Exception as e:
        print(f"Error al conectar con el servidor: {e}")
        return []

def crear_system_summary(tools):
    summary = {
        "servers": [
            {
                "server_id": "mcp_server_gravity",
                "tools": []
            }
        ]
    }

    for tool in tools:
        t = tool.model_dump()
        meta = t.get("meta", {}).copy()
        meta.pop("_fastmcp", None)
        
        entry = {
            "name": t["name"],
            "meta": meta
        }
        summary["servers"][0]["tools"].append(entry)

    return summary


try:
    # Verificar si ya existe un loop corriendo (caso Jupyter/IPython)
    asyncio.get_running_loop()
    
    # Si existe, no podemos usar asyncio.run() directamente.
    # Usamos un hilo separado para ejecutar la tarea asíncrona de forma síncrona para el import.
    import threading
    def run_async_in_thread(coro):
        result = [None]
        error = [None]
        def target():
            try:
                result[0] = asyncio.run(coro)
            except Exception as e:
                error[0] = e
        t = threading.Thread(target=target)
        t.start()
        t.join()
        if error[0]:
            raise error[0]
        return result[0]
        
    tools = run_async_in_thread(extraer_metadatos_server())

except RuntimeError:
    # No hay loop corriendo, caso script normal
    tools = asyncio.run(extraer_metadatos_server())


system_summary = crear_system_summary(tools)
print(system_summary)