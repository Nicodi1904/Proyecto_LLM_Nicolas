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

# Obtener metadatos al importar
tools = asyncio.run(extraer_metadatos_server())
system_summary = crear_system_summary(tools)

