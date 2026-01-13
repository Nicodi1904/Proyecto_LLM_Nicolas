import asyncio
from fastmcp import Client

async def extraer_metadatos_server():
    client = Client("http://127.0.0.1:8000/sse")
    print("conectando con el servidor")
    async with client:
        await client.ping()
        tools = await client.list_tools()
    print("metadatos obtenidos")
    return tools


def crear_system_summary(tools):
    summary = {
        "servers": [
            {
                "server_id": "mcp_local_server",
                "tools": []
            }
        ]
    }

    for tool in tools:
        t = tool.model_dump()

        entry = {
            "name": t["name"],
            "meta": t.get("meta", {})
        }

        summary["servers"][0]["tools"].append(entry)

    return summary


# Ejecutar:
tools = asyncio.run(extraer_metadatos_server())
system_summary = crear_system_summary(tools)

