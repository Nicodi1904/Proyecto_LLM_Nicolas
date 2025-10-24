import asyncio
from fastmcp import Client

async def main():
    # Conectarse a un servidor MCP HTTP/SSE
    client = Client("http://127.0.0.1:8000/mcp")

    async with client:
        # Verificar conectividad
        await client.ping()
        print("✅ Servidor MCP alcanzable")

        # Listar herramientas disponibles
        tools = await client.list_tools()
        print("🧰 Herramientas disponibles:", tools)


asyncio.run(main())