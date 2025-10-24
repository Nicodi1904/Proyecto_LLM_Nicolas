# Cliente_MCP_test2.py

import asyncio
from fastmcp import Client  # ✅ Import correcto según la documentación oficial

async def main():
    # Conexión al servidor MCP local
    client = Client("http://127.0.0.1:8000/sse")
    async with client:
        # Verificar conexión
        await client.ping()
        print("✅ Conectado al servidor MCP")

        # Listar herramientas
        tools = await client.list_tools()
        print("🧰 Herramientas disponibles:")
        for tool in tools:
            print(f" - {tool.name}")

        # Ejecutar una herramienta de ejemplo si existe
        if tools:
            tool_name = tools[0].name
            print(f"\n▶️ Ejecutando herramienta: {tool_name}")
            result = await client.call_tool(tool_name, {"a": 3, "b": 5})
            print("📤 Resultado:", result)

asyncio.run(main())
