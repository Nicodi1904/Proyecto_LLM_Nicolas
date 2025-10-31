import asyncio
from fastmcp import Client 

#Se abre sesión del cliente
async def main():
    # Dirección en la que se abrió el servidor y tipo de comunicación
    client = Client("http://127.0.0.1:8000/sse")
    async with client:
        # Verificar conexión
        await client.ping()
        print("Conectado al servidor MCP")

        ##########################################################################################
        #Inicio de la sesión del cliente (Se podría manejar acá toda una lógica de programación, pero pienso que es mejor solo pedir lo esencial al server y cerrar la sesión, luego procesar los datos recibidos sin mantener la sesión)
        ############################################################################################

        # Listar herramientas
        tools = await client.list_tools()
        print("Herramientas disponibles:")
        #print(tools)
        for tool in tools:
            print(f" - {tool.name}")

        # Ejecutar una herramienta de ejemplo si existe
        if tools:
            tool_name = tools[0].name
            print(f"\nEjecutando herramienta: {tool_name}")
            result = await client.call_tool(tool_name, {"a": 3, "b": 5})
            print("Resultado:", result)

asyncio.run(main())
