import asyncio
from fastmcp import Client 

#Se abre sesión del cliente
async def main():
    # Dirección en la que se abrió el servidor y tipo de comunicación
    client = Client("http://127.0.0.1:8000/sse")

    async with client:
        # Verificar conexión
        conexion=await client.ping()
        print("=========================================")
        print("Conectado al servidor MCP",": ",conexion)
        print("=========================================")
        
        
        # Listar herramientas expuesta@s en el servidor
        tools = await client.list_tools()
        print("=========================================")
        print("Herramientas expuesta@s:")
        print("=========================================")
        for tool in tools: 
            print(f" - {tool}") #Cada herramienta tiene una serie de metadatos compartidos en Json
        # Listar recursos expuesta@s en el servidor
        resources = await client.list_resources()
        print("=========================================")
        print("Recursos expuesta@s:")
        print("=========================================")
        for resource in resources:
            print(f" - {resource}")

       
        
         # Pruebas
        resources = await client.list_resources()
        print("=========================================")
        print("Pruebas")
        print("=========================================")
        # Ejecutar una herramienta de ejemplo si existe
        if tools:
            tool_name = tools[0].name
            print(f"\nEjecutando herramienta: {tool_name}")
            result = await client.call_tool(tool_name, {"a": 3, "b": 5})
            print("Resultado:", result)

asyncio.run(main())

#Estructura de la info de la tool traida desde el cliente.

#[name, title, description, imputschema={'properties':,'required':, 'type':}, outputSchema={'properties': {'result': {'type':}, required:, 'type':, 'x-fastmcp-wrap-result':}, icons=, annotations=, meta={'_fastmcp': {'tags': []}}]