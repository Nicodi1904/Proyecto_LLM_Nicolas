import asyncio
from fastmcp import Client 

# Se abre sesión del cliente
async def extraer_metadatos_server():
    # Dirección en la que se abrió el servidor y tipo de comunicación
    client = Client("http://127.0.0.1:8000/sse")

    async with client:
        # Verificar conexión
        conexion = await client.ping()
        print("=========================================")
        print("Conectado al servidor MCP, se procede a extraer metadatos", ": ", conexion)
        print("=========================================")
        
        tools = await client.list_tools()
        resources = await client.list_resources()

        print("Datos extraídos.")
        print("=========================================")

    return tools, resources


tools, resources = asyncio.run(extraer_metadatos_server())


def separar_metadatos_tools_cliente(tools):
    global tool_names, tool_descriptions, tool_input_schemas, tool_output_schemas, tool_examples, tool_metadata, tool_server_info
    tool_names = []
    tool_descriptions = []
    tool_input_schemas = []
    tool_output_schemas = []
    tool_examples = []
    tool_metadata = []
    tool_server_info = []

    for tool in tools:
        data = tool.model_dump()  # convertir a dict normal ya que MCP utiliza objetos pydantic

        # Extracción principal
        tool_names.append(data.get("name"))
        tool_descriptions.append(data.get("description"))

        # Sacar el input y output schem desde metadatos directamente (Debido a error del servidor al no inferir correctamente estos squemas)
        meta_data = data.get("meta", {})
        if meta_data:
            tool_input_schemas.append(meta_data.get("input_schema"))
            tool_output_schemas.append(meta_data.get("output_schema"))
        else:
            tool_input_schemas.append(None)
            tool_output_schemas.append(None)

        tool_examples.append(data.get("examples"))
        tool_metadata.append(data.get("metadata"))
        tool_server_info.append(data.get("server_info"))


separar_metadatos_tools_cliente(tools)

# Pruebas
print("------------------------PRUEBAS-------------------")
print(f"Nombres de las tools disponibles\n{tool_names[7]}\n")
print("--------------------------------------------------")
print(f"Descripción de la tool\n{tool_descriptions[7]}\n")
print("--------------------------------------------------")
print(f"Entradas esperadas de las tools disponibles\n{tool_input_schemas[7]}\n\n")
print("--------------------------------------------------")
print(f"Salidas esperadas de las tools disponibles\n{tool_output_schemas[7]}")
