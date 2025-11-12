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
def separar_metadatos_resources_cliente(resources):
    global resource_names, resource_descriptions, resource_mime_types, resource_uris, resource_values

    resource_names = []
    resource_descriptions = []
    resource_mime_types = []
    resource_uris = []
    resource_values = []

    for resource in resources:
        data = resource.model_dump()  # convertir a dict normal, igual que con tools

        # Extracción principal
        resource_names.append(data.get("name"))
        resource_descriptions.append(data.get("description"))
        resource_mime_types.append(data.get("mime_type"))
        resource_uris.append(data.get("uri"))

        # Algunos servidores pueden devolver el valor real del recurso bajo distintas claves
        # dependiendo de la implementación (ej. "value", "data" o "content")
        resource_values.append(
            data.get("value")
            or data.get("data")
            or data.get("content")
            or None
        )
separar_metadatos_resources_cliente(resources)

formato_system_summary = { #Lo que se le pasa al receptor
    "servers": [
        {
            "server_id": str,   # identificador del servidor MCP (ej. "server_analisis")
            "tools": [
                {
                    "name": str,         # nombre interno de la herramienta
                    "description": str,  # propósito o función general
                    "inputs": dict,      # esquema JSON de entrada reportado por el servidor
                    "outputs": dict      # esquema JSON de salida reportado por el servidor
                }
            ]
        }
    ]
}




def crear_summary(tool_names, tool_descriptions, tool_input_schemas, tool_output_schemas, tool_server_info):
    """
    Crea un resumen del sistema (system_summary) utilizando únicamente la información
    obtenida directamente del servidor MCP.
    """
    summary = {"servers": []}

    # Identificador del servidor (o valor por defecto)
    server_id = "server_desconocido"
    if tool_server_info and tool_server_info[0]:
        server_id = tool_server_info[0].get("id", "server_desconocido")

    server_data = {
        "server_id": server_id,
        "tools": []
    }

    for name, desc, input_schema, output_schema in zip(
        tool_names, tool_descriptions, tool_input_schemas, tool_output_schemas
    ):
        tool_data = {
            "name": name or "unknown_tool",
            "description": desc or "Sin descripción.",
            "inputs": input_schema or {},
            "outputs": output_schema or {}
        }
        server_data["tools"].append(tool_data)

    summary["servers"].append(server_data)
    return summary


# Crear el resumen del sistema
system_summary = crear_summary(
    tool_names, tool_descriptions, tool_input_schemas, tool_output_schemas, tool_server_info
)
































""" # Pruebas
print("------------------------PRUEBAS-------------------")
print(f"Nombres de las tools disponibles\n{tool_names[7]}\n")
print("--------------------------------------------------")
print(f"Descripción de la tool\n{tool_descriptions[7]}\n")
print("--------------------------------------------------")
print(f"Entradas esperadas de las tools disponibles\n{tool_input_schemas[7]}\n\n")
print("--------------------------------------------------")
print(f"Salidas esperadas de las tools disponibles\n{tool_output_schemas[7]}") """

#Con esto ya adquirimos los datos semánticos de las capacidades del servidor
#ahora hay que darle acceso a los datos al LLM, para ello hay que pensar cómo presentarselos


