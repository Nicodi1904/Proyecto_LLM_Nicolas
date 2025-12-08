import asyncio
from fastmcp import Client 

# ---------------------------------------------------------
# 1. EXTRAER METADATOS DEL SERVIDOR (tools + resources)
# ---------------------------------------------------------

async def extraer_metadatos_server():
    client = Client("http://127.0.0.1:8000/sse")

    async with client:
        conexion = await client.ping()
        print("=========================================")
        print("Conectado al servidor MCP, se procede a extraer metadatos:", conexion)
        print("=========================================")

        tools = await client.list_tools()
        resources = await client.list_resources()

        print("Datos extraídos.")
        print("=========================================")

    return tools, resources
tools, resources = asyncio.run(extraer_metadatos_server())


# ---------------------------------------------------------
# 2. PROCESAR TOOLS — SOLO LO QUE REALMENTE DEVUELVE EL SERVIDOR
# ---------------------------------------------------------

def separar_metadatos_tools_cliente(tools):
    global tool_names, tool_descriptions, tool_input_schemas, tool_output_schemas

    tool_names = []
    tool_descriptions = []
    tool_input_schemas = []
    tool_output_schemas = []

    for tool in tools:
        data = tool.model_dump()

        tool_names.append(data.get("name"))
        tool_descriptions.append(data.get("description"))

        # El servidor nuevo sí expone input/output Schema directamente
        tool_input_schemas.append(data.get("input_schema") or {})
        tool_output_schemas.append(data.get("output_schema") or {})


separar_metadatos_tools_cliente(tools)


# ---------------------------------------------------------
# 3. PROCESAR RESOURCES — OPCIONAL PERO ROBUSTO
# ---------------------------------------------------------

def separar_metadatos_resources_cliente(resources):
    global resource_names, resource_descriptions, resource_mime_types, resource_uris, resource_values

    resource_names = []
    resource_descriptions = []
    resource_mime_types = []
    resource_uris = []
    resource_values = []

    for resource in resources:
        data = resource.model_dump()

        resource_names.append(data.get("name"))
        resource_descriptions.append(data.get("description"))
        resource_mime_types.append(data.get("mime_type"))
        resource_uris.append(data.get("uri"))

        resource_values.append(
            data.get("value") or
            data.get("content") or
            data.get("data") or None
        )


separar_metadatos_resources_cliente(resources)


# ---------------------------------------------------------
# 4. FORMATO FINAL QUE CONSUME EL RECEPTOR
# ---------------------------------------------------------

formato_system_summary = { 
    "servers": [
        {
            "server_id": str,
            "tools": [
                {
                    "name": str,
                    "description": str,
                    "inputs": dict,
                    "outputs": dict
                }
            ]
        }
    ]
}


# ---------------------------------------------------------
# 5. CONSTRUCT SUMMARY — ADAPTADO AL NUEVO SERVIDOR
# ---------------------------------------------------------

def crear_summary(tool_names, tool_descriptions, tool_input_schemas, tool_output_schemas):
    """
    Crea un resumen del sistema usando únicamente los campos que el servidor MCP garantiza:
    - name
    - description
    - input_schema
    - output_schema
    """

    summary = {"servers": []}

    # El servidor actual NO expone server_info → ponemos identificador genérico
    server_id = "mcp_local_server"

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


# ---------------------------------------------------------
# 6. CREAR SUMMARY FINAL
# ---------------------------------------------------------

system_summary = crear_summary(
    tool_names, 
    tool_descriptions, 
    tool_input_schemas, 
    tool_output_schemas
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


