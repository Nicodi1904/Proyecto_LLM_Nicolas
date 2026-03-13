import asyncio
from fastmcp import Client
import json

class MCP_Client:
    def __init__(self):
        """
        Inicializa el cliente MCP.
        """
        pass

    async def ejecutar_plan(self, server_url: str, plan_filtrado: list) -> dict:
        """
        Ejecuta un plan de acciones VALIDADO contra un servidor MCP.
        Organiza el reporte final agrupado por ID de solicitud (ej: @1).
        """
        client = Client(server_url)
        resultados_raw = {} # Mapa ID Acción -> Resultado
        reporte_final = {}  # Mapa ID Solicitud -> Lista de Resultados
        
        # Pre-inicializar grupos en reporte_final
        for accion in plan_filtrado:
            id_accion = accion.get("id", "")
            # Extraer ID solicitud (todo antes del primer punto, ej @1.1 -> @1)
            if "." in id_accion:
                id_solicitud = id_accion.split(".")[0]
            else:
                id_solicitud = id_accion # Fallback si no hay punto
                
            if id_solicitud not in reporte_final:
                reporte_final[id_solicitud] = []

        print(f"Conectando a {server_url} para ejecutar plan validado...")
        
        try:
            async with client:
                await client.ping()
                
                for accion in plan_filtrado:
                    id_accion = accion.get("id")
                    tool_name = accion.get("tool")
                    descripcion = accion.get("descripcion", "")
                    inputs_originales = accion.get("inputs", {})
                    
                    # Identificar Grupo
                    id_solicitud = id_accion.split(".")[0] if "." in id_accion else id_accion
                    
                    print(f"Ejecutando {id_accion}: {tool_name}")
                    
                    # Resolver dependencias (referencias @N.M en inputs)
                    inputs_finales = {}
                    for k, v in inputs_originales.items():
                        # Si es string y parece una referencia
                        if isinstance(v, str) and v.startswith("@") and "." in v:
                            # Buscar en resultados previos
                            if v in resultados_raw:
                                # Usamos el resultado de la acción previa
                                inputs_finales[k] = resultados_raw[v].get("resultado")
                            else:
                                # Referencia no resuelta
                                inputs_finales[k] = v 
                        # Si es diccionario (caso anidado simple)
                        elif isinstance(v, dict):
                             inputs_finales[k] = v.copy()
                             for sub_k, sub_v in v.items():
                                 if isinstance(sub_v, str) and sub_v.startswith("@") and "." in sub_v:
                                     if sub_v in resultados_raw:
                                         inputs_finales[k][sub_k] = resultados_raw[sub_v].get("resultado")
                        else:
                            inputs_finales[k] = v
                    
                    # Estructura del resultado de esta acción
                    resultado_item = {
                        "accion_id": id_accion,
                        "tool": tool_name,
                        "descripcion": descripcion,
                        "resultado": None,
                        "error": None
                    }
                    
                    try:
                        res_tool = await client.call_tool(tool_name, inputs_finales)
                        
                        # Extraer el contenido real del objeto CallToolResult
                        dato_real = None
                        if hasattr(res_tool, 'content') and res_tool.content:
                            for c in res_tool.content:
                                if hasattr(c, 'type') and c.type == 'text' and hasattr(c, 'text'):
                                    try:
                                        dato_real = json.loads(c.text)
                                    except:
                                        dato_real = c.text
                                    break
                        
                        if dato_real is None:
                             dato_real = str(res_tool)

                        resultado_item["resultado"] = dato_real
                        print(f"  -> OK")
                    except Exception as e:
                        msg_error = f"Error ejecución tool: {str(e)}"
                        print(f"  -> {msg_error}")
                        resultado_item["error"] = msg_error
                    
                    # Guardar para dependencias futuras
                    resultados_raw[id_accion] = resultado_item
                    
                    # Agregar al reporte agrupado
                    reporte_final[id_solicitud].append(resultado_item)

        except Exception as e:
            print(f"Error general de conexión o ejecución MCP: {e}")
            
        return reporte_final

    def consolidar_reportes(self):
        """
        De momento vacía por requerimiento.
        """
        pass
