import asyncio
from fastmcp import Client
import json

class MCP_Client:
    def __init__(self):
        """
        Inicializa el cliente MCP.
        """
        pass

    def aux_timestamp_iso(self, plan: list) -> list:
        """
        Recorre el plan y reemplaza la 'T' por un espacio en cualquier 
        string que parezca un timestamp ISO (YYYY-MM-DDTHH:MM:SS).
        """
        import re
        # Patrón para detectar YYYY-MM-DDTHH:MM o YYYY-MM-DDTHH:MM:SS
        pattern = re.compile(r'^(\d{4}-\d{2}-\d{2})T(\d{2}:\d{2})(?::(\d{2}))?$')
        
        def process_item(item):
            if isinstance(item, str):
                match = pattern.match(item)
                if match:
                    # Para SQLite evitar exclusiones por len de string, si no trae seg añadir :00
                    segundos = f":{match.group(3)}" if match.group(3) else ":00"
                    return f"{match.group(1)} {match.group(2)}{segundos}"
                return item
            elif isinstance(item, dict):
                return {k: process_item(v) for k, v in item.items()}
            elif isinstance(item, list):
                return [process_item(i) for i in item]
            return item

        return process_item(plan)

    async def ejecutar_plan(self, server_url: str, plan_original: list) -> dict:
        """
        Ejecuta un plan de acciones VALIDADO contra un servidor MCP.
        Organiza el reporte final agrupado por ID de solicitud (ej: @1).
        """
        # Normalizar timestamps antes de procesar el plan
        plan_filtrado = self.aux_timestamp_iso(plan_original)
        
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

        print(f"\n🔌 [Cliente MCP] CONECTANDO A {server_url}")
        print(f"📦 [Cliente MCP] PLAN RECIBIDO A EJECUTAR:\n{plan_filtrado}")
        
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
                    print(f"\n▶️ [Cliente MCP] EJECUTANDO: {id_accion} ({tool_name})")
                    print(f"    └ Inputs Finales: {inputs_finales}")
                    resultado_item = {
                        "accion_id": id_accion,
                        "tool": tool_name,
                        "descripcion": descripcion,
                        "resultado": None,
                        "error": None
                    }
                    
                    try:
                        res_tool = await client.call_tool(tool_name, inputs_finales)
                        print(f"📡 [Servidor MCP] RESPUESTA CRUDA ({id_accion}): {res_tool}")
                        
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
            
        print(f"\n📊 [Cliente MCP] REPORTE FINAL GENERADO")
        print(reporte_final)
        print("="*50 + "\n")
        return reporte_final

    def consolidar_reportes(self):
        """
        De momento vacía por requerimiento.
        """
        pass

    def arreglar_reporte(self, reporte_ejecucion: dict) -> tuple[dict, dict]:
        """
        Toma el reporte de ejecución y genera dos versiones:
        1. reporte_worker3: Contiene SOLO las acciones de 'obtener_consumo' con todos los datos intactos.
        2. reporte_llm: Contiene TODO el plan, pero si los datos de 'obtener_consumo' 
           tienen más de 4 elementos, los trunca (2 iniciales, "...", 2 finales).
        """
        import copy
        
        reporte_worker3 = {}
        reporte_llm = copy.deepcopy(reporte_ejecucion)

        # 1. Construir reporte_worker3
        for req_id, acciones in reporte_ejecucion.items():
            acciones_consumo = [a for a in acciones if a.get("tool") == "obtener_consumo"]
            if acciones_consumo:
                reporte_worker3[req_id] = copy.deepcopy(acciones_consumo)

        # 2. Construir reporte_llm (truncar datos)
        for req_id, acciones in reporte_llm.items():
            for accion in acciones:
                if accion.get("tool") == "obtener_consumo":
                    resultado = accion.get("resultado", {})
                    if isinstance(resultado, dict) and resultado.get("status") == "success":
                        datos = resultado.get("datos", {})
                        
                        # Iterar sobre cada dispositivo en los datos
                        for dispositivo, valores in datos.items():
                            if isinstance(valores, dict) and len(valores) > 4:
                                list_items = list(valores.items())
                                # Tomar 2 primeros y 2 últimos
                                primeros_2 = dict(list_items[:2])
                                ultimos_2 = dict(list_items[-2:])
                                
                                # Reconstruir el diccionario truncado
                                truncado = primeros_2
                                truncado[". . ."] = ". . ."
                                truncado.update(ultimos_2)
                                
                                datos[dispositivo] = truncado

        return reporte_worker3, reporte_llm

if __name__ == "__main__":
    plan_acciones=[{'id': '@1.1', 'server_id': 'mcp_server_gravity', 'tool': 'obtener_consumo', 'inputs': {'dispositivos': ['PC'], 'fecha_inicio': '2024-10-23T00:00:00', 'fecha_fin': '2024-10-23T23:59:59', 'granularidad': 'hora'}, 'descripcion': 'Obtener el consumo energético horario de la nevera para el día de ayer (23 de octubre de 2024).'}, 
                   {'id': '@2.1', 'server_id': 'mcp_server_gravity', 'tool': 'analizar_comparacion', 'inputs': {'objetivo_a': {'dispositivo': 'PC', 'fecha_inicio': '2024-10-23T00:00:00', 'fecha_fin': '2024-10-23T23:59:59'}, 'objetivo_b': {'dispositivo': 'PC', 'fecha_inicio': '2024-10-21T00:00:00', 'fecha_fin': '2024-10-21T23:59:59'}}, 'descripcion': 'Comparar el consumo total de la nevera del día de ayer (23 de octubre de 2024) con el consumo total de la nevera del lunes (21 de octubre de 2024).'}]
    
    client = MCP_Client()
    # Ejecutamos el plan de forma asíncrona
    reporte = asyncio.run(client.ejecutar_plan("http://localhost:8000/sse", plan_acciones))
    
    # Aplicar la función arreglar_reporte
    reporte_worker3, reporte_llm = client.arreglar_reporte(reporte)
    
    print("\n--- REPORTE ORIGINAL (Simulado) ---")
    print("El reporte original tiene todos los datos completos.")

    print("\n--- REPORTE WORKER 3 (Solo obtener_consumo full data) ---")
    print(json.dumps(reporte_worker3, indent=2, ensure_ascii=False))

    print("\n--- REPORTE LLM (Truncado) ---")
    print(json.dumps(reporte_llm, indent=2, ensure_ascii=False))

"""  
Reporte_LLM={
  "@1": [
    {
      "accion_id": "@1.1",
      "tool": "obtener_consumo",
      "descripcion": "Obtener el consumo energético horario de la nevera para el día de ayer (23 de octubre de 2024).",
      "resultado": {
        "status": "success",
        "periodo": {
          "inicio": "2024-10-23 00:00:00",
          "fin": "2024-10-23 23:59:59"
        },
        "granularidad": "hora",
        "datos": {
          "PC": {
            "2024-10-23T00:00:00": 0.0,
            "2024-10-23T01:00:00": 0.0001,
            ". . .": ". . .",
            "2024-10-23T22:00:00": 0.0,
            "2024-10-23T23:00:00": 0.0
          }
        }
      },
      "error": null
    }
  ],
  "@2": [
    {
      "accion_id": "@2.1",
      "tool": "analizar_comparacion",
      "descripcion": "Comparar el consumo total de la nevera del día de ayer (23 de octubre de 2024) con el consumo total de la nevera del lunes (21 de octubre de 2024).",
      "resultado": {
        "status": "success",
        "comparacion": {
          "valor_a": 0.0118,
          "valor_b": 0.0024000000000000002,
          "diferencia_absoluta": 0.009399999999999999,
          "diferencia_porcentual": 391.67,
          "mayor_consumo": "A"
        }
      },
      "error": null
    }
  ]
}

Reporte_worker3={
  "@1": [
    {
      "accion_id": "@1.1",
      "tool": "obtener_consumo",
      "descripcion": "Obtener el consumo energético horario de la nevera para el día de ayer (23 de octubre de 2024).",
      "resultado": {
        "status": "success",
        "periodo": {
          "inicio": "2024-10-23 00:00:00",
          "fin": "2024-10-23 23:59:59"
        },
        "granularidad": "hora",
        "datos": {
          "PC": {
            "2024-10-23T00:00:00": 0.0,
            "2024-10-23T01:00:00": 0.0001,
            "2024-10-23T02:00:00": 0.0001,
            "2024-10-23T03:00:00": 0.0001,
            "2024-10-23T04:00:00": 0.0001,
            "2024-10-23T05:00:00": 0.0001,
            "2024-10-23T06:00:00": 0.0001,
            "2024-10-23T07:00:00": 0.0001,
            "2024-10-23T08:00:00": 0.0001,
            "2024-10-23T09:00:00": 0.0001,
            "2024-10-23T10:00:00": 0.0001,
            "2024-10-23T11:00:00": 0.0001,
            "2024-10-23T12:00:00": 0.0102,
            "2024-10-23T13:00:00": 0.0,
            "2024-10-23T14:00:00": 0.0001,
            "2024-10-23T15:00:00": 0.0001,
            "2024-10-23T16:00:00": 0.0001,
            "2024-10-23T17:00:00": 0.0001,
            "2024-10-23T18:00:00": 0.0,
            "2024-10-23T19:00:00": 0.0,
            "2024-10-23T20:00:00": 0.0,
            "2024-10-23T21:00:00": 0.0001,
            "2024-10-23T22:00:00": 0.0,
            "2024-10-23T23:00:00": 0.0
          }
        }
      },
      "error": null
    }
  ]
}
"""