import asyncio
from fastmcp import Client
import numpy as np

# Dirección del servidor
SERVER_URL = "http://127.0.0.1:8000/sse"

# Variable con plan estructurado que llama a todas las funciones
ejemplo_plan_completo = [
    {
        "id": 0, 
        "funcion": "obtener_consumo", 
        "desc": "Obtener consumo total de la casa en octubre 2024", 
        "dependencias": {
            "dispositivos": ["Total_Casa"], 
            "fecha_inicio": "2024-10-01T00:00", 
            "fecha_fin": "2024-10-31T23:59",
            "granularidad": "total"
        }
    },
    {
        "id": 1, 
        "funcion": "detectar_anomalias", 
        "desc": "Detectar anomalías en el consumo de la casa en el mismo periodo", 
        "dependencias": {
            "dispositivo": "Total_Casa", 
            "fecha_inicio": "2024-10-01T00:00", 
            "fecha_fin": "2024-10-31T23:59",
            "sensibilidad": 3.0
        }
    },
    {
        "id": 2, 
        "funcion": "analizar_tendencia", 
        "desc": "Analizar la tendencia de consumo en octubre", 
        "dependencias": {
            "dispositivo": "Total_Casa", 
            "fecha_inicio": "2024-10-01T00:00", 
            "fecha_fin": "2024-10-31T23:59"
        }
    },
    {
        "id": 3, 
        "funcion": "obtener_consumo", 
        "desc": "Obtener consumo de noviembre para comparación", 
        "dependencias": {
            "dispositivos": ["Total_Casa"], 
            "fecha_inicio": "2024-11-01T00:00", 
            "fecha_fin": "2024-11-30T23:59",
            "granularidad": "total"
        }
    },
    {
        "id": 4, 
        "funcion": "analizar_comparacion", 
        "desc": "Comparar consumo de octubre vs noviembre", 
        "dependencias": {
            "objetivo_a": {
                "dispositivo": "Total_Casa", "fecha_inicio": "2024-10-01T00:00", "fecha_fin": "2024-10-31T23:59"
            },
            "objetivo_b": {
                "dispositivo": "Total_Casa", "fecha_inicio": "2024-11-01T00:00", "fecha_fin": "2024-11-30T23:59"
            }
        }
    },
    {
        "id": 5,
        "funcion": "falta_informacion",
        "desc": "Simular caso de falta de información",
        "dependencias": {
            "datos_faltante": "ID del medidor para validación externa"
        }
    },
    {
        "id": 6,
        "funcion": "plan_inviable",
        "desc": "Simular caso de plan inviable",
        "dependencias": {
            "razon": "Solicitud de predicción climática futura no soportada"
        }
    }
]

async def ejecutar_plan(server_url: str, plan: list) -> dict:
    """
    Ejecuta un plan de tareas contra un servidor MCP, resolviendo dependencias.
    """
    client = Client(server_url)
    resultados = {}
    
    print(f"Conectando a {server_url} para ejecutar plan...")
    
    try:
        async with client:
            await client.ping()
            
            for paso in plan:
                id_paso = paso["id"]
                funcion_nombre = paso["funcion"]
                desc = paso["desc"]
                params_originales = paso["dependencias"]
                
                print(f"Ejecutando paso {id_paso}: {funcion_nombre} - {desc}")
                
                # Resolver dependencias (formato @ID)
                params_finales = {}
                for k, v in params_originales.items():
                    if isinstance(v, str) and v.startswith("@"):
                        try:
                            ref_id = int(v[1:])
                            if ref_id in resultados:
                                # Asumimos que queremos el 'result' o el dato principal de la respuesta anterior
                                # Ajustar según la estructura de respuesta de las tools
                                prev_res = resultados[ref_id]["resultado"]
                                # Si la tool anterior devolvió un dict complejo, aquí habría que ver qué campo pasar.
                                # Por simplicidad del ejemplo asumimos paso directo si es compatible.
                                params_finales[k] = prev_res
                            else:
                                params_finales[k] = v # No se encontró, se deja cual cual (posible error)
                        except ValueError:
                            params_finales[k] = v
                    else:
                        params_finales[k] = v
                
                # Llamada a la herramienta
                try:
                    resultado_tool = await client.call_tool(funcion_nombre, params_finales)
                    
                    # Guardamos el resultado "crudo" o procesado según se requiera
                    resultados[id_paso] = {
                        "desc": desc,
                        "funcion": funcion_nombre,
                        "resultado": resultado_tool
                    }
                    print(f"  -> OK")
                except Exception as e:
                    print(f"  -> Error en herramienta: {e}")
                    resultados[id_paso] = {
                        "desc": desc,
                        "funcion": funcion_nombre,
                        "error": str(e),
                        "resultado": None
                    }

    except Exception as e:
        print(f"Error general de conexión o ejecución: {e}")
        
    return resultados

if __name__ == "__main__":
    # Prueba directa
    informe = asyncio.run(ejecutar_plan(SERVER_URL, ejemplo_plan_completo))
    import json
    print(json.dumps(informe, indent=2, default=str)) # default=str para serializar datetimes o numpy si los hubiera

