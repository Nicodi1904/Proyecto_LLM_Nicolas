
import os
import sys
import json
from typing import List, Dict, Any, Union
from datetime import datetime

# -------------------------------------------------------------------------
# 1. Configuración de Imports
# -------------------------------------------------------------------------
try:
    from MCP_C_obtener_summary import system_summary
except Exception as e:
    print(f"Error inicial importando MCP_C_obtener_summary: {e}")
    # Intento agregando path explícito
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        if current_dir not in sys.path:
            sys.path.append(current_dir)
        from MCP_C_obtener_summary import system_summary
    except Exception as e2:
        print(f"Error fatal: No se pudo importar system_summary de MCP_C_obtener_summary.py. Detalles: {e2}")
        sys.exit(1)

# -------------------------------------------------------------------------
# 2. Funciones Auxiliares
# -------------------------------------------------------------------------

def _check_type(value: Any, expected_type_str: str) -> bool:
    """Verifica si el valor corresponde al tipo esperado de JSON Schema."""
    if expected_type_str == "string":
        return isinstance(value, str)
    elif expected_type_str == "number" or expected_type_str == "integer":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    elif expected_type_str == "boolean":
        return isinstance(value, bool)
    elif expected_type_str == "array":
        return isinstance(value, list)
    elif expected_type_str == "object":
        return isinstance(value, dict)
    elif expected_type_str == "null":
        return value is None
    return True # Tipos desconocidos o "any" pasan

def _generar_mapa_herramientas(system_summary: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Genera un mapa server_id -> tool_name -> tool_def para búsqueda rápida."""
    mapa = {}
    if "servers" in system_summary:
        for server in system_summary["servers"]:
            s_id = server.get("server_id")
            if not s_id: continue
            mapa[s_id] = {}
            for tool in server.get("tools", []):
                t_name = tool.get("name")
                if t_name:
                    mapa[s_id][t_name] = tool
    return mapa

# -------------------------------------------------------------------------
# 3. Función de Verificación Estructural
# -------------------------------------------------------------------------

def verificar_estructura(plan_acciones: List[Dict[str, Any]], system_summary: Dict[str, Any]) -> Dict[str, Any]:
    """
    Verifica:
    1. Existencia de servidor y herramienta.
    2. Inputs obligatorios presentes.
    3. Tipos de datos correctos (check de tipos).
    """
    reporte_general = {
        "tipo_verificacion": "Estructural (Schema & Tipos)",
        "total_acciones": len(plan_acciones),
        "acciones_validas": 0,
        "acciones_invalidas": 0,
        "detalles": []
    }

    mapa_herramientas = _generar_mapa_herramientas(system_summary)

    for accion in plan_acciones:
        id_accion = accion.get("id", "N/A")
        server_id = accion.get("server_id")
        tool_name = accion.get("tool")
        inputs = accion.get("inputs", {})
        
        estado_accion = {
            "id": id_accion,
            "tool": tool_name,
            "valido": False,
            "errores": []
        }

        # 1. Verificar existencia
        if not server_id or not tool_name:
            estado_accion["errores"].append("Falta 'server_id' o 'tool'.")
        elif server_id not in mapa_herramientas:
            estado_accion["errores"].append(f"Servidor desconocido: '{server_id}'.")
        elif tool_name not in mapa_herramientas[server_id]:
            estado_accion["errores"].append(f"Herramienta desconocida: '{tool_name}' en servidor '{server_id}'.")
        else:
            # 2. Verificar Inputs contra Schema
            tool_def = mapa_herramientas[server_id][tool_name]
            input_schema = tool_def.get("meta", {}).get("input_schema", {})
            properties = input_schema.get("properties", {})
            required = input_schema.get("required", [])

            # a) Campos obligatorios
            for field in required:
                if field not in inputs:
                    estado_accion["errores"].append(f"Falta parámetro obligatorio: '{field}'.")
                elif field in properties:
                    # b) Verificación de Tipos (solo si está presente)
                    expected_type = properties[field].get("type")
                    if expected_type and not _check_type(inputs[field], expected_type):
                         estado_accion["errores"].append(
                             f"Tipo incorrecto en '{field}': Se esperaba '{expected_type}', recibido '{type(inputs[field]).__name__}'."
                         )

            # c) Campos desconocidos y sus tipos
            known_fields = set(properties.keys())
            provided_fields = set(inputs.keys())
            
            # Revisar tipos de campos opcionales presentes
            for field in provided_fields:
                if field in known_fields:
                     expected_type = properties[field].get("type")
                     if expected_type and not _check_type(inputs[field], expected_type):
                         estado_accion["errores"].append(
                             f"Tipo incorrecto en opcional '{field}': Se esperaba '{expected_type}', recibido '{type(inputs[field]).__name__}'."
                         )
                else:
                    # Campo desconocido
                    estado_accion["errores"].append(f"Parámetro desconocido: '{field}'")

        if not estado_accion["errores"]:
            estado_accion["valido"] = True
            reporte_general["acciones_validas"] += 1
        else:
            reporte_general["acciones_invalidas"] += 1
            
        reporte_general["detalles"].append(estado_accion)

    return reporte_general

# -------------------------------------------------------------------------
# 4. Función de Verificación Lógica de Argumentos
# -------------------------------------------------------------------------

def validar_argumentos(
    plan_acciones: List[Dict[str, Any]], 
    system_summary: Dict[str, Any],
    dispositivos_validos: List[str], 
    temporal_context: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Verifica la lógica de los argumentos:
    - Que los dispositivos existan en la lista válida.
    - Que las fechas sean coherentes.
    - Que los valores de Enum (e.g. granularidad) sean permitidos según el schema.
    """
    reporte = {
        "tipo_verificacion": "Lógica de Argumentos (Enums, Fechas, Dispositivos)",
        "total_acciones": len(plan_acciones),
        "acciones_validas": 0,
        "acciones_invalidas": 0,
        "detalles": []
    }
    
    mapa_herramientas = _generar_mapa_herramientas(system_summary)
    
    referencia_str = temporal_context.get("referencia_actual")
    try:
        dt_ref = datetime.fromisoformat(referencia_str) if referencia_str else datetime.now()
    except ValueError:
        dt_ref = datetime.now() 

    for accion in plan_acciones:
        id_accion = accion.get("id", "N/A")
        server_id = accion.get("server_id")
        tool_name = accion.get("tool")
        inputs = accion.get("inputs", {})
        
        estado = {
            "id": id_accion,
            "tool": tool_name,
            "valido": True,
            "errores": []
        }
        
        # --- 0. Validación de Enums (Valores Permitidos) ---
        if server_id and tool_name and server_id in mapa_herramientas and tool_name in mapa_herramientas[server_id]:
             tool_def = mapa_herramientas[server_id][tool_name]
             properties = tool_def.get("meta", {}).get("input_schema", {}).get("properties", {})
             
             for field, value in inputs.items():
                 if field in properties:
                     prop_def = properties[field]
                     if "enum" in prop_def:
                         allowed_values = prop_def["enum"]
                         if value not in allowed_values:
                             estado["errores"].append(f"Valor no permitido en '{field}': '{value}'. Permitidos: {allowed_values}")

        # --- 1. Validación de Dispositivos ---
        # Caso A: Lista directa 'dispositivos'
        if "dispositivos" in inputs:
            if isinstance(inputs["dispositivos"], list):
                for d in inputs["dispositivos"]:
                    if isinstance(d, str) and d not in dispositivos_validos:
                        estado["errores"].append(f"Dispositivo desconocido: '{d}'")
        
        # Caso B: 'dispositivo' simple
        if "dispositivo" in inputs:
             if inputs["dispositivo"] not in dispositivos_validos:
                 estado["errores"].append(f"Dispositivo desconocido: '{inputs['dispositivo']}'")
                 
        # Caso C: Objetivos de comparación (objetivo_a/b -> dispositivo)
        for key in ["objetivo_a", "objetivo_b"]:
            if key in inputs and isinstance(inputs[key], dict):
                disp = inputs[key].get("dispositivo")
                if disp and disp not in dispositivos_validos:
                    estado["errores"].append(f"Dispositivo desconocido en {key}: '{disp}'")

        # --- 2. Validación de Fechas ---
        fechas_a_validar = []
        if "fecha_inicio" in inputs and "fecha_fin" in inputs:
            fechas_a_validar.append((inputs["fecha_inicio"], inputs["fecha_fin"], "General"))
            
        for key in ["objetivo_a", "objetivo_b"]:
            if key in inputs and isinstance(inputs[key], dict):
                f_ini = inputs[key].get("fecha_inicio")
                f_fin = inputs[key].get("fecha_fin")
                if f_ini and f_fin:
                    fechas_a_validar.append((f_ini, f_fin, f"{key}"))

        for f_ini_str, f_fin_str, contexto in fechas_a_validar:
            if not isinstance(f_ini_str, str) or not isinstance(f_fin_str, str):
                continue # Ya debería haber fallado en el chequeo de tipos
                
            try:
                dt_ini = datetime.fromisoformat(f_ini_str)
                dt_fin = datetime.fromisoformat(f_fin_str)
                
                if dt_ini > dt_fin:
                    estado["errores"].append(f"Fechas incoherentes en {contexto}: inicio ({f_ini_str}) > fin ({f_fin_str})")
                
                if dt_fin > dt_ref:
                    estado["errores"].append(f"Fecha futura en {contexto}: fin ({f_fin_str}) > referencia actual ({referencia_str})")
                    
            except ValueError:
                estado["errores"].append(f"Formato de fecha inválido en {contexto}")

        if estado["errores"]:
            estado["valido"] = False
            reporte["acciones_invalidas"] += 1
        else:
            reporte["acciones_validas"] += 1
            
        reporte["detalles"].append(estado)
        
    return reporte


# -------------------------------------------------------------------------
# 5. Bloque Principal de Prueba
# -------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"\n{'='*60}")
    print("INICIANDO WORKER VERIFICADOR")
    print(f"{'='*60}\n")
    
    # Contexto Simulado
    dispositivos_conocidos = ["nevera", "lavadora", "Total_Casa", "aire_acondicionado", "luces"]
    contexto_temporal = {
        "referencia_actual": "2024-12-25T18:00:00"
    }

    # Ejemplo con un error de tipo (granularidad=123) y un error de enum (granularidad="semanal") para probar
    plan_ejemplo = [{'id': '@1.1', 'server_id': 'mcp_server_gravity', 'tool': 'obtener_consumo', 'inputs': {'dispositivos': ['nevera'], 'fecha_inicio': '2024-11-14T18:00', 'fecha_fin': '2024-11-14T23:59', 'granularidad': 'total'}, 'descripcion': 'Obtener consumo de la nevera durante la noche de ayer (2024-11-14)'}, {'id': '@2.1', 'server_id': 'mcp_server_gravity', 'tool': 'obtener_consumo', 'inputs': {'dispositivos': ['lavadora'], 'fecha_inicio': '2024-11-09T06:00', 'fecha_fin': '2024-11-09T11:59', 'granularidad': 'total'}, 'descripcion': 'Obtener consumo de la lavadora durante la mañana del sábado pasado (2024-11-09)'}, {'id': '@3.1', 'server_id': 'mcp_server_gravity', 'tool': 'obtener_consumo', 'inputs': {'dispositivos': ['nevera'], 'fecha_inicio': '2024-11-14T00:00', 'fecha_fin': '2024-11-14T23:59', 'granularidad': 'total'}, 'descripcion': 'Obtener consumo diario de la nevera para comparación (periodo supuesto 2024-11-14)'}, {'id': '@3.2', 'server_id': 'mcp_server_gravity', 'tool': 'obtener_consumo', 'inputs': {'dispositivos': ['lavadora'], 'fecha_inicio': '2024-11-14T00:00', 'fecha_fin': '2024-11-14T23:59', 'granularidad': 'total'}, 'descripcion': 'Obtener consumo diario de la lavadora para comparación (periodo supuesto 2024-11-14)'}, {'id': '@3.3', 'server_id': 'mcp_server_gravity', 'tool': 'analizar_comparacion', 'inputs': {'objetivo_a': {'dispositivo': 'nevera', 'fecha_inicio': '2024-11-14T00:00', 'fecha_fin': '2024-11-14T23:59'}, 'objetivo_b': {'dispositivo': 'lavadora', 'fecha_inicio': '2024-11-14T00:00', 'fecha_fin': '2024-11-14T23:59'}}, 'descripcion': 'Comparar consumos entre nevera y lavadora en periodo común supuesto'}, {'id': '@4.1', 'server_id': 'mcp_server_gravity', 'tool': 'obtener_consumo', 'inputs': {'dispositivos': ['Total_Casa'], 'fecha_inicio': '2024-01-01T00:00', 'fecha_fin': '2024-12-31T23:59', 'granularidad': 'mes'}, 'descripcion': 'Obtener consumo mensual agregado de todos los dispositivos para 2024'}]

    if system_summary:
        print("System Summary cargado correctamente.\n")
        
        print("--- PASO 1: VERIFICACIÓN ESTRUCTURAL ---")
        resultado_est = verificar_estructura(plan_ejemplo, system_summary)
        print(f"  Válidas:  {resultado_est['acciones_validas']} / {resultado_est['total_acciones']}")
        
        for det in resultado_est['detalles']:
            if not det['valido']:
                print(f"    [X] {det['id']}: {'; '.join(det['errores'])}")
            else:
                print(f"    [OK] {det['id']}")
            
        print("\n--- PASO 2: VERIFICACIÓN LÓGICA ---")
        resultado_log = validar_argumentos(plan_ejemplo, system_summary, dispositivos_conocidos, contexto_temporal)
        print(f"  Válidas:  {resultado_log['acciones_validas']} / {resultado_log['total_acciones']}")
        
        for det in resultado_log['detalles']:
            if not det['valido']:
                print(f"    [X] {det['id']}: {'; '.join(det['errores'])}")
            else:
                print(f"    [OK] {det['id']}")
            
    else:
        print("Error crítico: system_summary está vacío o nulo.")
