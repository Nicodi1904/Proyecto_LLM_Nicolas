
import os
import sys
import json
from typing import List, Dict, Any

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
# 2. Función de Verificación Estructural
# -------------------------------------------------------------------------

def verificar_plan(plan_acciones: List[Dict[str, Any]], system_summary: Dict[str, Any]) -> Dict[str, Any]:
    """
    Verifica una lista de acciones planificadas contra el system_summary.
    
    Args:
        plan_acciones: Lista de acciones propuestas por el planeador.
        system_summary: Resumen de herramientas y schemas del sistema.
        
    Returns:
        Un diccionario con el estado general y una lista de reportes por acción.
    """
    reporte_general = {
        "tipo_verificacion": "Estructural (Schema)",
        "total_acciones": len(plan_acciones),
        "acciones_validas": 0,
        "acciones_invalidas": 0,
        "detalles": []
    }

    # Mapa rápido de herramientas: server_id -> tool_name -> tool_definition
    mapa_herramientas = {}
    if "servers" in system_summary:
        for server in system_summary["servers"]:
            s_id = server.get("server_id")
            if not s_id: continue
            mapa_herramientas[s_id] = {}
            for tool in server.get("tools", []):
                t_name = tool.get("name")
                if t_name:
                    mapa_herramientas[s_id][t_name] = tool

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

        # 1. Verificar existencia de la herramienta
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

            # a) Verificar campos obligatorios
            for field in required:
                if field not in inputs:
                    estado_accion["errores"].append(f"Falta parámetro obligatorio: '{field}'.")

            # b) Verificar campos desconocidos (opcional, pero buena práctica)
            known_fields = set(properties.keys())
            provided_fields = set(inputs.keys())
            unknown = provided_fields - known_fields
            if unknown:
                # Nota: A veces las llamadas internas usan referencias @N.M, que no están en el schema simple.
                # Asumiremos que si empiezan por ningún prefijo especial son error, 
                # pero validaremos solo los nombres de propiedad.
                estado_accion["errores"].append(f"Parámetros desconocidos: {list(unknown)}")

        if not estado_accion["errores"]:
            estado_accion["valido"] = True
            reporte_general["acciones_validas"] += 1
        else:
            reporte_general["acciones_invalidas"] += 1
            
        reporte_general["detalles"].append(estado_accion)

    return reporte_general

# -------------------------------------------------------------------------
# 3. Función de Verificación Lógica de Argumentos
# -------------------------------------------------------------------------

def validar_argumentos(plan_acciones: List[Dict[str, Any]], dispositivos_validos: List[str], temporal_context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Verifica la lógica de los argumentos:
    - Que los dispositivos existan en la lista de dispositivos válidos.
    - Que las fechas sean coherentes (inicio <= fin, fin <= referencia_actual).
    
    Args:
        plan_acciones: Lista de acciones.
        dispositivos_validos: Lista de nombres de dispositivos permitidos.
        temporal_context: Diccionario con 'referencia_actual'.
        
    Returns:
        Reporte de validación lógica.
    """
    reporte = {
        "tipo_verificacion": "Lógica de Argumentos",
        "total_acciones": len(plan_acciones),
        "acciones_validas": 0,
        "acciones_invalidas": 0,
        "detalles": []
    }
    
    referencia_str = temporal_context.get("referencia_actual")
    try:
        dt_ref = datetime.fromisoformat(referencia_str) if referencia_str else datetime.now()
    except ValueError:
        dt_ref = datetime.now() 

    for accion in plan_acciones:
        id_accion = accion.get("id", "N/A")
        inputs = accion.get("inputs", {})
        tool_name = accion.get("tool", "Desconocida")
        
        estado = {
            "id": id_accion,
            "tool": tool_name,
            "valido": True, # Asumimos válido hasta encontrar error
            "errores": []
        }
        
        # --- 1. Validación de Dispositivos ---
        # Caso A: Lista directa 'dispositivos'
        if "dispositivos" in inputs:
            for d in inputs["dispositivos"]:
                if d not in dispositivos_validos:
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
        # Helper para extraer fechas de inputs planos o anidados
        fechas_a_validar = [] # Tuplas (inicio, fin, contexto)
        
        # Input plano
        if "fecha_inicio" in inputs and "fecha_fin" in inputs:
            fechas_a_validar.append((inputs["fecha_inicio"], inputs["fecha_fin"], "General"))
            
        # Input anidado (comparacion)
        for key in ["objetivo_a", "objetivo_b"]:
            if key in inputs and isinstance(inputs[key], dict):
                f_ini = inputs[key].get("fecha_inicio")
                f_fin = inputs[key].get("fecha_fin")
                if f_ini and f_fin:
                    fechas_a_validar.append((f_ini, f_fin, f"{key}"))

        for f_ini_str, f_fin_str, contexto in fechas_a_validar:
            try:
                # Intento básico de parseo ISO
                dt_ini = datetime.fromisoformat(f_ini_str)
                dt_fin = datetime.fromisoformat(f_fin_str)
                
                # Regla 1: Inicio <= Fin
                if dt_ini > dt_fin:
                    estado["errores"].append(f"Fechas incoherentes en {contexto}: inicio ({f_ini_str}) > fin ({f_fin_str})")
                
                # Regla 2: Fin <= Referencia Actual
                # (Solo si la herramienta implica datos históricos, asumimos que sí por ahora)
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
# 4. Bloque Principal de Prueba
# -------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"\n{'='*60}")
    print("INICIANDO WORKER VERIFICADOR")
    print(f"{'='*60}\n")
    
    # 1. Datos de Contexto Simulados
    dispositivos_conocidos = ["nevera", "lavadora", "Total_Casa", "aire_acondicionado", "luces"]
    contexto_temporal = {
        "referencia_actual": "2024-12-25T18:00:00"
    }

    # 1. Lista estructurada de ejemplo (input del usuario)
    plan_ejemplo = [
        {'id': '@1.1', 'server_id': 'mcp_server_gravity', 'tool': 'obtener_consumo', 'inputs': {'dispositivos': ['nevera'], 'fecha_inicio': '2024-11-14T18:00', 'fecha_fin': '2024-11-14T23:59', 'granularidad': 'total'}, 'descripcion': 'Obtener consumo de la nevera durante la noche de ayer (2024-11-14)'}, 
        {'id': '@2.1', 'server_id': 'mcp_server_gravity', 'tool': 'obtener_consumo', 'inputs': {'dispositivos': ['lavadora'], 'fecha_inicio': '2024-11-09T06:00', 'fecha_fin': '2024-11-09T11:59', 'granularidad': 'total'}, 'descripcion': 'Obtener consumo de la lavadora durante la mañana del sábado pasado (2024-11-09)'}, 
        {'id': '@3.1', 'server_id': 'mcp_server_gravity', 'tool': 'obtener_consumo', 'inputs': {'dispositivos': ['nevera'], 'fecha_inicio': '2024-11-14T00:00', 'fecha_fin': '2024-11-14T23:59', 'granularidad': 'total'}, 'descripcion': 'Obtener consumo diario de la nevera para comparación (periodo supuesto 2024-11-14)'}, 
        {'id': '@3.2', 'server_id': 'mcp_server_gravity', 'tool': 'obtener_consumo', 'inputs': {'dispositivos': ['lavadora'], 'fecha_inicio': '2024-11-14T00:00', 'fecha_fin': '2024-11-14T23:59', 'granularidad': 'total'}, 'descripcion': 'Obtener consumo diario de la lavadora para comparación (periodo supuesto 2024-11-14)'}, 
        {'id': '@3.3', 'server_id': 'mcp_server_gravity', 'tool': 'analizar_comparacion', 'inputs': {'objetivo_a': {'dispositivo': 'nevera', 'fecha_inicio': '2024-11-14T00:00', 'fecha_fin': '2024-11-14T23:59'}, 'objetivo_b': {'dispositivo': 'lavadora', 'fecha_inicio': '2024-11-14T00:00', 'fecha_fin': '2024-11-14T23:59'}}, 'descripcion': 'Comparar consumos entre nevera y lavadora en periodo común supuesto'}, 
        {'id': '@4.1', 'server_id': 'mcp_server_gravity', 'tool': 'obtener_consumo', 'inputs': {'dispositivos': ['Total_Casa'], 'fecha_inicio': '2024-01-01T00:00', 'fecha_fin': '2024-12-31T23:59', 'granularidad': 'mes'}, 'descripcion': 'Obtener consumo mensual agregado de todos los dispositivos para 2024'}
    ]

    # 2. Ejecutar Verificación Estructural
    if system_summary:
        print("System Summary cargado correctamente.")
        print(f"Validando {len(plan_ejemplo)} acciones...\n")
        
        print("--- PASO 1: VERIFICACIÓN ESTRUCTURAL ---")
        resultado_est = verificar_plan(plan_ejemplo, system_summary)
        print(f"  Válidas:  {resultado_est['acciones_validas']} / {resultado_est['total_acciones']}")
        
        if resultado_est['acciones_invalidas'] > 0:
            print("\n  ! Errores Estructurales Detectados:")
            for det in resultado_est['detalles']:
                if not det['valido']:
                    print(f"    - Acción {det['id']}: {', '.join(det['errores'])}")
        else:
            print("  > Estructura Correcta.")
            
        print("\n--- PASO 2: VERIFICACIÓN LÓGICA ---")
        resultado_log = validar_argumentos(plan_ejemplo, dispositivos_conocidos, contexto_temporal)
        print(f"  Válidas:  {resultado_log['acciones_validas']} / {resultado_log['total_acciones']}")
        
        if resultado_log['acciones_invalidas'] > 0:
            print("\n  ! Errores Lógicos Detectados:")
            for det in resultado_log['detalles']:
                if not det['valido']:
                    print(f"    - Acción {det['id']}: {', '.join(det['errores'])}")
        else:
            print("  > Lógica Correcta.")
            
    else:
        print("Error crítico: system_summary está vacío o nulo.")
