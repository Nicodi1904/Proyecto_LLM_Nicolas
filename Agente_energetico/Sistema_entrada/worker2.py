
import os
import sys
import json
from typing import List, Dict, Any, Union, Optional
from datetime import datetime

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

def verificar_estructura(plan_acciones: List[Dict[str, Any]], system_summary: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Verifica:
    1. Existencia de servidor y herramienta.
    2. Inputs obligatorios presentes.
    3. Tipos de datos correctos (check de tipos).

    Returns:
        Lista de diccionarios con formato:
        {
            'id': str,
            'resultado': int (0=OK, 1=Error),
            'error': str | None
        }
    """
    reporte = []
    mapa_herramientas = _generar_mapa_herramientas(system_summary)

    for accion in plan_acciones:
        id_accion = accion.get("id", "N/A")
        server_id = accion.get("server_id")
        tool_name = accion.get("tool")
        inputs = accion.get("inputs", {})
        
        errores = []

        # 1. Verificar existencia
        if not server_id or not tool_name:
            errores.append("Falta 'server_id' o 'tool'.")
        elif server_id not in mapa_herramientas:
            errores.append(f"Servidor desconocido: '{server_id}'.")
        elif tool_name not in mapa_herramientas[server_id]:
            errores.append(f"Herramienta desconocida: '{tool_name}' en servidor '{server_id}'.")
        else:
            # 2. Verificar Inputs contra Schema
            tool_def = mapa_herramientas[server_id][tool_name]
            input_schema = tool_def.get("meta", {}).get("input_schema", {})
            properties = input_schema.get("properties", {})
            required = input_schema.get("required", [])

            # a) Campos obligatorios
            for field in required:
                if field not in inputs:
                    errores.append(f"Falta parámetro obligatorio: '{field}'.")
                elif field in properties:
                    # b) Verificación de Tipos (solo si está presente)
                    expected_type = properties[field].get("type")
                    if expected_type and not _check_type(inputs[field], expected_type):
                         errores.append(
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
                         errores.append(
                             f"Tipo incorrecto en opcional '{field}': Se esperaba '{expected_type}', recibido '{type(inputs[field]).__name__}'."
                         )
                else:
                    # Campo desconocido
                    errores.append(f"Parámetro desconocido: '{field}'")

        if not errores:
            reporte.append({
                'id': id_accion,
                'resultado': 0,
                'error': None
            })
        else:
            reporte.append({
                'id': id_accion,
                'resultado': 1,
                'error': "; ".join(errores)
            })

    return reporte

# -------------------------------------------------------------------------
# 4. Función de Verificación Lógica de Argumentos
# -------------------------------------------------------------------------

def validar_argumentos(
    plan_acciones: List[Dict[str, Any]], 
    system_summary: Dict[str, Any],
    dispositivos_validos: List[str], 
    temporal_context: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """
    Verifica la lógica de los argumentos:
    - Que los dispositivos existan en la lista válida.
    - Que las fechas sean coherentes.
    - Que los valores de Enum (e.g. granularidad) sean permitidos según el schema.
    
    Returns:
        Lista de diccionarios con formato:
        {
            'id': str,
            'resultado': int (0=OK, 1=Error),
            'error': str | None
        }
    """
    reporte = []
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
        
        errores = []
        
        # --- 0. Validación de Enums (Valores Permitidos) ---
        if server_id and tool_name and server_id in mapa_herramientas and tool_name in mapa_herramientas[server_id]:
             tool_def = mapa_herramientas[server_id][tool_name]
             properties = tool_def.get("meta", {}).get("input_schema", {}).get("properties", {})
             
             for field, value in inputs.items():
                 # Si es referencia, saltar validación de enum
                 if isinstance(value, str) and value.startswith("@"):
                     continue
                     
                 if field in properties:
                     prop_def = properties[field]
                     if "enum" in prop_def:
                         allowed_values = prop_def["enum"]
                         if value not in allowed_values:
                             errores.append(f"Valor no permitido en '{field}': '{value}'. Permitidos: {allowed_values}")

        # --- 1. Validación de Dispositivos ---
        # Caso A: Lista directa 'dispositivos'
        if "dispositivos" in inputs:
            if isinstance(inputs["dispositivos"], list):
                for d in inputs["dispositivos"]:
                    # Si es referencia, saltar validación
                    if isinstance(d, str) and d.startswith("@"):
                        continue
                        
                    if isinstance(d, str) and d not in dispositivos_validos:
                        errores.append(f"Dispositivo desconocido: '{d}'")
        
        # Caso B: 'dispositivo' simple
        if "dispositivo" in inputs:
             val = inputs["dispositivo"]
             # Si es referencia, saltar
             if not (isinstance(val, str) and val.startswith("@")):
                 if val not in dispositivos_validos:
                     errores.append(f"Dispositivo desconocido: '{val}'")
                 
        # Caso C: Objetivos de comparación (objetivo_a/b -> dispositivo)
        for key in ["objetivo_a", "objetivo_b"]:
            if key in inputs and isinstance(inputs[key], dict):
                disp = inputs[key].get("dispositivo")
                # Si es referencia, saltar (aunque es raro referencia anidada al campo dispositivo, es posible)
                if isinstance(disp, str) and disp.startswith("@"):
                    continue
                    
                if disp and disp not in dispositivos_validos:
                    errores.append(f"Dispositivo desconocido en {key}: '{disp}'")

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
                continue 
            
            # SKIP si son referencias
            if f_ini_str.startswith("@") or f_fin_str.startswith("@"):
                continue
                
            try:
                dt_ini = datetime.fromisoformat(f_ini_str)
                dt_fin = datetime.fromisoformat(f_fin_str)
                
                if dt_ini > dt_fin:
                    errores.append(f"Fechas incoherentes en {contexto}: inicio ({f_ini_str}) > fin ({f_fin_str})")
                
                if dt_fin > dt_ref:
                    errores.append(f"Fecha futura en {contexto}: fin ({f_fin_str}) > referencia actual ({referencia_str})")
                    
            except ValueError:
                errores.append(f"Formato de fecha inválido en {contexto}")

        if not errores:
            reporte.append({
                'id': id_accion,
                'resultado': 0,
                'error': None
            })
        else:
            reporte.append({
                'id': id_accion,
                'resultado': 1,
                'error': "; ".join(errores)
            })
        
    return reporte

# -------------------------------------------------------------------------
# 5. Funciones Unificadas
# -------------------------------------------------------------------------

def verificar_completo(
    plan_acciones: List[Dict[str, Any]], 
    system_summary: Dict[str, Any],
    dispositivos_validos: List[str], 
    temporal_context: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """
    Ejecuta ambas verificaciones (Estructural y Lógica) y combina los resultados.
    
    Returns:
        Lista unificada de diccionarios con formato:
        {
            'id': str,
            'resultado': int (0=OK, 1=Error),
            'error': str | None (Errores concatenados si hay múltiples)
        }
    """
    # 1. Ejecutar ambas verificaciones
    rep_estructural = verificar_estructura(plan_acciones, system_summary)
    rep_logico = validar_argumentos(plan_acciones, system_summary, dispositivos_validos, temporal_context)
    
    # 2. Indexar por ID para unificación rápida
    map_est = {item['id']: item for item in rep_estructural}
    map_log = {item['id']: item for item in rep_logico}
    
    reporte_unificado = []
    
    for accion in plan_acciones:
        id_acc = accion.get("id", "N/A")
        res_est = map_est.get(id_acc, {'resultado': 1, 'error': 'Error crítico al procesar ID'})
        res_log = map_log.get(id_acc, {'resultado': 1, 'error': 'Error crítico al procesar ID'})
        
        errores_combinados = []
        if res_est['error']:
            errores_combinados.append(f"[Estructural] {res_est['error']}")
        if res_log['error']:
            errores_combinados.append(f"[Lógico] {res_log['error']}")
            
        estado_final = 0 if (res_est['resultado'] == 0 and res_log['resultado'] == 0) else 1
        error_final = "; ".join(errores_combinados) if errores_combinados else None
        
        reporte_unificado.append({
            'id': id_acc,
            'resultado': estado_final,
            'error': error_final
        })
        
    return reporte_unificado

def filtrar_acciones(reporte_unificado: List[Dict[str, Any]], plan_original: List[Dict[str, Any]]) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Separa las acciones del plan original en válidas e inválidas basándose en el reporte unificado.
    
    Returns:
        (acciones_validas, acciones_invalidas)
    """
    validas = []
    invalidas = []
    
    # Mapa de estado por ID
    mapa_estado = {item['id']: item for item in reporte_unificado}
    
    for accion in plan_original:
        id_acc = accion.get('id')
        estado = mapa_estado.get(id_acc)
        
        if estado and estado['resultado'] == 0:
            validas.append(accion)
        else:
            # Adjuntamos el error a la acción para referencia (opcional, pero útil)
            accion_con_error = accion.copy()
            if estado:
                accion_con_error['error_verificacion'] = estado['error']
            invalidas.append(accion_con_error)
            
    return validas, invalidas
