import json
from typing import List, Dict, Any, Optional

def validar_salida_interpretador(
    solicitudes_categorizadas: Dict[str, Any], 
    escenarios_entrada: Dict[str, Any],
    notas: str
) -> Dict[str, Any]:
    """
    Gestiona las salidas producidas por el interpretador, revisando formato y coherencia.
    
    Verifica:
    1. Que las solicitudes estén debidamente enumeradas (@1, @2, ...).
    2. Que cada solicitud tenga los campos 'solicitud' y 'escenario'.
    3. Que los escenarios asignados existan en la lista de permitidos.
    4. Que el campo 'notas' no esté vacío.
    
    Returns:
        Dict con 'valido' (bool), 'errores' (list) y 'corregido' (dict/str).
    """
    errores = []
    peticiones_validas = {}
    
    # 1. Validar Peticiones Categorizadas
    if not isinstance(solicitudes_categorizadas, dict):
        return {
            "valido": False,
            "errores": ["Las peticiones categorizadas no tienen un formato de diccionario válido."],
            "peticiones": solicitudes_categorizadas,
            "notas": notas
        }

    if not solicitudes_categorizadas:
        errores.append("No se detectaron peticiones en el mensaje del usuario.")

    for key, val in solicitudes_categorizadas.items():
        # Validar formato de la clave @N
        if not (isinstance(key, str) and key.startswith("@")):
            errores.append(f"Clave de petición con formato incorrecto: {key}. Debe ser '@N'.")
        
        # Validar campos internos
        if not isinstance(val, dict):
            errores.append(f"El valor de la petición {key} debe ser un diccionario.")
            continue
            
        solicitud_texto = val.get("solicitud")
        escenario_nombre = val.get("escenario")
        
        if not solicitud_texto:
            errores.append(f"Falta el texto de la solicitud en {key}.")
        
        if not escenario_nombre:
            errores.append(f"Falta el escenario en {key}.")
        elif escenario_nombre not in escenarios_entrada:
            errores.append(f"Escenario desconocido en {key}: '{escenario_nombre}'.")
        
        # Si no hay errores críticos en esta petición, la consideramos válida
        peticiones_validas[key] = val

    # 2. Validar Notas
    if not notas or not isinstance(notas, str) or len(notas.strip()) < 10:
        errores.append("Las notas de razonamiento son insuficientes o están ausentes.")

    return {
        "valido": len(errores) == 0,
        "errores": errores,
        "peticiones": peticiones_validas,
        "notas": notas
    }

def corregir_formato_json(raw_string: str) -> Optional[Dict[str, Any]]:
    """
    Intenta extraer y corregir un JSON de una cadena de texto en caso de errores de formato leves.
    """
    try:
        # Intento directo
        return json.loads(raw_string)
    except json.JSONDecodeError:
        # Intento de limpieza básica (quitar markdown blocks si existen)
        if "```json" in raw_string:
            try:
                content = raw_string.split("```json")[1].split("```")[0].strip()
                return json.loads(content)
            except:
                pass
    return None
