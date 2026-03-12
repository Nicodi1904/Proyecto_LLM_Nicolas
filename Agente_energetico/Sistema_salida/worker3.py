import json
from typing import List, Dict, Any, Optional

def detectar_datos_graficables(reporte_acciones: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Dict[str, Any]]:
    """
    Analiza el reporte de acciones para detectar aquellas solicitudes que han producido
    datos transformables en información gráfica.
    
    Busca patrones de datos como:
    - Listas de diccionarios (series temporales).
    - Diccionarios con múltiples valores numéricos (comparaciones).
    - Resultados de herramientas de análisis de tendencias.
    
    Returns:
        Dict mapeando ID de solicitud (@N) a un descriptor de vista gráfica sugerida.
    """
    vistas_sugeridas = {}

    for id_solicitud, acciones in reporte_acciones.items():
        for accion in acciones:
            resultado = accion.get("resultado")
            if not resultado or accion.get("error"):
                continue
            
            es_graficable = False
            tipo_grafico = None
            
            # Caso 1: Series Temporales (Listas de objetos con tiempo y valor)
            if isinstance(resultado, list) and len(resultado) > 1:
                if all(isinstance(item, dict) and any(k in item for k in ["fecha", "timestamp", "hora"]) for item in resultado[:2]):
                    es_graficable = True
                    tipo_grafico = "line_chart"
            
            # Caso 2: Comparaciones o Agregaciones (Diccionarios con claves y valores numéricos)
            elif isinstance(resultado, dict):
                valores_numericos = [v for v in resultado.values() if isinstance(v, (int, float))]
                if len(valores_numericos) >= 2:
                    es_graficable = True
                    tipo_grafico = "bar_chart"
            
            if es_graficable:
                vistas_sugeridas[id_solicitud] = {
                    "accion_id": accion.get("accion_id"),
                    "tipo_sugerido": tipo_grafico,
                    "herramienta": accion.get("tool"),
                    "fuente_datos": "resultado_ejecucion"
                }
                # Una vez detectado que una solicitud es graficable, pasamos a la siguiente
                break

    return vistas_sugeridas

def preparar_contexto_visual(vistas_sugeridas: Dict[str, Any], reporte_acciones: Dict[str, Any]) -> Dict[str, Any]:
    """
    Suministra al usuario una vista adicional para interpretar la información
    cuando Worker3 detecta datos transformables.
    """
    contexto_visual = {
        "activar_vistas_adicionales": len(vistas_sugeridas) > 0,
        "configuraciones_visuales": vistas_sugeridas,
        "mensaje_sistema": "Se han detectado datos adecuados para representación visual." if vistas_sugeridas else None
    }
    return contexto_visual
