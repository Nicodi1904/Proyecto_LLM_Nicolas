import os
import json
import sys
import pandas as pd
from typing import Dict, Any, List

# --- 1. CONFIGURACIÓN DE RUTAS ---
# Ruta base: Pruebas_planeador/Pruebas_planeador/evaluacion_estructural/
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
# Resultados de inferencia están en ../Resultados
RESULTADOS_DIR = os.path.abspath(os.path.join(BASE_PATH, "..", "Resultados"))
# Salida de esta evaluación
OUTPUT_DIR = BASE_PATH
# Ruta al agente (para importar PlaneadorAgente)
AGENTE_PATH = os.path.abspath(os.path.join(BASE_PATH, "..", "..", "..", "Agente_energetico", "Sistema_entrada", "Planeador"))

if AGENTE_PATH not in sys.path:
    sys.path.append(AGENTE_PATH)

try:
    from evaluacion_estructural.modelos_evaluacion import Accion, PlanAcciones, validar_inputs_herramienta
except ImportError:
    # Si estamos corriendo desde la carpeta evaluacion_estructural/
    try:
        from modelos_evaluacion import Accion, PlanAcciones, validar_inputs_herramienta
    except ImportError as e:
        print(f"Error importando modelos_evaluacion: {e}")
        sys.exit(1)

def cargar_resultados(filtro_modelo: str = None) -> Dict[str, Any]:
    """Carga todos los archivos de respuestas en la carpeta Resultados."""
    resultados_por_modelo = {}
    
    if not os.path.exists(RESULTADOS_DIR):
        print(f"Error: No se encontró la carpeta de resultados en {RESULTADOS_DIR}")
        return {}

    for archivo in os.listdir(RESULTADOS_DIR):
        if archivo.startswith("respuestas_") and archivo.endswith(".json"):
            modelo_nombre = archivo.replace("respuestas_", "").replace(".json", "")
            
            if filtro_modelo and filtro_modelo not in modelo_nombre:
                continue
                
            ruta_archivo = os.path.join(RESULTADOS_DIR, archivo)
            try:
                with open(ruta_archivo, 'r', encoding='utf-8') as f:
                    resultados_por_modelo[modelo_nombre] = json.load(f)
            except Exception as e:
                print(f"Error cargando {archivo}: {e}")
                
    return resultados_por_modelo

def extraer_casos_recursivo(data: Any) -> Dict[str, Dict[str, Any]]:
    """Extrae todos los casos (IDs que tienen plan_acciones) de un JSON anidado."""
    casos = {}
    if isinstance(data, dict):
        if "plan_acciones" in data:
            return data
        for k, v in data.items():
            if isinstance(v, dict) and "plan_acciones" in v:
                casos[k] = v
            else:
                sub_casos = extraer_casos_recursivo(v)
                if isinstance(sub_casos, dict) and "plan_acciones" in sub_casos:
                    casos.update({k: sub_casos})
                elif isinstance(sub_casos, dict):
                    casos.update(sub_casos)
    return casos

def categorizar_error(error_msg: str) -> str:
    """Mapea mensajes de error técnicos a categorías de la tesis."""
    msg = error_msg.lower()
    if "formato válido (@n.m)" in msg or "match pattern" in msg or "id" in msg:
        if "pattern" in msg or "@" in msg: return "ID_MALFORMADO"
    if "falta parámetro obligatorio" in msg or "field required" in msg:
        return "PARAMETRO_FALTANTE"
    if "herramienta desconocida" in msg or "herramienta inexistente" in msg:
        return "HERRAMIENTA_INEXISTENTE"
    if "el plan de acciones está vacío" in msg:
        return "PLAN_VACIO"
    if "debe ser una lista" in msg:
        return "ESTRUCTURA_INVALIDA"
    if "discriminator" in msg or "union" in msg or "type" in msg or "input" in msg:
        return "INPUTS_INVALIDOS"
    return "ERROR_FORMATO_GENERAL"

def validar_plan_con_pydantic(plan: Any) -> Dict[str, Any]:
    """
    Validación determinista para la tesis usando Pydantic.
    Retorna un diccionario similar al worker2.
    """
    errores = []
    
    if not isinstance(plan, list):
        return {"valido": False, "errores": ["El plan de acciones debe ser una lista."]}
    
    if not plan:
        return {"valido": False, "errores": ["El plan de acciones está vacío."]}

    for i, accion_dict in enumerate(plan):
        try:
            # 1. Validar estructura de la acción
            acc = Accion(**accion_dict)
            
            # 2. Validar inputs específicos
            validar_inputs_herramienta(acc.tool, acc.inputs)
            
        except Exception as e:
            # Capturar errores de Pydantic detallados
            if hasattr(e, "errors"):
                for error in e.errors():
                    loc = " -> ".join([str(x) for x in error['loc']])
                    errores.append(f"Acción {i}: {loc}: {error['msg']}")
            else:
                errores.append(f"Acción {i}: {str(e)}")
                
    return {
        "valido": len(errores) == 0,
        "errores": errores
    }

def ejecutar_evaluacion_estructural():
    print("🚀 Iniciando Evaluación Estructural del Planeador (Determinista - Pydantic Local)...")
    
    resultados_modelos = cargar_resultados()
    detalles_globales = []

    for modelo, data_raw in resultados_modelos.items():
        print(f"  Analizando modelo: {modelo}...")
        
        casos_dict = extraer_casos_recursivo(data_raw)
        print(f"    Detectados {len(casos_dict)} casos.")
        
        conteo_validos = 0
        detalles_modelo = []

        for case_id, caso in casos_dict.items():
            plan = caso.get("plan_acciones")
            
            reporte = validar_plan_con_pydantic(plan)
            
            punto = 1 if reporte["valido"] else 0
            conteo_validos += punto
            
            # Categorizar errores únicos
            categorias = list(set([categorizar_error(e) for e in reporte["errores"]]))
            
            resumen_caso = {
                "Modelo": modelo,
                "Case_ID": case_id,
                "Validez_Estructural": punto,
                "Categorias_Error": ", ".join(categorias) if categorias else "NINGUNO",
                "Detalle_Tecnico": "; ".join(reporte["errores"]) if not reporte["valido"] else ""
            }
            
            detalles_globales.append(resumen_caso)
            detalles_modelo.append(resumen_caso)

        output_file = os.path.join(OUTPUT_DIR, f"eval_estructural_{modelo}.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(detalles_modelo, f, indent=4, ensure_ascii=False)
        
        print(f"    ✅ Archivo JSON actualizado: {os.path.basename(output_file)}")
        print(f"    📊 Resultado: {conteo_validos}/{len(casos_dict)} válidos.")

    if detalles_globales:
        df = pd.DataFrame(detalles_globales)
        excel_path = os.path.join(OUTPUT_DIR, "Resumen_Evaluacion_Estructural.xlsx")
        
        try:
            pivote = df.pivot(index='Case_ID', columns='Modelo', values='Validez_Estructural')
            
            with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
                pivote.to_excel(writer, sheet_name='Matriz_Binaria')
                df.to_excel(writer, sheet_name='Detalle_Errores', index=False)
                
            print(f"\n✨ Evaluación completada. Informe Excel regenerado en: {excel_path}")
        except Exception as e:
            print(f"Error generando Excel: {e}")
            df.to_excel(os.path.join(OUTPUT_DIR, "Resumen_Detallado.xlsx"), index=False)

if __name__ == "__main__":
    ejecutar_evaluacion_estructural()
