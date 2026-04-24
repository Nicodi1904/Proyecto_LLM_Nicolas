"""
Evaluación Estructural Determinista del Agente Interpretador.

Lee la columna 'Respuesta_Modelo' del Excel comparativa_inferenciador.xlsx,
parsea el dict de solicitudes_categorizadas, lo valida con Pydantic y
genera un JSON por modelo + un Excel resumen.
"""

import os
import re
import ast
import json
import sys
import pandas as pd
from typing import Any, Dict, List

# ---------------------------------------------------------------------------
# RUTAS
# ---------------------------------------------------------------------------
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
EXCEL_PATH = os.path.abspath(os.path.join(
    BASE_PATH, "..", "Informes_Inferenciador_DeepEval", "comparativa_inferenciador.xlsx"
))
OUTPUT_DIR = BASE_PATH

# Importar modelos de validación del mismo directorio
sys.path.insert(0, BASE_PATH)
try:
    from modelos_evaluacion_interpretador import validar_solicitudes_categorizadas
except ImportError as e:
    print(f"Error importando modelos_evaluacion_interpretador: {e}")
    sys.exit(1)

# Hojas del Excel que NO son modelos
HOJAS_META = {"Comparaciones", "Orden_preguntas"}

# ---------------------------------------------------------------------------
# PARSEO DE Respuesta_Modelo
# ---------------------------------------------------------------------------

def parsear_respuesta(texto: str) -> Any:
    """
    Extrae el diccionario de solicitudes_categorizadas del string crudo.

    El formato observado es:
        " {'@1': {'solicitud': '...', 'escenario': '...'}} | notas: ..."

    Estrategia:
      1. Separar por ' | notas:' y quedarse con la parte izquierda.
      2. Intentar ast.literal_eval para parsear el dict Python.
      3. Si falla (modelo usó comillas dobles/JSON), intentar json.loads.
      4. Si falla todo, retornar None.
    """
    if not isinstance(texto, str):
        return None

    # Tomar solo la parte antes del separador de notas
    partes = texto.split(" | notas:")
    cuerpo = partes[0].strip()

    # Intento 1: ast.literal_eval (maneja dicts Python con comillas simples)
    try:
        resultado = ast.literal_eval(cuerpo)
        if isinstance(resultado, dict):
            return resultado
    except Exception:
        pass

    # Intento 2: json.loads (por si el modelo generó JSON válido)
    try:
        resultado = json.loads(cuerpo)
        if isinstance(resultado, dict):
            return resultado
    except Exception:
        pass

    # Intento 3: buscar con regex el primer {...} del texto completo
    match = re.search(r"\{.*\}", texto, re.DOTALL)
    if match:
        try:
            resultado = ast.literal_eval(match.group())
            if isinstance(resultado, dict):
                return resultado
        except Exception:
            pass
        try:
            resultado = json.loads(match.group())
            if isinstance(resultado, dict):
                return resultado
        except Exception:
            pass

    return None  # no se pudo parsear


# ---------------------------------------------------------------------------
# CATEGORIZACIÓN DE ERRORES
# ---------------------------------------------------------------------------

def categorizar_error(error_msg: str) -> str:
    msg = error_msg.lower()
    if "formato '@n'" in msg or "clave" in msg:
        return "CLAVE_MAL_FORMADA"
    if "escenario desconocido" in msg:
        return "ESCENARIO_INVALIDO"
    if "solicitud" in msg and ("field required" in msg or "vacío" in msg or "min_length" in msg):
        return "SOLICITUD_VACIA"
    if "field required" in msg:
        return "CAMPO_FALTANTE"
    if "debe ser un diccionario" in msg or "dict" in msg:
        return "ESTRUCTURA_INVALIDA"
    if "vacío" in msg or "empty" in msg:
        return "SALIDA_VACIA"
    return "ERROR_FORMATO_GENERAL"


# ---------------------------------------------------------------------------
# EVALUACIÓN PRINCIPAL
# ---------------------------------------------------------------------------

def ejecutar_evaluacion_estructural():
    print("🚀 Iniciando Evaluación Estructural del Interpretador (Determinista - Pydantic)...")

    if not os.path.exists(EXCEL_PATH):
        print(f"❌ No se encontró el Excel: {EXCEL_PATH}")
        sys.exit(1)

    # Cargar el Excel completo (todas las hojas)
    xl = pd.ExcelFile(EXCEL_PATH)
    hojas_modelos = [s for s in xl.sheet_names if s not in HOJAS_META]
    print(f"   Modelos detectados: {hojas_modelos}\n")

    detalles_globales: List[Dict] = []

    for modelo in hojas_modelos:
        print(f"  Analizando modelo: {modelo}...")
        df = xl.parse(modelo)

        # Verificar que la columna Respuesta_Modelo exista
        if "Respuesta_Modelo" not in df.columns:
            print(f"    ⚠️  Columna 'Respuesta_Modelo' no encontrada en hoja '{modelo}'. Saltando.")
            continue

        detalles_modelo: List[Dict] = []
        conteo_validos = 0

        for idx, row in df.iterrows():
            case_id = row.get("ID_PREGUNTA", idx + 1)
            texto_raw = row.get("Respuesta_Modelo", "")

            # 1. Parsear el string a dict
            solicitudes_dict = parsear_respuesta(texto_raw)

            if solicitudes_dict is None:
                # No se pudo parsear: error de formato fundamental
                reporte = {
                    "valido": False,
                    "errores": [f"No se pudo parsear la respuesta del modelo como diccionario Python/JSON."],
                }
            else:
                # 2. Validar con Pydantic
                reporte = validar_solicitudes_categorizadas(solicitudes_dict)

            punto = 1 if reporte["valido"] else 0
            conteo_validos += punto

            categorias = list(set(categorizar_error(e) for e in reporte["errores"]))

            resumen_caso = {
                "Modelo": modelo,
                "Case_ID": case_id,
                "Validez_Estructural": punto,
                "Categorias_Error": ", ".join(categorias) if categorias else "NINGUNO",
                "Detalle_Tecnico": "; ".join(reporte["errores"]) if not reporte["valido"] else "",
            }

            detalles_globales.append(resumen_caso)
            detalles_modelo.append(resumen_caso)

        # Guardar JSON del modelo
        output_json = os.path.join(OUTPUT_DIR, f"eval_estructural_{modelo}.json")
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(detalles_modelo, f, indent=4, ensure_ascii=False)

        print(f"    ✅ JSON generado: {os.path.basename(output_json)}")
        print(f"    📊 Resultado: {conteo_validos}/{len(detalles_modelo)} válidos.")

    # ---------------------------------------------------------------------------
    # GENERAR EXCEL RESUMEN
    # ---------------------------------------------------------------------------
    if detalles_globales:
        df_all = pd.DataFrame(detalles_globales)
        excel_out = os.path.join(OUTPUT_DIR, "Resumen_Evaluacion_Estructural_Interpretador.xlsx")

        try:
            pivote = df_all.pivot(index="Case_ID", columns="Modelo", values="Validez_Estructural")
            with pd.ExcelWriter(excel_out, engine="openpyxl") as writer:
                pivote.to_excel(writer, sheet_name="Matriz_Binaria")
                df_all.to_excel(writer, sheet_name="Detalle_Errores", index=False)
            print(f"\n✨ Excel generado: {excel_out}")
        except Exception as e:
            print(f"   Error generando Excel con pivote ({e}). Guardando sin pivote...")
            df_all.to_excel(os.path.join(OUTPUT_DIR, "Resumen_Detallado_Interpretador.xlsx"), index=False)


if __name__ == "__main__":
    ejecutar_evaluacion_estructural()
