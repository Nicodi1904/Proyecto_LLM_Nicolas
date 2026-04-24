import os
import json
import pandas as pd

def generar_excel_desde_json(ruta_eval, ruta_results, ruta_questions, ruta_salida):
    """Convierte un archivo JSON de evaluación a un Excel con preguntas y respuestas."""
    with open(ruta_eval, 'r', encoding='utf-8') as f:
        data_eval = json.load(f)
    
    with open(ruta_questions, 'r', encoding='utf-8') as f:
        data_questions = json.load(f)
        
    with open(ruta_results, 'r', encoding='utf-8') as f:
        data_results = json.load(f)

    # 1. Extraer datos para la tabla de casos
    filas_casos = []
    
    for nivel, info_nivel in data_eval.get("niveles", {}).items():
        promedio_nivel = info_nivel.get("promedio", 0)
        for id_caso, detalle in info_nivel.get("casos", {}).items():
            
            # Obtener pregunta y respuesta original
            pregunta = data_questions.get(nivel, {}).get(id_caso, {})
            respuesta = data_results.get(nivel, {}).get(id_caso, {})
            
            # Extraer notas específicas si existen
            notas_modelo = respuesta.get("notas", "") if isinstance(respuesta, dict) else ""
            
            filas_casos.append({
                "Nivel": nivel,
                "Promedio Nivel": promedio_nivel,
                "ID Caso": id_caso,
                "Pregunta Original": json.dumps(pregunta, indent=2, ensure_ascii=False),
                "Notas del Modelo": notas_modelo,
                "Respuesta del Modelo": json.dumps(respuesta, indent=2, ensure_ascii=False),
                "Score": detalle.get("score", 0),
                "Reason": detalle.get("reason", "")
            })

    df_casos = pd.DataFrame(filas_casos)

    # 2. Crear Resumen Global
    resumen_global = {
        "Métrica": ["Promedio Global", "Total Casos Evaluados"],
        "Valor": [data_eval.get("promedio_global", 0), data_eval.get("total_casos_evaluados", 0)]
    }
    df_resumen = pd.DataFrame(resumen_global)

    # 3. Guardar en Excel con formato
    with pd.ExcelWriter(ruta_salida, engine='openpyxl') as writer:
        df_casos.to_excel(writer, sheet_name='Detalle de Evaluación', index=False)
        df_resumen.to_excel(writer, sheet_name='Resumen Global', index=False)

    print(f"📊 Excel generado: {ruta_salida}")

def main():
    # Rutas
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    EVAL_DIR = os.path.join(BASE_DIR, "resultados_evaluador")
    RESULTADOS_DIR = os.path.join(BASE_DIR, "Resultados")
    CASOS_DIR = os.path.join(BASE_DIR, "Set_casos")
    INFORMES_DIR = os.path.join(BASE_DIR, "informes_evaluador")

    # Archivo maestro de preguntas
    ruta_questions = os.path.join(CASOS_DIR, "Golden_questions_planeador.json")

    # Crear carpeta de informes si no existe
    if not os.path.exists(INFORMES_DIR):
        os.makedirs(INFORMES_DIR)
        print(f"📁 Carpeta creada: {INFORMES_DIR}")

    # Buscar archivos de evaluación JSON
    archivos_eval = [f for f in os.listdir(EVAL_DIR) if f.startswith("evaluacion_") and f.endswith(".json")]

    if not archivos_eval:
        print("⚠️ No se encontraron archivos de evaluación en 'resultados_evaluador'.")
        return

    for archivo in archivos_eval:
        modelo_nombre = archivo.replace("evaluacion_", "").replace(".json", "")
        ruta_eval = os.path.join(EVAL_DIR, archivo)
        
        # Buscar el archivo de respuestas correspondiente
        archivo_respuestas = f"respuestas_{modelo_nombre}.json"
        ruta_results = os.path.join(RESULTADOS_DIR, archivo_respuestas)
        
        if not os.path.exists(ruta_results):
            print(f"⚠️ No se encontró el archivo de respuestas para {modelo_nombre} en {ruta_results}")
            continue

        ruta_excel = os.path.join(INFORMES_DIR, f"Informe_Evaluacion_{modelo_nombre}.xlsx")
        
        try:
            generar_excel_desde_json(ruta_eval, ruta_results, ruta_questions, ruta_excel)
        except Exception as e:
            print(f"❌ Error procesando {modelo_nombre}: {str(e)}")

    print("\n✅ Proceso de generación de informes finalizado.")

if __name__ == "__main__":
    main()
