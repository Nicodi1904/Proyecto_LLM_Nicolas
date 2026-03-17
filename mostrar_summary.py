import json
import os

ruta = r"c:\Users\Owner\Documents\2025-2\Tesis-MAS-LLM\Agente_energetico\Sistema_entrada\Planeador\system_summary.json"
with open(ruta, 'r', encoding='utf-8') as f:
    system_summary = json.load(f)

summary_recortado = {"servers": []}
for s in system_summary.get("servers", []):
    servidor = {"server_id": s.get("server_id"), "tools": []}
    for t in s.get("tools", []):
        meta = t.get("meta", {})
        out_schema = meta.get("output_schema", {})
        claves_salida = list(out_schema.get("properties", {}).keys()) if isinstance(out_schema, dict) else []
        out_resumen = f"Diccionario con las claves: {claves_salida}" if claves_salida else "No especificado"
        herramienta = {
            "name": t.get("name"),
            "meta": {
                "proposito": meta.get("proposito"),
                "usar_si": meta.get("usar_si"),
                "input_schema": meta.get("input_schema"),
                "output_keys": out_resumen
            }
        }
        servidor["tools"].append(herramienta)
    summary_recortado["servers"].append(servidor)

print(json.dumps(summary_recortado, indent=2, ensure_ascii=False))
