import json
ruta = r"c:\Users\Owner\Documents\2025-2\Tesis-MAS-LLM\Agente_energetico\Sistema_entrada\Planeador\system_summary.json"

with open(ruta, 'r', encoding='utf-8') as f:
    summary = json.load(f)

for s in summary.get("servers", []):
    for t in s.get("tools", []):
        meta = t.get("meta", {})
        out_schema = meta.get("output_schema", {})
        claves = list(out_schema.get("properties", {}).keys()) if isinstance(out_schema, dict) else []
        t["meta"]["output_schema"] = f"Diccionario con las claves: {claves}" if claves else "No especificado"
        if "fastmcp" in t["meta"]:
            del t["meta"]["fastmcp"]

with open(ruta, 'w', encoding='utf-8') as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

print("✅ system_summary.json compactado estáticamente con éxito.")
