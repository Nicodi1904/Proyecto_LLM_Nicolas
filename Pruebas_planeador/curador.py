import json
import os

path = r'C:\Users\Owner\Documents\2025-2\Tesis-MAS-LLM\Pruebas_planeador\Pruebas_planeador\Calidad_planeador_local.ipynb'
out_path = r'C:\Users\Owner\Documents\2025-2\Tesis-MAS-LLM\Pruebas_planeador\Pruebas_planeador\Calidad_planeador_local_windows.ipynb'

with open(path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

new_cells = []
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = cell['source']
        text = ''.join(source)
        
        # Omitir celdas de instalacion linux / colab
        if '!apt-get' in text or 'curl -fsSL' in text or 'subprocess.Popen' in text:
            # Skip this cell
            continue
            
        # Modificar celda de drive/rutas
        if 'drive.mount' in text or 'BASE_PATH' in text:
            new_source = []
            for line in source:
                if 'drive.mount' in line or 'from google.colab import drive' in line:
                    continue
                elif 'BASE_PATH =' in line:
                    new_source.append("BASE_PATH = '.'\n")
                else:
                    new_source.append(line)
            cell['source'] = new_source
            
    new_cells.append(cell)

nb['cells'] = new_cells

with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=2, ensure_ascii=False)
    
print("Notebook modificado con éxito.")
