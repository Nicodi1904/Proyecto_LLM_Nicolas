import json
from datetime import datetime

with open('casos_evaluacion_planeador.json', 'r', encoding='utf-8') as f:
    casos = json.load(f)

with open('respuestas_planeador.json', 'r', encoding='utf-8') as f:
    respuestas = json.load(f)

with open('Agente_energetico/Sistema_entrada/Planeador/system_summary.json', 'r', encoding='utf-8') as f:
    summary = json.load(f)

tools_schema = {t['name']: t for t in summary['servers'][0]['tools']}
context = datetime.strptime('2026-03-30T10:25', '%Y-%m-%dT%H:%M')
errores = []

for nivel, level_data in respuestas.items():
    for caso_id, datos in level_data.items():
        if caso_id not in casos[nivel]:
            errores.append(f'Caso {caso_id} existe en respuestas pero no en casos.')
            continue
            
        solicitud = list(casos[nivel][caso_id]['solicitudes_categorizadas'].values())[0]['solicitud']
        plan = datos.get('plan_acciones', [])
            
        for accion in plan:
            tool_name = accion.get('tool')
            if tool_name not in tools_schema:
                errores.append(f'{caso_id}: Tool {tool_name} no existe en summary.')
                
            inputs = accion.get('inputs', {})
            
            fechas_pares = []
            if 'fecha_inicio' in inputs:
                fechas_pares.append((inputs.get('fecha_inicio'), inputs.get('fecha_fin'), 'directo'))
            if 'objetivo_a' in inputs:
                fechas_pares.append((inputs['objetivo_a'].get('fecha_inicio'), inputs['objetivo_a'].get('fecha_fin'), 'obj_a'))
            if 'objetivo_b' in inputs:
                fechas_pares.append((inputs['objetivo_b'].get('fecha_inicio'), inputs['objetivo_b'].get('fecha_fin'), 'obj_b'))
                
            for ini, fin, tag in fechas_pares:
                if not ini or not fin:
                    continue
                try:
                    dt_ini = datetime.strptime(ini, '%Y-%m-%dT%H:%M')
                    dt_fin = datetime.strptime(fin, '%Y-%m-%dT%H:%M')
                    
                    if dt_ini >= dt_fin:
                        errores.append(f'{caso_id}: f_inicio >= f_fin en {tag} ({ini} vs {fin})')
                        
                    if dt_fin > context and tool_name != 'accion_imposible':
                        errores.append(f'{caso_id}: viaje al futuro no bloqueado en {tag}. ({fin})')
                except Exception as e:
                    errores.append(f'{caso_id}: Formato fecha invalido {ini} o {fin}')

if not errores:
    print('✅ Analisis 360 grados completado: NINGUN ERROR.')
else:
    print('Se encontraron errores:')
    for e in set(errores):
        print(e)
