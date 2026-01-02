import dash
from dash import dcc, html, Input, Output, State, dash_table
import dash_bootstrap_components as dbc
import plotly.express as px
import pandas as pd
import dspy
import os
import json
import asyncio
from dotenv import load_dotenv

# Importar agentes y verificadores
try:
    from Test_Interpretador import Interpretador, escenarios_entrada
    from Test_planeador import Planeador
    from test_worker_verificador import verificar_completo, filtrar_acciones
    from MCP_C import ejecutar_plan, consolidar_reportes
    from Test_gerente import Gerente
    from MCP_C_obtener_summary import system_summary
except ImportError:
    import sys
    sys.path.append(os.path.dirname(__file__))
    from Test_Interpretador import Interpretador, escenarios_entrada
    from Test_planeador import Planeador
    from test_worker_verificador import verificar_completo, filtrar_acciones
    from MCP_C import ejecutar_plan, consolidar_reportes
    from Test_gerente import Gerente
    from MCP_C_obtener_summary import system_summary

# Cargar API Keys
env_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(env_path)

# Configuración de LLM (Default: Llama 3.1)
llama31 = dspy.LM('ollama_chat/llama3.1:latest', api_base='http://localhost:11434', api_key='')
dspy.configure(lm=llama31)

# Inicializar App
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])
app.title = "Agente Energético"

# Estilos
COLORS = {
    'bg': '#f8f9fa',
    'header': '#2c3e50',
    'accent': '#18bc9c',
    'valid': '#d4edda',
    'invalid': '#f8d7da'
}

# --- Layout ---
app.layout = dbc.Container([
    # Header
    dbc.Row([
        dbc.Col(html.H1("Agente Energético", className="text-center text-white p-4", 
                        style={'backgroundColor': COLORS['header'], 'borderRadius': '0 0 15px 15px'}), width=12)
    ], className="mb-4"),

    # Input Section
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5("¿En qué te puedo ayudar hoy?", className="card-title"),
                    dcc.Textarea(
                        id='input-prompt',
                        placeholder='Ej: Dime el consumo del ventilador ayer por la mañana y compáralo con el PC...',
                        style={'width': '100%', 'height': '100px', 'borderRadius': '10px', 'padding': '10px'},
                        value='Consumo ventilador ayer por la tarde'
                    ),
                    html.Br(),
                    dbc.Button("Ejecutar Pipeline AI", id='run-button', color="primary", className="mt-2 w-100", size="lg")
                ])
            ], style={'borderRadius': '15px', 'boxShadow': '0 4px 6px rgba(0,0,0,0.1)'})
        ], width=12)
    ], className="mb-4"),

    # Tabs Section
    dcc.Loading(
        id="loading-pipeline",
        type="circle",
        children=dbc.Row([
            dbc.Col([
                dbc.Tabs([
                    # Tab 1: Interpretación
                    dbc.Tab(label="1. Interpretación", children=[
                        dbc.Card([
                            dbc.CardBody([
                                html.P("Análisis de solicitudes identificadas por el Interpretador.", className="text-muted"),
                                html.Div(id='interpretador-output')
                            ])
                        ], className="mt-3")
                    ], tab_id="tab-interpret"),

                    # Tab 2: Planeación & Verificación
                    dbc.Tab(label="2. Plan & Verificación", children=[
                        dbc.Card([
                            dbc.CardBody([
                                html.P("Acciones propuestas y su estado de validación.", className="text-muted"),
                                html.Div(id='plan-output')
                            ])
                        ], className="mt-3")
                    ], tab_id="tab-plan"),

                    # Tab 3: Datos & Ejecución
                    dbc.Tab(label="3. Visualización", children=[
                        dbc.Card([
                            dbc.CardBody([
                                html.P("Resultados numéricos y gráficos del consumo.", className="text-muted"),
                                html.Div(id='charts-container'),
                                html.Div(id='execution-raw-output')
                            ])
                        ], className="mt-3")
                    ], tab_id="tab-vis"),

                    # Tab 4: Respuesta Final (Gerente)
                    dbc.Tab(label="4. Respuesta Final", children=[
                        dbc.Card([
                            dbc.CardBody([
                                html.H4("Informe del Gerente", className="card-title text-success"),
                                dcc.Markdown(id='gerente-output', style={'fontSize': '1.1rem'})
                            ])
                        ], className="mt-3")
                    ], tab_id="tab-gerente"),

                ], id="tabs-agentes", active_tab="tab-gerente")
            ], width=12)
        ])
    ),

    # Footer
    dbc.Row([
        dbc.Col(html.P("Agente Energético - MAS Pipeline Visualization", className="text-center text-muted small mt-5"), width=12)
    ])

], fluid=True, style={'backgroundColor': COLORS['bg'], 'minHeight': '100vh'})

# --- Callbacks ---

@app.callback(
    [Output('interpretador-output', 'children'),
     Output('plan-output', 'children'),
     Output('gerente-output', 'children'),
     Output('charts-container', 'children'),
     Output('execution-raw-output', 'children')],
    [Input('run-button', 'n_clicks')],
    [State('input-prompt', 'value')],
    prevent_initial_call=True
)
def run_full_pipeline(n_clicks, prompt):
    if not n_clicks or not prompt:
        return dash.no_update

    # Contexto Temporal Simulado (Igual que en Sistema_LLM_entrada)
    temporal_context = {
        "referencia_actual": "2024-12-30T13:00:00",
        "zona_horaria": "America/Bogota",
        "rangos_horarios": {
            "madrugada": {"inicio": "00:00", "fin": "05:59"},
            "mañana": {"inicio": "06:00", "fin": "11:59"},
            "tarde": {"inicio": "12:00", "fin": "17:59"},
            "noche": {"inicio": "18:00", "fin": "23:59"}
        }
    }
    dispositivos_conocidos = ["Ventilador", "PC", "TV", "Total_Casa", "AC", "Lampara"]
    server_url = "http://localhost:8000/sse" # Ajustar segun servidor real

    try:
        # 1. Interpretación
        res_interp = dspy.Predict(Interpretador)(prompt_usuario=prompt, escenarios_entrada=escenarios_entrada)
        solicitudes = res_interp.solicitudes_categorizadas

        interp_display = html.Ul([
            html.Li([
                html.B(f"{k}: "), html.Span(v['solicitud']), 
                html.Span(f" [{v['escenario']}]", className="badge bg-info ms-2")
            ]) for k,v in solicitudes.items()
        ])

        # 2. Planificación
        res_plan = dspy.ChainOfThought(Planeador)(
            solicitudes_categorizadas=solicitudes, 
            system_summary=system_summary, 
            temporal_context=temporal_context
        )
        plan_acciones = res_plan.plan_acciones
        print(f"DEBUG - Plan generado: {json.dumps(plan_acciones, indent=2)}")

        # 3. Verificación
        reporte_verif = verificar_completo(plan_acciones, system_summary, dispositivos_conocidos, temporal_context)
        validas, invalidas = filtrar_acciones(reporte_verif, plan_acciones)

        plan_rows = []
        for accion in plan_acciones:
            id_acc = accion['id']
            # Buscar error si existe
            err = next((inv['error_verificacion'] for inv in invalidas if inv['id'] == id_acc), None)
            style = {'backgroundColor': COLORS['invalid']} if err else {'backgroundColor': COLORS['valid']}
            plan_rows.append(html.Tr([
                html.Td(id_acc),
                html.Td(accion['tool']),
                html.Td(accion['descripcion']),
                html.Td("RECHAZADA: " + err if err else "VALIDADA", 
                        style={'color': 'red' if err else 'green', 'fontWeight': 'bold'})
            ], style=style))

        plan_display = html.Table([
            html.Thead(html.Tr([html.Th("ID"), html.Th("Tool"), html.Th("Descripción"), html.Th("Estado")])),
            html.Tbody(plan_rows)
        ], className="table table-bordered")

        # 4. Ejecución (MCP_C)
        # Nota: ejecutar_plan es async, necesitamos ejecutarlo síncronamente aquí para el callback
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            reporte_ejecucion = loop.run_until_complete(ejecutar_plan(server_url, validas))
            loop.close()
        except Exception as e:
            reporte_ejecucion = {}
            print(f"Error en ejecución MCP: {e}")

        reporte_consolidado = consolidar_reportes(reporte_ejecucion, invalidas)
        print(f"DEBUG - Reporte Consolidado: {json.dumps(reporte_consolidado, indent=2)}")

        # 5. Visualización (Plotly)
        charts = []
        for sol_id, acciones_res in reporte_ejecucion.items():
            for res in acciones_res:
                if res['tool'] == 'obtener_consumo' and res['resultado'] and res['resultado'].get('status') == 'success':
                    datos = res['resultado'].get('datos', {})
                    granularidad = res['resultado'].get('granularidad', 'total')
                    descripcion = res['descripcion']
                    accion_id = res['accion_id']
                    
                    if not datos:
                        continue
                        
                    if granularidad == 'total':
                        # Datos: {"Ventilador": 10.5, "PC": 5.2}
                        df_plot = pd.DataFrame([{'Dispositivo': d, 'Consumo': v} for d, v in datos.items()])
                        fig = px.bar(df_plot, x='Dispositivo', y='Consumo', color='Dispositivo',
                                    title=f"[{accion_id}] {descripcion}", 
                                    template="plotly_white")
                    else:
                        # Datos: {"Device1": {"2024-11-14T18:00": 0.5, ...}}
                        rows = []
                        for disp, series in datos.items():
                            if isinstance(series, dict):
                                for ts, val in series.items():
                                    rows.append({'Timestamp': ts, 'Consumo': val, 'Dispositivo': disp})
                            else:
                                rows.append({'Timestamp': 'Total', 'Consumo': series, 'Dispositivo': disp})
                        
                        df_plot = pd.DataFrame(rows)
                        if not df_plot.empty:
                            fig = px.line(df_plot, x='Timestamp', y='Consumo', color='Dispositivo', markers=True,
                                         title=f"[{accion_id}] {descripcion}",
                                         template="plotly_white")
                        else:
                            continue
                            
                    charts.append(dbc.Card([
                        dbc.CardBody([
                            dcc.Graph(figure=fig)
                        ])
                    ], className="mb-3"))
        
        if not charts:
            charts_display = html.Div("Sin datos de consumo para graficar", className="text-muted text-center p-4")
        else:
            charts_display = html.Div(charts)

        exec_raw = html.Pre(json.dumps(reporte_consolidado, indent=2, ensure_ascii=False), 
                            style={'fontSize': '0.8rem', 'maxHeight': '300px', 'overflowY': 'auto'})

        # 6. Gerente
        res_gerente = dspy.ChainOfThought(Gerente)(
            solicitudes_categorizadas=solicitudes,
            reporte_acciones=reporte_consolidado
        )
        gerente_display = res_gerente.respuesta_usuario

        return interp_display, plan_display, gerente_display, charts_display, exec_raw

    except Exception as e:
        error_msg = html.Div(f"Ocurrió un error en el pipeline: {str(e)}", className="text-danger")
        return error_msg, error_msg, f"Error: {str(e)}", px.scatter(), html.Div()

if __name__ == '__main__':
    app.run(debug=True, port=8050)
