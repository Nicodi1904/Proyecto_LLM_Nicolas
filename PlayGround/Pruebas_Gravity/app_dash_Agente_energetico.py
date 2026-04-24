import dash
from dash import dcc, html, Input, Output, State, dash_table
import dash_bootstrap_components as dbc
import plotly.express as px
import pandas as pd
import dspy
import os
import json
import asyncio
# from dotenv import load_dotenv

# Importar agentes y verificadores
try:
    from Test_Interpretador import Interpretador, escenarios_entrada
    from Test_planeador import Planeador
    from test_worker_verificador import verificar_completo, filtrar_acciones
    from MCP_C import ejecutar_plan, consolidar_reportes
    from Test_gerente import Gerente
    from MCP_C_obtener_summary import system_summary
    from modelos_disponibles import get_model, listar_modelos_configurados
except ImportError:
    import sys
    sys.path.append(os.path.dirname(__file__))
    from Test_Interpretador import Interpretador, escenarios_entrada
    from Test_planeador import Planeador
    from test_worker_verificador import verificar_completo, filtrar_acciones
    from MCP_C import ejecutar_plan, consolidar_reportes
    from Test_gerente import Gerente
    from MCP_C_obtener_summary import system_summary
    from modelos_disponibles import get_model, listar_modelos_configurados

from datetime import datetime

# Cargar API Keys
# Cargar API Keys
# env_path = os.path.join(os.path.dirname(__file__), '.env')
# load_dotenv(env_path)

# Configuración de LLM (Default: llama31, pero puede ser deepseek)
# NOTE: La configuración inicial es solo un placeholder, la real ocurre en los callbacks
try:
    default_lm = get_model("deepseek-r1:8b")
except:
    default_lm = dspy.LM('ollama_chat/llama3.1:latest', api_base="http://localhost:11434", api_key='')

dspy.configure(lm=default_lm)

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
        dbc.Col(
            html.Div([
                html.H1("Agente Energético", style={'margin': '0'}),
                html.Div(id='live-clock', style={'fontSize': '1.2rem', 'fontWeight': 'bold'})
            ], style={
                'display': 'flex', 
                'justifyContent': 'space-between', 
                'alignItems': 'center',
                'backgroundColor': COLORS['header'], 
                'borderRadius': '0 0 15px 15px',
                'padding': '20px 40px',
                'color': 'white'
            }), 
            width=12
        )
    ], className="mb-4"),

    dcc.Interval(
        id='interval-component',
        interval=60*1000, # 1 minuto
        n_intervals=0
    ),

    # Input Section
    dbc.Row([
        # Columna Izquierda: Petición
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.Div([
                        html.H5("¿En qué te puedo ayudar hoy?", className="card-title", style={'margin': '0'}),
                        dbc.Button([
                            html.Span("⚙️", style={'fontSize': '1.8rem', 'marginRight': '5px'}),
                            html.Span("Ajustes", style={'fontSize': '1.0rem'})
                        ], id="toggle-config", color="link", className="p-0 d-flex align-items-center", style={'textDecoration': 'none'})
                    ], style={'display': 'flex', 'justifyContent': 'space-between', 'alignItems': 'center', 'marginBottom': '15px'}),
                    
                    dcc.Textarea(
                        id='input-prompt',
                        placeholder='¿Cuánta energía consumió la nevera ayer entre 6 pm y 10 pm?',
                        style={'width': '100%', 'height': '120px', 'borderRadius': '10px', 'padding': '15px', 'borderColor': '#ddd'},
                        value=''
                    ),
                    html.P("💡 Hint: Puedes consultar consumo por dispositivo, rango temporal o hacer comparaciones.", 
                           className="text-muted small mt-2", style={'fontStyle': 'italic'}),
                    html.Br(),
                    dbc.Button("Realizar consulta", id='run-button', color="primary", className="mt-2 w-100", size="lg")
                ])
            ], style={'borderRadius': '15px', 'boxShadow': '0 4px 6px rgba(0,0,0,0.1)'})
        ], width=9, md=9, sm=12),

        # Columna Colapsable: Ajustes con Pestañas
        dbc.Col([
            dbc.Collapse(
                dbc.Card([
                    dbc.CardBody([
                        dbc.Tabs([
                            # Pestaña 1: Modelo
                            dbc.Tab(label="Ajustes de Modelo", children=[
                                html.Div([
                                    html.Label("Modelo de lenguaje:", style={'fontSize': '0.95rem', 'fontWeight': 'bold', 'marginTop': '15px'}),
                                    dcc.Dropdown(
                                        id='model-selector',
                                        options=listar_modelos_configurados(),
                                        value='deepseek-r1:8b',
                                        clearable=False,
                                        style={'marginBottom': '20px', 'fontSize': '0.9rem'}
                                    ),
                                    html.Label("Optimizadores:", style={'fontSize': '0.95rem', 'fontWeight': 'bold'}),
                                    dbc.Checklist(
                                        options=[{"label": "Few Shots Examples", "value": 1}],
                                        value=[],
                                        id="few-shots-toggle",
                                        switch=True,
                                        style={'fontSize': '0.9rem'}
                                    ),
                                    html.P("Ayuda a mejorar respuestas, pero aumenta consumo de tokens (recomendado en modelos pequeños).",
                                           className="text-muted mt-2", style={'fontSize': '0.8rem', 'lineHeight': '1.3'})
                                ], style={'padding': '10px'})
                            ], tab_id="tab-settings-model"),

                            # Pestaña 2: Preferencias Temporales
                            dbc.Tab(label="Preferencias", children=[
                                html.Div([
                                    html.Label("Configuración de Rangos Horarios:", style={'fontSize': '0.95rem', 'fontWeight': 'bold', 'marginTop': '15px', 'display': 'block'}),
                                    
                                    # Filas para cada rango
                                    dbc.Row([
                                        dbc.Col(html.Small("Madrugada:"), width=5),
                                        dbc.Col(dbc.Input(id='time-mad-start', value='00:00', size="sm"), width=3),
                                        dbc.Col(dbc.Input(id='time-mad-end', value='05:59', size="sm"), width=3),
                                    ], className="mb-2 align-items-center"),
                                    
                                    dbc.Row([
                                        dbc.Col(html.Small("Mañana:"), width=5),
                                        dbc.Col(dbc.Input(id='time-man-start', value='06:00', size="sm"), width=3),
                                        dbc.Col(dbc.Input(id='time-man-end', value='11:59', size="sm"), width=3),
                                    ], className="mb-2 align-items-center"),

                                    dbc.Row([
                                        dbc.Col(html.Small("Tarde:"), width=5),
                                        dbc.Col(dbc.Input(id='time-tar-start', value='12:00', size="sm"), width=3),
                                        dbc.Col(dbc.Input(id='time-tar-end', value='17:59', size="sm"), width=3),
                                    ], className="mb-2 align-items-center"),

                                    dbc.Row([
                                        dbc.Col(html.Small("Noche:"), width=5),
                                        dbc.Col(dbc.Input(id='time-noc-start', value='18:00', size="sm"), width=3),
                                        dbc.Col(dbc.Input(id='time-noc-end', value='23:59', size="sm"), width=3),
                                    ], className="mb-2 align-items-center"),
                                    
                                    html.P("Formato: HH:MM (24h)", className="text-muted x-small mt-2", style={'fontSize': '0.75rem'})
                                ], style={'padding': '10px'})
                            ], tab_id="tab-settings-pref")
                        ], id="settings-tabs", active_tab="tab-settings-model")
                    ], style={'padding': '15px'})
                ], style={'borderRadius': '15px', 'boxShadow': '0 4px 6px rgba(0,0,0,0.1)', 'minHeight': '320px'}),
                id="collapse-config",
                is_open=False,
            )
        ], width=3, md=3, sm=12)
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
                                dcc.Loading(id="loading-interp", type="dot", children=html.Div(id='interpretador-output'))
                            ])
                        ], className="mt-3")
                    ], tab_id="tab-interpret"),

                    # Tab 2: Planeación & Verificación
                    dbc.Tab(label="2. Plan & Verificación", children=[
                        dbc.Card([
                            dbc.CardBody([
                                html.P("Acciones propuestas y su estado de validación.", className="text-muted"),
                                dcc.Loading(id="loading-plan", type="dot", children=html.Div(id='plan-output'))
                            ])
                        ], className="mt-3")
                    ], tab_id="tab-plan"),

                    # Tab 3: Datos & Ejecución
                    dbc.Tab(label="3. Visualización", children=[
                        dbc.Card([
                            dbc.CardBody([
                                html.P("Resultados numéricos y gráficos del consumo.", className="text-muted"),
                                dcc.Loading(id="loading-vis", type="dot", children=html.Div(id='charts-container')),
                                html.Div(id='execution-raw-output')
                            ])
                        ], className="mt-3")
                    ], tab_id="tab-vis"),

                    # Tab 4: Respuesta Final (Gerente)
                    dbc.Tab(label="4. Respuesta Final", children=[
                        dbc.Card([
                            dbc.CardBody([
                                html.H4("Informe del Gerente", className="card-title text-success"),
                                dcc.Loading(id="loading-gerente", type="dot", children=dcc.Markdown(id='gerente-output', style={'fontSize': '1.1rem'}))
                            ])
                        ], className="mt-3")
                    ], tab_id="tab-gerente"),

                ], id="tabs-agentes", active_tab="tab-interpret")
            ], width=12)
        ])
    ),

    # Footer
    dbc.Row([
        dbc.Col(html.P("Agente Energético - MAS Pipeline Visualization", className="text-center text-muted small mt-5"), width=12)
    ]),

    # Stores for progressive updates
    dcc.Store(id='store-context'),
    dcc.Store(id='store-interprete'),
    dcc.Store(id='store-plan'),
    dcc.Store(id='store-execution'),

], fluid=True, style={'backgroundColor': COLORS['bg'], 'minHeight': '100vh'})

# --- Callbacks ---

@app.callback(
    Output("collapse-config", "is_open"),
    [Input("toggle-config", "n_clicks")],
    [State("collapse-config", "is_open")],
)
def toggle_setting_panel(n, is_open):
    if n:
        return not is_open
    return is_open

@app.callback(
    Output('live-clock', 'children'),
    [Input('interval-component', 'n_intervals')]
)
def update_metrics(n):
    return datetime.now().strftime("%H:%M")

# --- Chained Callbacks for Progressive Updates ---

# 1. Initialization: Set Context and Trigger Interpretation
@app.callback(
    [Output('store-context', 'data'),
     Output('tabs-agentes', 'active_tab')],
    [Input('run-button', 'n_clicks')],
    [State('input-prompt', 'value'),
     State('model-selector', 'value'),
     State('time-mad-start', 'value'), State('time-mad-end', 'value'),
     State('time-man-start', 'value'), State('time-man-end', 'value'),
     State('time-tar-start', 'value'), State('time-tar-end', 'value'),
     State('time-noc-start', 'value'), State('time-noc-end', 'value')],
    prevent_initial_call=True
)
def step_1_init(n_clicks, prompt, selected_model, 
               mad_s, mad_e, man_s, man_e, tar_s, tar_e, noc_s, noc_e):
    if not n_clicks or not prompt:
        return dash.no_update, dash.no_update
    
    temporal_context = {
        "referencia_actual": datetime.now().isoformat(),
        "zona_horaria": "America/Bogota",
        "rangos_horarios": {
            "madrugada": {"inicio": mad_s, "fin": mad_e},
            "mañana": {"inicio": man_s, "fin": man_e},
            "tarde": {"inicio": tar_s, "fin": tar_e},
            "noche": {"inicio": noc_s, "fin": noc_e}
        }
    }
    
    context_data = {
        "prompt": prompt,
        "model": selected_model,
        "temporal_context": temporal_context
    }
    
    # Return context and switch to first tab
    return context_data, "tab-interpret"

# 2. Interpretation Agent
@app.callback(
    [Output('interpretador-output', 'children'),
     Output('store-interprete', 'data')],
    [Input('store-context', 'data')],
    prevent_initial_call=True
)
def step_2_interpret(context_data):
    if not context_data:
        return dash.no_update, dash.no_update
    
    prompt = context_data['prompt']
    selected_model = context_data['model']
    
    # Configure Model (Safe context)
    chosen_lm = get_model(selected_model)
    
    try:
        with dspy.context(lm=chosen_lm):
            res_interp = dspy.Predict(Interpretador)(prompt_usuario=prompt, escenarios_entrada=escenarios_entrada)
            solicitudes = res_interp.solicitudes_categorizadas
            
            # Validation: Ensure solicitudes is a valid dict
            if not isinstance(solicitudes, dict):
                print(f"DEBUG - Error: Interpretador devolvió {type(solicitudes)}: {solicitudes}")
                return html.Div(f"Error: El modelo no generó solicitudes válidas. Respuesta cruda: {solicitudes}", className="text-danger"), None

            interp_display = html.Ul([
                html.Li([
                    html.B(f"{k}: "), html.Span(v['solicitud']), 
                    html.Span(f" [{v['escenario']}]", className="badge bg-info ms-2")
                ]) for k,v in solicitudes.items()
            ])
            
            return interp_display, solicitudes
    except Exception as e:
        return html.Div(f"Error en Interpretación: {str(e)}", className="text-danger"), None

# 3. Planner & Verifier Agent
@app.callback(
    [Output('plan-output', 'children'),
     Output('store-plan', 'data'),
     Output('tabs-agentes', 'active_tab', allow_duplicate=True)],
    [Input('store-interprete', 'data')],
    [State('store-context', 'data')],
    prevent_initial_call=True
)
def step_3_plan(solicitudes, context_data):
    if not solicitudes or not context_data:
        return dash.no_update, dash.no_update, dash.no_update
    
    selected_model = context_data['model']
    temporal_context = context_data['temporal_context']
    
    # Configure Model (Safe context)
    chosen_lm = get_model(selected_model)
    
    with dspy.context(lm=chosen_lm):
        try:
            print(f"DEBUG - System Summary enviado al Planeador: {json.dumps(system_summary, indent=2)}")
            res_plan = dspy.ChainOfThought(Planeador)(
                solicitudes_categorizadas=solicitudes, 
                system_summary=system_summary, 
                temporal_context=temporal_context
            )
            plan_acciones = res_plan.plan_acciones
            
            # Fix: Handle NoneType if model fails to generate list
            if plan_acciones is None:
                print("DEBUG - El modelo no generó plan_acciones (None). Asignando lista vacía.")
                print(f"DEBUG - Respuesta cruda del modelo (res_plan): {res_plan}")
                plan_acciones = []

            
            # Verificación
            dispositivos_conocidos = ["Ventilador", "PC", "TV", "Total_Casa", "AC", "Lampara"]
            reporte_verif = verificar_completo(plan_acciones, system_summary, dispositivos_conocidos, temporal_context)
            validas, invalidas = filtrar_acciones(reporte_verif, plan_acciones)
            
            plan_rows = []
            for accion in plan_acciones:
                id_acc = accion['id']
                err = next((inv['error_verificacion'] for inv in invalidas if inv['id'] == id_acc), None)
                style = {'backgroundColor': COLORS['invalid']} if err else {'backgroundColor': COLORS['valid']}
                plan_rows.append(html.Tr([
                    html.Td(id_acc), html.Td(accion['tool']), html.Td(accion['descripcion']),
                    html.Td("RECHAZADA: " + err if err else "VALIDADA", 
                            style={'color': 'red' if err else 'green', 'fontWeight': 'bold'})
                ], style=style))

            plan_display = html.Table([
                html.Thead(html.Tr([html.Th("ID"), html.Th("Tool"), html.Th("Descripción"), html.Th("Estado")])),
                html.Tbody(plan_rows)
            ], className="table table-bordered")
            
            plan_data = {
                "validas": validas,
                "invalidas": invalidas,
                "plan_acciones": plan_acciones
            }
            
            return plan_display, plan_data, "tab-plan"
            
        except Exception as e:
            return html.Div(f"Error en Planificación: {str(e)}", className="text-danger"), None, dash.no_update

# 4. Execution & Visualization
@app.callback(
    [Output('charts-container', 'children'),
     Output('execution-raw-output', 'children'),
     Output('store-execution', 'data'),
     Output('tabs-agentes', 'active_tab', allow_duplicate=True)],
    [Input('store-plan', 'data')],
    prevent_initial_call=True
)
def step_4_execute(plan_data):
    if not plan_data:
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update
    
    validas = plan_data['validas']
    invalidas = plan_data['invalidas']
    server_url = "http://localhost:8000/sse"

    try:
        # Run async execution in sync callback
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            reporte_ejecucion = loop.run_until_complete(ejecutar_plan(server_url, validas))
        finally:
            loop.close()
            
        reporte_consolidado = consolidar_reportes(reporte_ejecucion, invalidas)

        # Visualization
        charts = []
        for sol_id, acciones_res in reporte_ejecucion.items():
            for res in acciones_res:
                if res['tool'] == 'obtener_consumo' and res['resultado'] and res['resultado'].get('status') == 'success':
                    datos = res['resultado'].get('datos', {})
                    granularidad = res['resultado'].get('granularidad', 'total')
                    descripcion = res['descripcion']
                    accion_id = res['accion_id']
                    if not datos: continue
                    if granularidad == 'total':
                        df_plot = pd.DataFrame([{'Dispositivo': d, 'Consumo': v} for d, v in datos.items()])
                        fig = px.bar(df_plot, x='Dispositivo', y='Consumo', color='Dispositivo', title=f"[{accion_id}] {descripcion}", template="plotly_white")
                    else:
                        rows = []
                        for disp, series in datos.items():
                            if isinstance(series, dict):
                                for ts, val in series.items(): rows.append({'Timestamp': ts, 'Consumo': val, 'Dispositivo': disp})
                            else: rows.append({'Timestamp': 'Total', 'Consumo': series, 'Dispositivo': disp})
                        df_plot = pd.DataFrame(rows)
                        if not df_plot.empty:
                            fig = px.line(df_plot, x='Timestamp', y='Consumo', color='Dispositivo', markers=True, title=f"[{accion_id}] {descripcion}", template="plotly_white")
                        else: continue
                    charts.append(dbc.Card([dbc.CardBody([dcc.Graph(figure=fig)])], className="mb-3"))
        
        charts_display = html.Div(charts) if charts else html.Div("Sin datos de consumo para graficar", className="text-muted text-center p-4")
        exec_raw = html.Pre(json.dumps(reporte_consolidado, indent=2, ensure_ascii=False), style={'fontSize': '0.8rem', 'maxHeight': '300px', 'overflowY': 'auto'})
        
        return charts_display, exec_raw, reporte_consolidado, "tab-vis"
        
    except Exception as e:
        err_msg = html.Div(f"Error en Ejecución: {str(e)}", className="text-danger")
        return err_msg, err_msg, None, dash.no_update

# 5. Manager Agent (Final Response)
@app.callback(
    [Output('gerente-output', 'children'),
     Output('tabs-agentes', 'active_tab', allow_duplicate=True)],
    [Input('store-execution', 'data')],
    [State('store-interprete', 'data'),
     State('store-context', 'data')],
    prevent_initial_call=True
)
def step_5_gerente(reporte_consolidado, solicitudes, context_data):
    if not reporte_consolidado or not solicitudes or not context_data:
        return dash.no_update, dash.no_update
    
    selected_model = context_data['model']
    chosen_lm = get_model(selected_model)
    
    try:
        with dspy.context(lm=chosen_lm):
            res_gerente = dspy.ChainOfThought(Gerente)(solicitudes_categorizadas=solicitudes, reporte_acciones=reporte_consolidado)
            
            # Fix: Handle NoneType response
            if hasattr(res_gerente, 'respuesta_usuario') and res_gerente.respuesta_usuario:
                gerente_display = res_gerente.respuesta_usuario
            else:
                gerente_display = "Lo siento, no pude generar un resumen final detallado debido a un error en el modelo."

            
        return gerente_display, "tab-gerente"
    except Exception as e:
        return html.Div(f"Error en Gerente: {str(e)}", className="text-danger"), dash.no_update

if __name__ == '__main__':
    app.run(debug=True, port=8050)
