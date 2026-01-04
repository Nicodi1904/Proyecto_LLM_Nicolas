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
env_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(env_path)

# Configuración de LLM (Default: Llama 3.1)
llama31 = get_model("llama3.1:latest")
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
                                        value='llama3.1:latest',
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

@app.callback(
    [Output('interpretador-output', 'children'),
     Output('plan-output', 'children'),
     Output('gerente-output', 'children'),
     Output('charts-container', 'children'),
     Output('execution-raw-output', 'children')],
    [Input('run-button', 'n_clicks')],
    [State('input-prompt', 'value'),
     State('model-selector', 'value'),
     State('time-mad-start', 'value'), State('time-mad-end', 'value'),
     State('time-man-start', 'value'), State('time-man-end', 'value'),
     State('time-tar-start', 'value'), State('time-tar-end', 'value'),
     State('time-noc-start', 'value'), State('time-noc-end', 'value')],
    prevent_initial_call=True
)
def run_full_pipeline(n_clicks, prompt, selected_model, 
                      mad_s, mad_e, man_s, man_e, tar_s, tar_e, noc_s, noc_e):
    if not n_clicks or not prompt:
        return dash.no_update

    # Re-configurar el modelo elegido de forma segura para hilos usando context
    chosen_lm = get_model(selected_model)
    
    with dspy.context(lm=chosen_lm):
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
        print(f"DEBUG - Contexto Temporal Enviado: {json.dumps(temporal_context, indent=2)}")
        dispositivos_conocidos = ["Ventilador", "PC", "TV", "Total_Casa", "AC", "Lampara"]
        server_url = "http://localhost:8000/sse"

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

            # 4. Ejecución (MCP_C)
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                reporte_ejecucion = loop.run_until_complete(ejecutar_plan(server_url, validas))
                loop.close()
            except Exception as e:
                reporte_ejecucion = {}
                print(f"Error en ejecución MCP: {e}")

            reporte_consolidado = consolidar_reportes(reporte_ejecucion, invalidas)

            # 5. Visualización
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

            # 6. Gerente
            res_gerente = dspy.ChainOfThought(Gerente)(solicitudes_categorizadas=solicitudes, reporte_acciones=reporte_consolidado)
            gerente_display = res_gerente.respuesta_usuario

            return interp_display, plan_display, gerente_display, charts_display, exec_raw

        except Exception as e:
            error_msg = html.Div(f"Ocurrió un error en el pipeline: {str(e)}", className="text-danger")
            return error_msg, error_msg, f"Error: {str(e)}", dcc.Graph(figure=px.scatter(title="Error en la consulta")), html.Div()

if __name__ == '__main__':
    app.run(debug=True, port=8050)
