from fastmcp import FastMCP
import pandas as pd
import numpy as np
from datetime import datetime
import sys
import os

# Agregar directorio padre al path para importar módulos compartidos
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from PlayGround.cargar_CSV import cargar_dataset_sinselejo

# Initialize FastMCP server
mcp = FastMCP("MCP_Server_Gravity")

# -------------------------
# Cargar dataset global
# -------------------------
# Construir ruta absoluta al CSV
csv_path = os.path.join(os.path.dirname(__file__), "Energy Consumption in KWh of a Typical House Sincelejo Colombia.csv")
DATASET = cargar_dataset_sinselejo(csv_path)
# Asegurar que TimeStamp es datetime
DATASET['TimeStamp'] = pd.to_datetime(DATASET['TimeStamp'])

""" 
#-------------------------
# Herramienta auxiliar
#-------------------------

@mcp.tool(
    meta={
        "proposito": (
            "Calcula rangos de fechas precisos en formato ISO 8601 a partir de expresiones "
            "temporales relativas y una fecha de referencia."
        ),
        "usar_si": [
            "El usuario usa términos como 'ayer', 'hoy', 'semana pasada', 'mes pasado'",
            "Se requiere convertir periodos como 'ayer por la noche' a timestamps exactos",
            "El LLM necesita ayuda para determinar fechas de inicio y fin"
        ],
        "input_schema": {
            "type": "object",
            "properties": {
                "expresion": {
                    "type": "string",
                    "description": "Expresión temporal relativa (ej: 'ayer noche', 'semana pasada')."
                },
                "fecha_referencia": {
                    "type": "string",
                    "description": "Fecha actual del sistema en formato ISO (ej: '2024-11-15T10:00')."
                },
                "rangos_horarios": {
                    "type": "object",
                    "description": "Definición opcional de horas para madrugada, mañana, tarde, noche.",
                    "default": {
                        "madrugada": {"inicio": "00:00", "fin": "05:59"},
                        "mañana": {"inicio": "06:00", "fin": "11:59"},
                        "tarde": {"inicio": "12:00", "fin": "17:59"},
                        "noche": {"inicio": "18:00", "fin": "23:59"}
                    }
                }
            },
            "required": ["expresion", "fecha_referencia"]
        },
        "output_schema": {
            "type": "object",
            "properties": {
                "status": {"type": "string"},
                "fecha_inicio": {"type": "string"},
                "fecha_fin": {"type": "string"},
                "mensaje": {"type": "string"}
            }
        }
    }
)
def determinar_rango_temporal(expresion: str, fecha_referencia: str, rangos_horarios: dict = None) -> dict:
    from datetime import timedelta
    import re
    
    if rangos_horarios is None:
        rangos_horarios = {
            "madrugada": {"inicio": "00:00", "fin": "05:59"},
            "mañana": {"inicio": "06:00", "fin": "11:59"},
            "tarde": {"inicio": "12:00", "fin": "17:59"},
            "noche": {"inicio": "18:00", "fin": "23:59"}
        }

    try:
        ref_dt = pd.to_datetime(fecha_referencia)
        expresion = expresion.lower().strip()
        
        # 1. Determinar el DÍA base
        target_date = ref_dt
        days_offset = 0
        is_week_or_month = False
        start_dt = None
        end_dt = None
        
        # Palabras clave para días
        if "ayer" in expresion or "anoche" in expresion:
            days_offset = -1
        elif "antier" in expresion or "anteayer" in expresion:
            days_offset = -2
        elif "hoy" in expresion:
            days_offset = 0
            
        # Detección de rangos más amplios
        if "semana pasada" in expresion:
            is_week_or_month = True
            current_weekday = ref_dt.weekday() # 0=Lunes
            this_monday = ref_dt - timedelta(days=current_weekday)
            last_monday = this_monday - timedelta(days=7)
            start_dt = last_monday.replace(hour=0, minute=0, second=0)
            end_dt = last_monday + timedelta(days=6, hours=23, minutes=59, seconds=59)
            
        elif "mes pasado" in expresion:
            is_week_or_month = True
            first_this_month = ref_dt.replace(day=1)
            last_prev_month = first_this_month - timedelta(days=1)
            start_prev_month = last_prev_month.replace(day=1)
            start_dt = start_prev_month.replace(hour=0, minute=0, second=0)
            end_dt = last_prev_month.replace(hour=23, minute=59, second=59)

        if not is_week_or_month:
            # Es un día Específico (Hoy, Ayer, Antier, o fecha implícita)
            # Nota: Si no encuentra keyword, asume '0' (hoy) o el contexto dado
            target_date = ref_dt + timedelta(days=days_offset)
            
            # 2. Determinar la FRANJA HORARIA dentro de ese día
            # Buscar coincidencia con claves de rangos_horarios (madrugada, mañana, tarde, noche)
            franja_encontrada = None
            
            # Prioridad: buscar tokens completos para evitar "mañana" (futuro) vs "mañana" (hora)
            # Solución simple: si dice "ayer por la mañana", el "ayer" ya definió el día -1.
            
            for key in rangos_horarios:
                # Usar regex para buscar la palabra completa (ej: evitar que 'anochecer' active 'noche' erróneamente si fuera el caso, 
                # o simplificar búsqueda)
                if key in expresion:
                    franja_encontrada = rangos_horarios[key]
                    break
            
            # Caso especial: "Anoche" implica ayer + noche
            if "anoche" in expresion:
                franja_encontrada = rangos_horarios["noche"]
                # Ya el offset se puso en -1 arriba

            if franja_encontrada:
                inicio_str = franja_encontrada["inicio"]
                fin_str = franja_encontrada["fin"]
                
                start_dt = datetime.strptime(f"{target_date.date()} {inicio_str}", "%Y-%m-%d %H:%M")
                end_dt = datetime.strptime(f"{target_date.date()} {fin_str}", "%Y-%m-%d %H:%M")
            else:
                # Todo el día (00:00 a 23:59)
                start_dt = target_date.replace(hour=0, minute=0, second=0)
                end_dt = target_date.replace(hour=23, minute=59, second=59)

        return {
            "status": "success",
            "fecha_inicio": start_dt.isoformat(),
            "fecha_fin": end_dt.isoformat(),
            "mensaje": f"Rango calculado para '{expresion}'"
        }

    except Exception as e:
        return {"status": "error", "mensaje": f"Error calculando fechas: {str(e)}"}

 """
# =====================================================
# ESCENARIO 1: CONSULTAS DE CONSUMO ENERGÉTICO BÁSICO
# =====================================================

@mcp.tool(
    meta = {
    "proposito": (
        "Obtiene valores de consumo energético para uno o varios dispositivos dentro "
        "de un rango temporal definido. "
        "La forma del resultado depende explícitamente del nivel de granularidad solicitado, "
        "lo que permite comparar consumos agregados o analizar su evolución temporal."
    ),
    "usar_si": [
        "Se necesita comparar el consumo total entre uno o varios dispositivos en un periodo definido (usar granularidad 'total')",
        "Se desea analizar cómo varía el consumo de uno o varios dispositivos a lo largo del tiempo (usar granularidad 'hora', 'dia' o 'mes')",
        "La solicitud del usuario implica una visualización comparativa entre dispositivos o una visualización temporal del consumo"
    ],
    "input_schema": {
        "type": "object",
        "properties": {
            "dispositivos": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Lista de dispositivos a consultar. "
                    "Cada dispositivo se devuelve como una serie independiente o como un valor agregado, "
                    "dependiendo de la granularidad. "
                    "Usar 'Total_Casa' para obtener el consumo agregado de todos los dispositivos."
                )
            },
            "fecha_inicio": {
                "type": "string",
                "description": (
                    "Fecha y hora de inicio del intervalo en formato ISO 8601 (YYYY-MM-DDTHH:MM). "
                    "Debe seleccionarse de forma coherente con la granularidad solicitada."
                )
            },
            "fecha_fin": {
                "type": "string",
                "description": (
                    "Fecha y hora de fin del intervalo en formato ISO 8601 (YYYY-MM-DDTHH:MM). "
                    "Define el límite superior del análisis temporal."
                )
            },
            "granularidad": {
                "type": "string",
                "enum": ["hora", "dia", "mes", "total"],
                "default": "total",
                "description": (
                    "Nivel temporal de agregación de los resultados. "
                    "Usar 'total' cuando se requiera un único valor acumulado por dispositivo "
                    "para facilitar comparaciones directas. "
                    "Usar 'hora', 'dia' o 'mes' cuando se requiera una serie temporal "
                    "que represente la evolución del consumo en el tiempo."
                )
            }
        },
        "required": ["dispositivos", "fecha_inicio", "fecha_fin"]
    },
    "output_schema": {
        "type": "object",
        "properties": {
            "status": {
                "type": "string",
                "enum": ["success", "no_data", "error"]
            },
            "periodo": {
                "type": "object",
                "properties": {
                    "inicio": {"type": "string"},
                    "fin": {"type": "string"}
                }
            },
            "granularidad": {
                "type": "string",
                "enum": ["hora", "dia", "mes", "total"]
            },
            "datos": {
                "type": "object",
                "description": (
                    "Resultados del consumo por dispositivo. "
                    "Con granularidad 'total', cada dispositivo devuelve un único valor numérico "
                    "representando el consumo acumulado en todo el periodo. "
                    "Con granularidad 'hora', 'dia' o 'mes', cada dispositivo devuelve una serie temporal "
                    "indexada por timestamps ISO 8601, adecuada para representar evolución en el tiempo."
                ),
                "additionalProperties": {
                    "oneOf": [
                        {"type": "number"},
                        {
                            "type": "object",
                            "additionalProperties": {"type": "number"}
                        }
                    ]
                }
            },
            "mensaje": {
                "type": "string"
            }
        }
    }
}
)

def obtener_consumo(dispositivos: list[str], fecha_inicio: str, fecha_fin: str, granularidad: str = "total") -> dict:
    try:
        # Filtrado temporal
        start_dt = pd.to_datetime(fecha_inicio)
        end_dt = pd.to_datetime(fecha_fin)
        
        mask = (DATASET['TimeStamp'] >= start_dt) & (DATASET['TimeStamp'] <= end_dt)
        df_filtered = DATASET.loc[mask].copy()
        
        if df_filtered.empty:
            return {"status": "no_data", "mensaje": "No hay datos en el rango seleccionado."}
        
        # Selección de columnas
        cols_to_sum = []
        for disp in dispositivos:
            if disp == "Total_Casa":
                # Asumimos que todas las columnas excepto TimeStamp son consumo
                numeric_cols = df_filtered.select_dtypes(include=[np.number]).columns.tolist()
                df_filtered["Total_Casa"] = df_filtered[numeric_cols].sum(axis=1)
                cols_to_sum.append("Total_Casa")
            elif disp in df_filtered.columns:
                cols_to_sum.append(disp)
            else:
                return {"status": "error", "mensaje": f"Dispositivo no encontrado: {disp}"}
                
        # Resampling
        df_filtered.set_index('TimeStamp', inplace=True)
        
        resample_rule = {
            "hora": "H",
            "dia": "D",
            "mes": "M",
            "total": None
        }.get(granularidad)
        
        result_data = {}
        
        if resample_rule:
            resampled = df_filtered[cols_to_sum].resample(resample_rule).sum()
            # Convertir a dict con fechas como string
            for col in cols_to_sum:
                # Filtrar valores 0 resultantes de huecos si es deseado, o mantenerlos
                result_data[col] = {k.isoformat(): v for k, v in resampled[col].items()}
        else:
            # Total del periodo
            totals = df_filtered[cols_to_sum].sum()
            for col in cols_to_sum:
                result_data[col] = float(totals[col])
                
        return {
            "status": "success",
            "periodo": {"inicio": fecha_inicio, "fin": fecha_fin},
            "granularidad": granularidad,
            "datos": result_data
        }
        
    except Exception as e:
        return {"status": "error", "mensaje": str(e)}


# =====================================================
# ESCENARIO 2: COMPARACIONES ENTRE DISPOSITIVOS O PERIODOS
# =====================================================

@mcp.tool(
    meta = {
    "proposito": (
        "Realiza un análisis comparativo entre dos objetivos de consumo energético, "
        "calculando diferencias absolutas y porcentuales a partir de valores acumulados. "
        "Esta función aporta interpretación analítica sobre consumos ya definidos, "
        "complementando la obtención directa de datos de consumo."
    ),
    "usar_si": [
        "Se requiere un análisis explícito de diferencias entre dos consumos acumulados",
        "El usuario solicita identificar cuál de dos objetivos consumió más energía y por cuánto",
        "Se desea complementar valores de consumo con un resumen comparativo e interpretativo"
    ],
    "input_schema": {
        "type": "object",
        "properties": {
            "objetivo_a": {
                "type": "object",
                "properties": {
                    "dispositivo": {"type": "string"},
                    "fecha_inicio": {"type": "string"},
                    "fecha_fin": {"type": "string"}
                },
                "required": ["dispositivo", "fecha_inicio", "fecha_fin"],
                "description": (
                    "Primer objetivo de comparación. "
                    "Representa el consumo total de un dispositivo dentro de un intervalo temporal definido."
                )
            },
            "objetivo_b": {
                "type": "object",
                "properties": {
                    "dispositivo": {"type": "string"},
                    "fecha_inicio": {"type": "string"},
                    "fecha_fin": {"type": "string"}
                },
                "required": ["dispositivo", "fecha_inicio", "fecha_fin"],
                "description": (
                    "Segundo objetivo de comparación. "
                    "Se interpreta de la misma forma que el objetivo A, permitiendo contrastar consumos acumulados."
                )
            }
        },
        "required": ["objetivo_a", "objetivo_b"]
    },
    "output_schema": {
        "type": "object",
        "properties": {
            "status": {
                "type": "string",
                "enum": ["success"]
            },
            "comparacion": {
                "type": "object",
                "properties": {
                    "valor_a": {"type": "number"},
                    "valor_b": {"type": "number"},
                    "diferencia_absoluta": {"type": "number"},
                    "diferencia_porcentual": {"type": "number"},
                    "mayor_consumo": {
                        "type": "string",
                        "enum": ["A", "B", "Iguales"]
                    }
                },
                "description": (
                    "Resultado del análisis comparativo entre ambos objetivos. "
                    "Los valores corresponden a consumos totales del periodo y las diferencias "
                    "constituyen un resumen analítico, no una serie temporal ni datos de visualización directa."
                )
            }
        }
    }
}
)

def analizar_comparacion(objetivo_a: dict, objetivo_b: dict) -> dict:
    
    def get_val(obj):
        # Reutilizamos lógica interna simple
        start = pd.to_datetime(obj['fecha_inicio'])
        end = pd.to_datetime(obj['fecha_fin'])
        mask = (DATASET['TimeStamp'] >= start) & (DATASET['TimeStamp'] <= end)
        df = DATASET.loc[mask]
        if df.empty: return 0.0
        disp = obj['dispositivo']
        if disp == "Total_Casa":
             numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
             return df[numeric_cols].sum().sum()
        return df[disp].sum() if disp in df.columns else 0.0

    val_a = get_val(objetivo_a)
    val_b = get_val(objetivo_b)
    
    diff_abs = val_a - val_b
    diff_pct = ((val_a - val_b) / val_b * 100) if val_b != 0 else 0.0
    
    mayor = "A" if val_a > val_b else ("B" if val_b > val_a else "Iguales")
    
    return {
        "status": "success",
        "comparacion": {
            "valor_a": val_a,
            "valor_b": val_b,
            "diferencia_absoluta": diff_abs,
            "diferencia_porcentual": round(diff_pct, 2),
            "mayor_consumo": mayor
        }
    }

# =====================================================
# ESCENARIO 3: DETECCIÓN DE ANOMALÍAS Y TENDENCIAS
# =====================================================

@mcp.tool(
    meta = {
    "proposito": (
        "Identifica eventos de consumo atípico dentro de un periodo temporal "
        "mediante análisis estadístico basado en Z-Score. "
        "Esta función permite detectar picos o valores inusuales respecto al patrón general del consumo."
    ),
    "usar_si": [
        "Se desea identificar picos, caídas o eventos puntuales fuera del comportamiento normal",
        "El usuario pregunta por consumos anormales, inesperados o atípicos",
        "Se requiere complementar el análisis del consumo con detección de eventos excepcionales"
    ],
    "input_schema": {
        "type": "object",
        "properties": {
            "dispositivo": {
                "type": "string",
                "description": (
                    "Dispositivo sobre el cual se analizan anomalías. "
                    "El análisis se realiza sobre la serie de consumo implícita en el intervalo definido."
                )
            },
            "fecha_inicio": {
                "type": "string",
                "description": "Inicio del periodo de análisis temporal."
            },
            "fecha_fin": {
                "type": "string",
                "description": "Fin del periodo de análisis temporal."
            },
            "sensibilidad": {
                "type": "number",
                "description": (
                    "Umbral del Z-Score utilizado para definir qué se considera una anomalía. "
                    "Valores menores implican mayor sensibilidad y detección de más eventos."
                )
            }
        },
        "required": ["dispositivo", "fecha_inicio", "fecha_fin"]
    },
    "output_schema": {
        "type": "object",
        "properties": {
            "status": {
                "type": "string",
                "enum": ["success", "no_data", "error"]
            },
            "estadisticas": {
                "type": "object",
                "properties": {
                    "media": {"type": "number"},
                    "std": {"type": "number"}
                },
                "description": (
                    "Estadísticas descriptivas de la serie de consumo utilizadas como referencia "
                    "para la detección de anomalías."
                )
            },
            "total_anomalias": {"type": "number"},
            "eventos": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "fecha": {"type": "string"},
                        "valor": {"type": "number"},
                        "z_score": {"type": "number"}
                    }
                },
                "description": (
                    "Lista de eventos detectados como atípicos. "
                    "Cada evento representa un punto puntual fuera del patrón normal, "
                    "no una tendencia ni una serie temporal completa."
                )
            },
            "mensaje": {"type": "string"}
        }
    }
}
)

def detectar_anomalias(dispositivo: str, fecha_inicio: str, fecha_fin: str, sensibilidad: float = 3.0) -> dict:
    start = pd.to_datetime(fecha_inicio)
    end = pd.to_datetime(fecha_fin)
    mask = (DATASET['TimeStamp'] >= start) & (DATASET['TimeStamp'] <= end)
    df = DATASET.loc[mask].copy()
    
    if df.empty:
        return {"status": "no_data"}
        
    target_col = dispositivo
    if dispositivo == "Total_Casa":
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        df["Total_Casa"] = df[numeric_cols].sum(axis=1)
        target_col = "Total_Casa"
    elif dispositivo not in df.columns:
         return {"status": "error", "mensaje": "Dispositivo no encontrado"}
         
    series = df[target_col]
    mean = series.mean()
    std = series.std()
    
    if std == 0:
        return {"status": "success", "anomalias": [], "mensaje": "Desviación estándar 0, datos planos."}
        
    # Cálculo Z-Score
    df['z_score'] = (series - mean) / std
    anomalias_df = df[df['z_score'].abs() > sensibilidad]
    
    eventos = []
    for idx, row in anomalias_df.iterrows():
        eventos.append({
            "fecha": row['TimeStamp'].isoformat(),
            "valor": row[target_col],
            "z_score": row['z_score']
        })
        
    return {
        "status": "success",
        "estadisticas": {"media": mean, "std": std},
        "total_anomalias": len(eventos),
        "eventos": eventos
    }

@mcp.tool(
    meta = {
    "proposito": (
        "Determina la tendencia general del consumo energético en un intervalo temporal "
        "mediante un ajuste de regresión lineal. "
        "Esta función resume la dirección y estabilidad del consumo a lo largo del tiempo."
    ),
    "usar_si": [
        "Se desea conocer si el consumo presenta una tendencia creciente, decreciente o estable",
        "El usuario pregunta por la dirección general o comportamiento global del consumo",
        "Se requiere un análisis sintético de la evolución del consumo en el periodo"
    ],
    "input_schema": {
        "type": "object",
        "properties": {
            "dispositivo": {
                "type": "string",
                "description": (
                    "Dispositivo cuyo consumo se analiza. "
                    "El cálculo se basa en la serie temporal implícita del intervalo seleccionado."
                )
            },
            "fecha_inicio": {
                "type": "string",
                "description": "Inicio del periodo usado para estimar la tendencia."
            },
            "fecha_fin": {
                "type": "string",
                "description": "Fin del periodo usado para estimar la tendencia."
            }
        },
        "required": ["dispositivo", "fecha_inicio", "fecha_fin"]
    },
    "output_schema": {
        "type": "object",
        "properties": {
            "status": {
                "type": "string",
                "enum": ["success", "no_data"]
            },
            "tendencia": {
                "type": "object",
                "properties": {
                    "direccion": {
                        "type": "string",
                        "enum": [
                            "Creciente",
                            "Decreciente",
                            "Estable",
                            "Sin tendencia clara (ruido)"
                        ]
                    },
                    "pendiente": {"type": "number"},
                    "r_cuadrado": {"type": "number"}
                },
                "description": (
                    "Resumen analítico de la tendencia del consumo. "
                    "Describe la dirección general y la calidad del ajuste, "
                    "no eventos puntuales ni valores individuales."
                )
            }
        }
    }
}
)

def analizar_tendencia(dispositivo: str, fecha_inicio: str, fecha_fin: str) -> dict:
    start = pd.to_datetime(fecha_inicio)
    end = pd.to_datetime(fecha_fin)
    mask = (DATASET['TimeStamp'] >= start) & (DATASET['TimeStamp'] <= end)
    df = DATASET.loc[mask].copy()
    
    if df.empty: return {"status": "no_data"}
    
    target_col = dispositivo
    if dispositivo == "Total_Casa":
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        df["Total_Casa"] = df[numeric_cols].sum(axis=1)
        target_col = "Total_Casa"
        
    # Preparar datos para regresión
    # Convertir fecha a numérico (días desde el inicio)
    df['dias_relativos'] = (df['TimeStamp'] - df['TimeStamp'].min()).dt.total_seconds() / 86400
    
    X = df['dias_relativos'].values
    y = df[target_col].values
    
    # Regresión lineal simple: y = mx + b
    # np.polyfit grado 1
    slope, intercept = np.polyfit(X, y, 1)
    
    # Calcular R^2
    y_pred = slope * X + intercept
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    # Interpretación
    direccion = "Estable"
    if slope > 0.05: direccion = "Creciente" # Umbral arbitrario pequeño
    elif slope < -0.05: direccion = "Decreciente"
    
    if r_squared < 0.1:
        direccion = "Sin tendencia clara (ruido)"
        
    return {
        "status": "success",
        "tendencia": {
            "direccion": direccion,
            "pendiente": slope,
            "r_cuadrado": r_squared
        }
    }



if __name__ == "__main__":
    mcp.run(transport="sse")