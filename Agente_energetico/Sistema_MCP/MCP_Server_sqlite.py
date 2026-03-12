from fastmcp import FastMCP
import pandas as pd
import numpy as np
import sqlite3

# Initialize FastMCP server
mcp = FastMCP("MCP_Server_Gravity")

# -------------------------
# Configuración Base de Datos
# -------------------------
DB_PATH = r'C:\sqlite_tesis\Base_datos_tesis\Hogar_Sincelejo.db'
TABLE_NAME = "Energy Consumption in KWh of a Typical House Sincelejo Colombia"

def get_data_from_db(fecha_inicio: str = None, fecha_fin: str = None) -> pd.DataFrame:
    """
    Conecta a la DB y devuelve el dataset (o un fragmento) como DataFrame.
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        query = f'SELECT * FROM "{TABLE_NAME}"'
        
        if fecha_inicio and fecha_fin:
            query += f" WHERE TimeStamp BETWEEN '{fecha_inicio}' AND '{fecha_fin}'"
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        # Asegurar tipos
        if not df.empty:
            df['TimeStamp'] = pd.to_datetime(df['TimeStamp'])
        return df
    except Exception as e:
        print(f"Error accediendo a la base de datos: {e}")
        return pd.DataFrame()


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
        # Cargar datos filtrados desde DB
        df_filtered = get_data_from_db(fecha_inicio, fecha_fin)
        
        if df_filtered.empty:
            return {"status": "no_data", "mensaje": "No hay datos en el rango seleccionado."}
        
        # Selección de columnas
        cols_to_sum = []
        for disp in dispositivos:
            if disp == "Total_Casa":
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
            for col in cols_to_sum:
                result_data[col] = {k.isoformat(): v for k, v in resampled[col].items()}
        else:
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
        # Usamos la función de acceso a DB para obtener solo el fragmento necesario
        df = get_data_from_db(obj['fecha_inicio'], obj['fecha_fin'])
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
    # Cargar datos desde DB
    df = get_data_from_db(fecha_inicio, fecha_fin)
    
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
    # Cargar datos desde DB
    df = get_data_from_db(fecha_inicio, fecha_fin)
    
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