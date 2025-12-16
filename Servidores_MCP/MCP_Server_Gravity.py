from fastmcp import FastMCP
import pandas as pd
import numpy as np
from datetime import datetime
import sys
import os

# Agregar directorio padre al path para importar módulos compartidos
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from cargar_CSV import cargar_dataset_sinselejo

# Initialize FastMCP server
mcp = FastMCP("MCP_Server_Gravity")

# -------------------------
# Cargar dataset global
# -------------------------
# Construir ruta absoluta al CSV (asumiendo que está en el directorio padre)
csv_path = os.path.join(os.path.dirname(__file__), '..', "Energy Consumption in KWh of a Typical House Sincelejo Colombia.csv")
DATASET = cargar_dataset_sinselejo(csv_path)
# Asegurar que TimeStamp es datetime
DATASET['TimeStamp'] = pd.to_datetime(DATASET['TimeStamp'])

def meta_energy(proposito: str, usar_si: list):
    """
    Genera los metadatos estructurados para las herramientas.
    """
    return { 
        "proposito": proposito,
        "usar_si": usar_si
    }

# =====================================================
# ESCENARIO 1: CONSULTAS DE CONSUMO ENERGÉTICO BÁSICO
# =====================================================

@mcp.tool(
    meta={
        **meta_energy(
            proposito="Obtiene datos brutos o sumatorias de consumo energético filtrando por rango de fechas y dispositivo específico o general.",
            usar_si=[
                "Necesitas valores históricos",
                "Consultas consumo total de la casa",
                "Analizas un periodo de tiempo definido"
            ]
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "dispositivos": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Lista de dispositivos a consultar. Usar 'Total_Casa' para el consumo global."
                },
                "fecha_inicio": {"type": "string", "description": "Fecha inicio ISO 8601 (YYYY-MM-DDTHH:MM)"},
                "fecha_fin": {"type": "string", "description": "Fecha fin ISO 8601 (YYYY-MM-DDTHH:MM)"},
                "granularidad": {
                    "type": "string",
                    "enum": ["hora", "dia", "mes", "total"],
                    "default": "total",
                    "description": "Cómo agrupar los datos temporalmente."
                }
            },
            "required": ["dispositivos", "fecha_inicio", "fecha_fin"]
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
    meta={
        **meta_energy(
            proposito="Calcula y contrasta la diferencia absoluta y porcentual entre dos conjuntos de datos de consumo definidos por periodo y dispositivo.",
            usar_si=[
                "El usuario pide comparar dos meses/días",
                "Se requiere saber qué dispositivo consumió más"
            ]
        ),
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
                    "required": ["dispositivo", "fecha_inicio", "fecha_fin"]
                },
                "objetivo_b": {
                    "type": "object",
                    "properties": {
                        "dispositivo": {"type": "string"},
                        "fecha_inicio": {"type": "string"},
                        "fecha_fin": {"type": "string"}
                    },
                    "required": ["dispositivo", "fecha_inicio", "fecha_fin"]
                }
            },
            "required": ["objetivo_a", "objetivo_b"]
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
    meta={
        **meta_energy(
            proposito="Evalúa estadísticamente un periodo de consumo usando Z-Score para identificar picos o comportamientos fuera del promedio estándar.",
            usar_si=[
                "Buscas outliers o eventos inusuales",
                "El usuario pregunta por consumos extraños"
            ]
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "dispositivo": {"type": "string"},
                "fecha_inicio": {"type": "string"},
                "fecha_fin": {"type": "string"},
                "sensibilidad": {"type": "number", "default": 3.0, "description": "Umbral Z-Score (defecto 3.0)"}
            },
            "required": ["dispositivo", "fecha_inicio", "fecha_fin"]
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
    meta={
        **meta_energy(
            proposito="Ajusta una regresión lineal sobre los datos de consumo temporal para determinar si la tendencia es creciente, decreciente o estable.",
            usar_si=[
                "Se necesita proyección o dirección del consumo",
                "El usuario pregunta si está ahorrando o gastando más"
            ]
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "dispositivo": {"type": "string"},
                "fecha_inicio": {"type": "string"},
                "fecha_fin": {"type": "string"}
            },
            "required": ["dispositivo", "fecha_inicio", "fecha_fin"]
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

# =====================================================
# ESCENARIO 5: ENTRADAS INADMISIBLES Y CONTROL
# =====================================================

@mcp.tool(
    meta={
        **meta_energy(
            proposito="Informa al usuario que su solicitud no es realizable con las herramientas actuales explicando la razón técnica.",
            usar_si=["No existe ninguna herramienta aplicable a la intención"]
        ),
        "input_schema": {
            "type": "object",
            "properties": {"razon": {"type": "string"}},
            "required": ["razon"]
        }
    }
)
def plan_inviable(razon: str) -> str:
    print(f"[TOOL_USE] Plan inviable: {razon}")
    return f"Plan inviable: {razon}"


@mcp.tool(
    meta={
        **meta_energy(
            proposito="Detiene el flujo para solicitar al usuario parámetros obligatorios faltantes necesarios para ejecutar otra herramienta.",
            usar_si=["Falta un parámetro necesario"]
        ),
        "input_schema": {
            "type": "object",
            "properties": {"datos_faltante": {"type": "string"}},
            "required": ["datos_faltante"]
        }
    }
)
def falta_informacion(datos_faltante: str) -> str:
    print(f"[TOOL_USE] Faltan datos: {datos_faltante}")
    return f"Faltan datos: {datos_faltante}"

if __name__ == "__main__":
    mcp.run(transport="sse")
