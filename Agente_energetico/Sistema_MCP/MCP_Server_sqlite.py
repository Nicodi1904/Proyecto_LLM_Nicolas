#---------------
#SERVIDOR MCP CARGADO
#---------------

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

# Mapeo de nombres humanos a nombres de columnas en la DB
MAPEO_DISPOSITIVOS = {
    # AC
    "ac": "AC",
    "aire": "AC",
    "aire acondicionado": "AC",
    "refrigeracion": "AC",
    "clima": "AC",
    "enfriamiento": "AC",
    
    # Ventilador
    "ventilador": "Ventilador",
    "abanico": "Ventilador",
    "fan": "Ventilador",
    "soplador": "Ventilador",
    
    # PC
    "pc": "PC",
    "computador": "PC",
    "computadora": "PC",
    "ordenador": "PC",
    "computador personal": "PC",
    "laptop": "PC",
    "portatil": "PC",
    
    # Lampara
    "lampara": "Lampara",
    "luz": "Lampara",
    "luces": "Lampara",
    "iluminacion": "Lampara",
    "iluminación": "Lampara",
    "bombillo": "Lampara",
    "foco": "Lampara",
    "luminaria": "Lampara",
    
    # TV
    "tv": "TV",
    "televisor": "TV",
    "tele": "TV",
    "pantalla": "TV",
    "television": "TV",
    
    # Total
    "total_casa": "Total_Casa",
    "total": "Total_Casa",
    "casa": "Total_Casa",
    "todo": "Total_Casa",
    "consumo general": "Total_Casa",
    "consumo total": "Total_Casa",
    "hogar": "Total_Casa"
}

def normalizar_dispositivo(nombre: str) -> str:
    """
    Normaliza el nombre del dispositivo usando el mapeo.
    Si no se encuentra, devuelve el nombre original.
    """
    if not isinstance(nombre, str): return nombre
    nombre_clean = nombre.lower().strip()
    return MAPEO_DISPOSITIVOS.get(nombre_clean, nombre)

def get_data_from_db(fecha_inicio: str = None, fecha_fin: str = None) -> pd.DataFrame:
    """
    Conecta a la DB y devuelve el dataset (o un fragmento) como DataFrame.
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        query = f'SELECT * FROM "{TABLE_NAME}"'
        
        if fecha_inicio and fecha_fin:
            # Normalizar formato ISO (T -> espacio) para compatibilidad con SQLite string compare
            fi = fecha_inicio.replace("T", " ")
            ff = fecha_fin.replace("T", " ")
            query += f" WHERE TimeStamp BETWEEN '{fi}' AND '{ff}'"
        
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
# ESCENARIO 1: CONSULTAS DE CONSUMO ENERGÉTICO BÁSICO.
# =====================================================

@mcp.tool(
    meta = {
        "descripcion": "Obtiene consumo energético de uno o varios dispositivos en un intervalo temporal.",
        "cuando_usar": [
            "Comparaciones de consumo → usar granularidad 'total'",
            "Análisis de evolución temporal → usar 'hora', 'dia' o 'mes'",
            "Visualización de consumo en el tiempo o entre dispositivos"
        ],
        "nota_modelo": "Devuelve valores acumulados si granularidad='total', o series temporales si es 'hora','dia','mes'.",
        "input_schema": {
            "type": "object",
            "properties": {
                "dispositivos": {
                    "type": "array",
                    "items": { "type": "string" },
                    "description": "Lista de dispositivos. Usar 'Total_Casa' para consumo agregado."
                },
                "fecha_inicio": {
                    "type": "string",
                    "description": "Inicio en formato ISO 8601 (YYYY-MM-DDTHH:MM)."
                },
                "fecha_fin": {
                    "type": "string",
                    "description": "Fin en formato ISO 8601 (YYYY-MM-DDTHH:MM)."
                },
                "granularidad": {
                    "type": "string",
                    "enum": ["hora", "dia", "mes", "total"],
                    "default": "total",
                    "description": "Nivel de agregación: 'total' para comparación directa, otros para series temporales."
                }
            },
            "required": ["dispositivos", "fecha_inicio", "fecha_fin"]
        },
        "output_schema": "['status', 'periodo', 'granularidad', 'datos', 'mensaje']"
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
        for disp_raw in dispositivos:
            disp = normalizar_dispositivo(disp_raw)
            if disp == "Total_Casa":
                numeric_cols = df_filtered.select_dtypes(include=[np.number]).columns.tolist()
                df_filtered["Total_Casa"] = df_filtered[numeric_cols].sum(axis=1)
                cols_to_sum.append("Total_Casa")
            elif disp in df_filtered.columns:
                cols_to_sum.append(disp)
            else:
                return {"status": "error", "mensaje": f"Dispositivo no encontrado: {disp_raw} (Normalizado como: {disp})"}
                
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
        "descripcion": "Compara dos consumos acumulados y calcula diferencias absolutas y porcentuales.",
        "cuando_usar": [
            "Comparar explícitamente dos consumos",
            "Determinar cuál consumo es mayor",
            "Obtener diferencias absolutas o porcentuales"
        ],
        "nota_modelo": "Los inputs deben representar consumos acumulados (no series temporales).",
        "input_schema": {
            "type": "object",
            "properties": {
                "objetivo_a": {
                    "type": "object",
                    "properties": {
                        "dispositivo": { "type": "string" },
                        "fecha_inicio": { "type": "string" },
                        "fecha_fin": { "type": "string" }
                    },
                    "required": ["dispositivo", "fecha_inicio", "fecha_fin"]
                },
                "objetivo_b": {
                    "type": "object",
                    "properties": {
                        "dispositivo": { "type": "string" },
                        "fecha_inicio": { "type": "string" },
                        "fecha_fin": { "type": "string" }
                    },
                    "required": ["dispositivo", "fecha_inicio", "fecha_fin"]
                }
            },
            "required": ["objetivo_a", "objetivo_b"]
        },
        "output_schema": "['status', 'comparacion']"
    }
)

def analizar_comparacion(objetivo_a: dict, objetivo_b: dict) -> dict:
    
    def get_val(obj):
        # Usamos la función de acceso a DB para obtener solo el fragmento necesario
        df = get_data_from_db(obj['fecha_inicio'], obj['fecha_fin'])
        if df.empty: return 0.0
        disp_raw = obj['dispositivo']
        disp = normalizar_dispositivo(disp_raw)
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
        "descripcion": "Detecta consumos atípicos en un intervalo usando análisis estadístico (Z-Score).",
        "cuando_usar": [
            "Identificar picos o valores anormales",
            "Detectar comportamientos fuera de lo esperado",
            "Analizar eventos puntuales en el consumo"
        ],
        "nota_modelo": "Requiere una serie temporal implícita en el intervalo; no usar con valores agregados.",
        "input_schema": {
            "type": "object",
            "properties": {
                "dispositivo": {
                    "type": "string",
                    "description": "Dispositivo a analizar."
                },
                "fecha_inicio": {
                    "type": "string",
                    "description": "Inicio del intervalo (ISO 8601)."
                },
                "fecha_fin": {
                    "type": "string",
                    "description": "Fin del intervalo (ISO 8601)."
                },
                "sensibilidad": {
                    "type": "number",
                    "description": "Umbral de detección (menor valor = más sensible)."
                }
            },
            "required": ["dispositivo", "fecha_inicio", "fecha_fin"]
        },
        "output_schema": "['status', 'estadisticas', 'total_anomalias', 'eventos', 'mensaje']"
    }
)

def detectar_anomalias(dispositivo: str, fecha_inicio: str, fecha_fin: str, sensibilidad: float = 3.0) -> dict:
    # Cargar datos desde DB
    df = get_data_from_db(fecha_inicio, fecha_fin)
    
    if df.empty:
        return {"status": "no_data"}
        
    target_col = normalizar_dispositivo(dispositivo)
    if target_col == "Total_Casa":
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        df["Total_Casa"] = df[numeric_cols].sum(axis=1)
    elif target_col not in df.columns:
         return {"status": "error", "mensaje": f"Dispositivo no encontrado: {dispositivo} (Normalizado como: {target_col})"}
         
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
        "descripcion": "Determina la tendencia del consumo (creciente, decreciente o estable) en un intervalo.",
        "cuando_usar": [
            "Evaluar dirección general del consumo",
            "Analizar comportamiento global en el tiempo"
        ],
        "nota_modelo": "Se basa en una serie temporal del intervalo; no es para eventos puntuales ni valores únicos.",
        "input_schema": {
            "type": "object",
            "properties": {
                "dispositivo": {
                    "type": "string",
                    "description": "Dispositivo a analizar."
                },
                "fecha_inicio": {
                    "type": "string",
                    "description": "Inicio del intervalo (ISO 8601)."
                },
                "fecha_fin": {
                    "type": "string",
                    "description": "Fin del intervalo (ISO 8601)."
                }
            },
            "required": ["dispositivo", "fecha_inicio", "fecha_fin"]
        },
        "output_schema": "['status', 'tendencia']"
    }
)

def analizar_tendencia(dispositivo: str, fecha_inicio: str, fecha_fin: str) -> dict:
    # Cargar datos desde DB
    df = get_data_from_db(fecha_inicio, fecha_fin)
    
    if df.empty: return {"status": "no_data"}
    
    target_col = normalizar_dispositivo(dispositivo)
    if target_col == "Total_Casa":
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        df["Total_Casa"] = df[numeric_cols].sum(axis=1)
    elif target_col not in df.columns:
         return {"status": "error", "mensaje": f"Dispositivo no encontrado: {dispositivo} (Normalizado como: {target_col})"}
        
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

@mcp.tool(
    name="accion_imposible",
    description="Registra solicitudes que el agente determina como imposibles de ejecutar.",
    meta={
        "descripcion": "Se usa como vía de escape cuando la solicitud pide procesar información de una hora o fecha que aún no ocurre, o es inviable lógicamente.",
        "cuando_usar": [
            "El usuario pide datos de eventos futuros ('la tarde de hoy' cuando es de mañana)",
            "Fechas incoherentes o solapamientos absurdos"
        ],
        "nota_modelo": "Proporciona la solicitud exacta y justifica por qué no se puede ejecutar en base al contexto temporal.",
        "input_schema": {
            "type": "object",
            "properties": {
                "solicitud": {
                    "type": "string",
                    "description": "La solicitud original o acción que se pretendía tomar."
                },
                "justificacion": {
                    "type": "string",
                    "description": "Explicación lógica de por qué la acción es imposible (ej. 'La hora solicitada (20:00) está en el futuro')."
                }
            },
            "required": ["solicitud", "justificacion"]
        },
        "output_schema": "['status', 'solicitud', 'justificacion']"
    }
)
def accion_imposible(solicitud: str, justificacion: str) -> dict:
    return {
        "status": "impossible",
        "solicitud": solicitud,
        "justificacion": justificacion
    }

if __name__ == "__main__":
    mcp.run(transport="sse")