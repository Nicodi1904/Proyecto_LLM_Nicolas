import dspy
import os
import json
from dotenv import load_dotenv

# -------------------------------------------------------------------------
# Configuración e Importaciones
# -------------------------------------------------------------------------

# Cargar API Keys
env_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(env_path)

# Importar summary (Metadata de herramientas)
try:
    from MCP_C_obtener_summary import system_summary
except ImportError:
    import sys
    sys.path.append(os.path.dirname(__file__))
    from MCP_C_obtener_summary import system_summary

# Importaciones del Pipeline (Interpretador, Planeador y Verificador)
try:
    from Test_Interpretador import Interpretador, escenarios_entrada
except ImportError:
    print("Advertencia: No se pudo importar Interpretador o escenarios de Test_Interpretador.py")
    Interpretador = None
    escenarios_entrada = None

try:
    from Test_planeador import Planeador
except ImportError:
    print("Advertencia: No se pudo importar Planeador de Test_planeador.py")
    Planeador = None

try:
    from test_worker_verificador import verificar_completo, filtrar_acciones
except ImportError:
    print("Advertencia: No se pudo importar funciones de verificación.")
    verificar_completo = None
    filtrar_acciones = None

# -------------------------------------------------------------------------
# Configuración de LLM (Llama 3.1 8b por defecto)
# -------------------------------------------------------------------------
MODELOS = {
    "llama3.1": dspy.LM('ollama_chat/llama3.1:latest', api_base='http://localhost:11434', api_key=''),
    # Se pueden agregar más si se requieren
}

print("Sistema inicializado correctamente con módulos externos.")

# -------------------------------------------------------------------------
# Pipeline Principal
# -------------------------------------------------------------------------
def ejecutar_pipeline_completo():
    """
    Ejecuta: Input -> Interpretador -> Planeador -> Verificador -> Output JSON
    """
    print(f"\n{'='*60}")
    print("SISTEMA DE GESTIÓN ENERGÉTICA - MODO INTERACTIVO")
    print(f"{'='*60}\n")

    # Selección de LLM
    nombre_modelo = "llama3.1" 
    lm = MODELOS.get(nombre_modelo)
    if not lm:
        print(f"Error: Modelo {nombre_modelo} no encontrado.")
        return
    dspy.configure(lm=lm)
    print(f">> Modelo activo: {nombre_modelo}\n")

    # Input Usuario
    prompt_usuario = input("¿En qué te puedo ayudar hoy? ")
    if not prompt_usuario:
        print("Entrada vacía. Saliendo.")
        return

    # Contexto Temporal Simulado
    temporal_context = {
        "referencia_actual": "2024-12-29T09:00:00",
        "zona_horaria": "America/Bogota",
        "rangos_horarios": {
            "madrugada": {"inicio": "00:00", "fin": "05:59"},
            "mañana": {"inicio": "06:00", "fin": "11:59"},
            "tarde": {"inicio": "12:00", "fin": "17:59"},
            "noche": {"inicio": "18:00", "fin": "23:59"}
        }
    }
    
    # Dispositivos Válidos Simulados
    dispositivos_conocidos = ["Ventilador", "PC", "TV", "Total_Casa", "AC", "Lampara"]

    # 1. Interpretación
    print(f"\n[1/3] Interpretando solicitud...")
    try:
        resultado_interpretador = dspy.Predict(Interpretador)(
            prompt_usuario=prompt_usuario,
            escenarios_entrada=escenarios_entrada
        )
        solicitudes = resultado_interpretador.solicitudes_categorizadas
        print(f"   > Solicitudes identificadas: {len(solicitudes) if solicitudes else 0}")
    except Exception as e:
        print(f"Error en Interpretador: {e}")
        return

    if not solicitudes:
        print("No se identificaron solicitudes válidas.")
        return

    # 2. Planificación
    print(f"\n[2/3] Planificando acciones...")
    if not Planeador:
        print("Error: Clase Planeador no disponible.")
        return

    try:
        resultado_planeador = dspy.Predict(Planeador)(
            solicitudes_categorizadas=solicitudes,
            system_summary=system_summary,
            temporal_context=temporal_context
        )
        plan_acciones = resultado_planeador.plan_acciones
        print(f"   > Acciones propuestas: {len(plan_acciones) if plan_acciones else 0}")
    except Exception as e:
        print(f"Error en Planeador: {e}")
        return

    if not plan_acciones:
        print("El planeador no generó ninguna acción.")
        return

    # 3. Verificación
    print(f"\n[3/3] Verificando y Filtrando...")
    if not verificar_completo or not filtrar_acciones:
        print("Error: Funciones de verificación no disponibles.")
        return

    try:
        reporte_verificacion = verificar_completo(
            plan_acciones, system_summary, dispositivos_conocidos, temporal_context
        )
        acciones_validas, acciones_invalidas = filtrar_acciones(reporte_verificacion, plan_acciones)
        
        print(f"   > Acciones Válidas: {len(acciones_validas)}")
        print(f"   > Acciones Inválidas: {len(acciones_invalidas)}")
        
        if acciones_invalidas:
            print("\n   [!] Detalle Inválidas:")
            for inv in acciones_invalidas:
                print(f"       - {inv.get('id')}: {inv.get('error_verificacion')}")
    except Exception as e:
        print(f"Error en Verificador: {e}")
        return

    # Salida Final
    print(f"\n{'='*60}")
    print("OUTPUT PARA EJECUTOR (MCP_C)")
    print(f"{'='*60}")
    print(json.dumps(acciones_validas, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    ejecutar_pipeline_completo()

