from PySide6.QtGui import QColor, QFont

class Tema:
    """
    Centralización de toda la estética visual del Agente Energético.
    Define colores, fuentes y constantes de diseño para mantener la coherencia.
    """
    # ── Paleta de Colores ────────────────────────────────────────────────────
    AMARILLO = "#FFB300"           # Color principal de marca
    AMARILLO_HOVER = "#FFA000"     # Variación para estados de ratón encima
    CRISTAL_BG = QColor(45, 30, 20, 245) # Fondo oscuro con alta opacidad (Efecto Cristal)
    MARCO_WHITE = "rgba(255, 255, 255, 30)" # Bordes sutiles para contenedores
    TEXTO_SUAVE = "rgba(255, 255, 255, 200)" # Color para textos secundarios
    
    # ── Tipografía ───────────────────────────────────────────────────────────
    FUENTE_TITULO = ("Segoe UI", 26, QFont.Bold)
    FUENTE_SUBTITULO = ("Segoe UI", 22, QFont.Bold)
    FUENTE_BOTON = ("Segoe UI", 16, QFont.Bold)
    
    # ── Configuración de Widget Flotante ─────────────────────────────────────
    WIDGET_TEXT_BG = (255, 255, 255, 200)
    WIDGET_TEXT_BORDER = (0, 0, 0, 150)
    WIDGET_ESTILO_TEXTO = ("Segoe UI", 8, QFont.Bold)
    WIDGET_RECT_AJUSTE = (5, 80, -5, -5)
