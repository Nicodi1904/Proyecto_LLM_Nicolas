# Evaluación del Agente Planeador - Distribución de Diferentes Niveles

Este documento describe la estructura y el propósito de los 50 casos de prueba incluidos en el archivo `casos_evaluacion_planeador.json`. Estos casos fueron diseñados para evaluar de forma exhaustiva y progresiva las capacidades cognitivas, lógicas y analíticas del Agente Planeador.

## Distribución del JSON
El archivo JSON agrupa los 50 casos en 5 llaves (keys) principales, cada una representando un nivel de dificultad computacional para el LLM (L1 a L5). Dentro de cada nivel, hay 10 diccionarios que simulan las distintas entradas que haría el Interpretador una vez recibida la voz/petición del usuario final. 

---

### Nivel 1 — Básico (L1_01 a L1_10)
**Objetivo:** Evaluar el uso correcto, directo y sin ambigüedades de la herramienta base para la obtención de datos (`obtener_consumo`).
- **Mecánica:** Se procesa una única solicitud (`@1`) con fechas explícitas y exactas (bien sea usando formato ISO 8601 directamente o mediante términos muy cerrados como *"ayer"*).
- **El Reto:** Validar que el agente es capaz de abstraer correctamente extracciones entidad-relación básicas: identificar qué dispositivo se está buscando y mapearlo correctamente al parámetro de la herramienta. Es la línea base (Baseline).

---

### Nivel 2 — Temporal Relativo (L2_01 a L2_10)
**Objetivo:** Introducir la matemática temporal y la capacidad del sistema de interpretar lenguajes de rangos de calendario.
- **Mecánica:** Se procesa una solicitud (`@1`), pero ahora el objetivo utiliza lenguaje coloquial para agrupar días: *"últimos 3 días"*, *"esta semana"*, *"desde hace 4 días hasta hoy"*.
- **El Reto:** Fuerza de pleno al Planeador a utilizar el parámetro `temporal_context` inyectado en el *Prompt* para hacer matemáticas. El modelo debe tomar el campo simulado de `"fecha_actual"` y restarle días lógicamente antes de inyectarlos como `fecha_inicio` y `fecha_fin` en el formato estricto *ISO 8601*.

---

### Nivel 3 — Temporal Difuso (L3_01 a L3_10)
**Objetivo:** Poner a prueba la integración de referencias de la Interfaz (UI) a lógica sistémica.
- **Mecánica:** Se introducen conceptos extremadamente vagos dependientes de la definición personal del usuario: *"noche"*, *"madrugada"*, *"media tarde"*.
- **El Reto:** Aquí se busca generar la fusión perfecta. Por un lado, el LLM debe leer la llave de formato horario en sus `temporal_preferences` (que previamente ha sido filtrado al estándar de 24 horas) y, de forma cruzada, debe tomar el `temporal_context` para entender qué día en específico armar. Ej: *"media tarde de ayer"* debe ensamblar un inicio y fin sintético perfecto.

---

### Nivel 4 — Multi-Step Analógico (L4_01 a L4_10)
**Objetivo:** Probar la composición cruzada de los tres niveles anteriores dirigidos, a su vez, a una herramienta más densa computacionalmente.
- **Mecánica:** Mantiene una sola solicitud (`@1`) bajo escenarios explícitamente etiquetados como `comparacion_consumos`, desafiando comparaciones implícitas como `"Comparar X entre el tiempo A y el tiempo B"`.
- **El Reto:** Exigirle al Planeador invocar la herramienta `analizar_comparacion`. Requiere no solo deducción matemática de fechas (por duplicado de una forma compleja), sino rellenar los datos precisos de dos nodos internos: `objetivo_a` y `objetivo_b`. Comprueba qué tan bien unifica el control parametral.

---

### Nivel 5 — Complejo y Realista (L5_01 a L5_10)
**Objetivo:** Simular la aleatoriedad y complejidad del lenguaje coloquial humano (Cross-Solicitud en Cascada).
- **Mecánica:** Abandona las directivas únicas e inserta múltiples solicitudes concatenadas (`@1`, `@2`, `@3`) dentro de un mismo prompt, además de inyectar dependencias y referencias cruzadas entre ellas.
- **El Reto:** Es el nivel más avanzado y probará los límites nativos del agente en cuanto a "Resolución de Anáforas". Frases en el aire como *"Y compáralo con el consumo del lunes"* u *"Analizar SU tendencia"* dependen inherentemente de retener en memoria semántica que el objeto base era el de la instrucción anterior (`@1`), mientras se calculan paralelamente anomalías y herramientas simultáneas sin cruzar parámetros perjudicialmente.
