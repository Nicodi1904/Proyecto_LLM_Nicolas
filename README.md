# Agente Energético - Monitoreo Inteligente con LLMs

Este repositorio contiene el software desarrollado para el proyecto de tesis: **"INTEGRACIÓN DE UNA HERRAMIENTA BASADA EN MODELOS DE LENGUAJE DE GRAN TAMAÑO PARA EL MONITOREO ENERGÉTICO EN UN HOGAR INTELIGENTE"**.

El sistema permite a los usuarios interactuar con los datos de consumo energético de su hogar mediante lenguaje natural, utilizando una arquitectura de micro-agentes que interpretan, planean y ejecutan tareas analíticas.

## 🚀 Tecnologías Principales
*   **Lenguaje:** Python 3.12+
*   **Orquestación de LLMs:** [DSPy](https://github.com/stanfordnlp/dspy) (Programación declarativa de agentes).
*   **Interfaz Gráfica:** [PySide6](https://pypi.org/project/PySide6/) (Qt para Python) con estética moderna y asíncrona.
*   **Protocolo de Comunicación:** [Model Context Protocol (MCP)](https://modelcontextprotocol.io/) para el desacoplamiento de herramientas.
*   **Base de Datos:** SQLite para el almacenamiento de históricos y configuraciones locales.
*   **Análisis de Datos:** Pandas, Numpy y Plotly para la generación de gráficas interactivas.

## 📂 Estructura del Proyecto
*   `Agente_energetico/`: Contiene el código fuente principal del sistema (Interfaz, Agentes, Servidor MCP y Controladores).
*   `Pruebas_Inferenciador/` y `Pruebas_planeador/`: Módulos de evaluación y benchmarking del rendimiento de los modelos.

## 📖 Documentación
Para facilitar el uso y la comprensión del sistema, se han creado los siguientes manuales:
*   📄 **[Manual de Usuario](Manual_Usuario.txt):** Guía práctica para instalar, configurar modelos y realizar consultas.
*   📄 **[Documentación Técnica](Documentacion_Agente_Energetico.txt):** Explicación detallada de cada script, flujo de datos y lógica interna del sistema multi-agente.

## 📊 Datos
La base de datos de consumo utilizada para el entrenamiento y pruebas del sistema puede encontrarse en:
[Energy Consumption Dataset - Mendeley Data](https://data.mendeley.com/datasets/y5jjfcfrjz/1)

---
**Autor:** Nicolás D.  
**Contacto:** nicolas2200506@correo.uis.edu.co  
**Institución:** Universidad Industrial de Santander (UIS)

