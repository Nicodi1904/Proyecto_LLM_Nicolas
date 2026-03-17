import plotly.graph_objects as go

class Worker3:
    """
    Worker 3 se encarga de analizar los resultados puros de 'obtener_consumo'
    (provenientes del reporte_worker3) y generar visualizaciones gráficas
    con Plotly dependiendo de la naturaleza de los datos.
    """
    
    def generar_graficas(self, reporte_worker3: dict) -> dict:
        """
        Recorre el reporte estructurado por IDs de solicitud y crea figuras de Plotly.
        Retorna un diccionario con la misma estructura, pero donde las acciones
        contienen un objeto 'figura' de Plotly si fue posible generarla.
        """
        resultados_graficos = {}

        for req_id, acciones in reporte_worker3.items():
            resultados_graficos[req_id] = []
            
            for accion in acciones:
                accion_con_grafica = accion.copy()
                resultado = accion.get("resultado", {})
                
                if resultado.get("status") == "success":
                    granularidad = resultado.get("granularidad")
                    datos = resultado.get("datos", {})
                    
                    titulo = accion.get("descripcion", f"Consumo Energético ({granularidad})")
                    fig = None

                    # Caso 1: Granularidad "total" -> Gráfico de Barras Simple
                    if granularidad == "total":
                        dispositivos = list(datos.keys())
                        valores = [datos[d] for d in dispositivos]
                        
                        fig = go.Figure(data=[
                            go.Bar(
                                x=dispositivos, 
                                y=valores, 
                                text=valores,
                                textposition='auto',
                                marker_color='#00d2ff'
                            )
                        ])
                        fig.update_layout(
                            title=titulo,
                            xaxis_title="Dispositivo",
                            yaxis_title="Consumo Total (kWh)",
                            template="plotly_dark",
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)',
                            margin=dict(l=20, r=20, t=40, b=20)
                        )

                    # Caso 2: Series de tiempo ("hora", "dia", "mes")
                    elif granularidad in ["hora", "dia", "mes"] and datos:
                        num_dispositivos = len(datos)
                        fig = go.Figure()
                        
                        for i, (dispositivo, valores_temp) in enumerate(datos.items()):
                            # Convertir las claves (fechas ISO) y valores
                            fechas = list(valores_temp.keys())
                            consumos = list(valores_temp.values())
                            
                            # Usar Gráfico de Barras con valores unitarios
                            fig.add_trace(go.Bar(
                                x=fechas, 
                                y=consumos, 
                                name=dispositivo,
                                text=consumos,
                                textposition='auto'
                            ))
                                
                        fig.update_layout(
                            title=titulo,
                            xaxis_title="Tiempo",
                            yaxis_title="Consumo (kWh)",
                            template="plotly_dark",
                            legend_title="Dispositivos",
                            hovermode="x unified",
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)',
                            margin=dict(l=20, r=20, t=40, b=20)
                        )
                    
                    if fig:
                        accion_con_grafica["figura"] = fig
                
                resultados_graficos[req_id].append(accion_con_grafica)

        return resultados_graficos







if __name__ == "__main__":
    # Prueba del Worker 3 con datos simulados
    reporte_ejemplo = {
      "@1": [
        {
          "accion_id": "@1.1",
          "tool": "obtener_consumo",
          "descripcion": "Consumo de ayer Nevera vs PC",
          "resultado": {
            "status": "success",
            "granularidad": "hora",
            "datos": {
              "nevera": {
                "2024-10-23 00:00:00": 0.1,
                "2024-10-23 01:00:00": 0.12,
                "2024-10-23 02:00:00": 0.08,
                "2024-10-23 03:00:00": 0.15
              },
              "PC": {
                "2024-10-23 00:00:00": 0.0,
                "2024-10-23 01:00:00": 0.0,
                "2024-10-23 02:00:00": 0.05,
                "2024-10-23 03:00:00": 0.1
              }
            }
          }
        }
      ],
      "@2": [
        {
          "accion_id": "@2.1",
          "tool": "obtener_consumo",
          "descripcion": "Consumo Total Acumulado",
          "resultado": {
            "status": "success",
            "granularidad": "total",
            "datos": {
              "nevera": 15.4,
              "PC": 8.2,
              "TV": 3.1
            }
          }
        }
      ]
    }

    worker = Worker3()
    reporte_grafico = worker.generar_graficas(reporte_ejemplo)

    # Verificar si se crearon las figuras
    for req, acciones in reporte_grafico.items():
        for acc in acciones:
            print(f"[{acc['accion_id']}] Figura generada: {'figura' in acc}")
            if 'figura' in acc:
                # Para ver la gráfica en el navegador (Descomentar para probar manual)
                acc['figura'].show()
                pass
