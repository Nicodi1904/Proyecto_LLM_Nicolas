import csv
import random
from datetime import datetime, timedelta

# Semilla para replicabilidad
random.seed(42)

# Dispositivos y consumos base
dispositivos = {
    "Nevera": {"standby": 8, "activo": (110, 140)},  # standby + ciclos compresor
    "Lavadora": (500, 1000),
    "Aire acondicionado": (1000, 200),  # media + desviación
    "Televisor": (100, 120),
    "Computador": (150, 250),
    "Bombillos": (8, 12),
    "Microondas": (900, 1100),
}

# Fecha de inicio (ejemplo: hoy a las 00:00)
fecha_inicio = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

with open("consumo_hogar.csv", mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["timestamp", "dispositivo", "consumo_watts"])

    # 96 intervalos de 15 minutos en un día
    for i in range(96):
        timestamp = fecha_inicio + timedelta(minutes=15 * i)

        # --- Nevera ---
        if random.random() < 0.25:  # 25% de prob que el compresor prenda
            consumo = random.randint(*dispositivos["Nevera"]["activo"])
        else:
            consumo = dispositivos["Nevera"]["standby"]
        writer.writerow([timestamp.isoformat(), "Nevera", consumo])

        # --- Lavadora (solo de 9 a 11 am, simulando ciclo completo) ---
        if 9 <= timestamp.hour <= 11:
            consumo = random.randint(*dispositivos["Lavadora"])
        else:
            consumo = 0
        writer.writerow([timestamp.isoformat(), "Lavadora", consumo])

        # --- Aire acondicionado (de 13h a 23h) ---
        if 13 <= timestamp.hour <= 23:
            media, desv = dispositivos["Aire acondicionado"]
            consumo = int(random.gauss(media, desv))  # distribución normal
        else:
            consumo = 0
        writer.writerow([timestamp.isoformat(), "Aire acondicionado", consumo])

        # --- Televisor (18h–23h) ---
        if 18 <= timestamp.hour <= 23:
            consumo = random.randint(*dispositivos["Televisor"])
        else:
            consumo = 0
        writer.writerow([timestamp.isoformat(), "Televisor", consumo])

        # --- Computador (8h–17h) ---
        if 8 <= timestamp.hour <= 17:
            consumo = random.randint(*dispositivos["Computador"])
        else:
            consumo = 0
        writer.writerow([timestamp.isoformat(), "Computador", consumo])

        # --- Bombillos (19h–6h) ---
        if timestamp.hour >= 19 or timestamp.hour <= 6:
            consumo = random.randint(*dispositivos["Bombillos"])
        else:
            consumo = 0
        writer.writerow([timestamp.isoformat(), "Bombillos", consumo])

        # --- Microondas (7h, 12h, 20h, con probabilidad de uso) ---
        if timestamp.hour in [7, 12, 20] and random.random() < 0.3:
            consumo = random.randint(*dispositivos["Microondas"])
        else:
            consumo = 0
        writer.writerow([timestamp.isoformat(), "Microondas", consumo])

print("✅ Dataset realista generado en 'consumo_hogar.csv'")
