import sqlite3

conn = sqlite3.connect(r'C:\sqlite_tesis\Base_datos_tesis\Hogar_Sincelejo.db')

q = """SELECT TimeStamp FROM "Energy Consumption in KWh of a Typical House Sincelejo Colombia" 
WHERE TimeStamp BETWEEN '2024-01-01T00:00' AND '2024-01-01T23:59'"""

print("Result of between with T:")
res = conn.execute(q).fetchall()
print(f"Count: {len(res)}")

q2 = """SELECT TimeStamp FROM "Energy Consumption in KWh of a Typical House Sincelejo Colombia" 
WHERE TimeStamp BETWEEN '2024-01-01 00:00' AND '2024-01-01 23:59'"""

print("\nResult of between with space:")
res2 = conn.execute(q2).fetchall()
print(f"Count: {len(res2)}")
