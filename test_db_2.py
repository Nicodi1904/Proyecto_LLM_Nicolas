import sqlite3

conn = sqlite3.connect(':memory:')
conn.execute("CREATE TABLE test (TimeStamp TEXT)")
conn.execute("INSERT INTO test VALUES ('2024-01-01 00:00:00')")
conn.execute("INSERT INTO test VALUES ('2024-01-01 12:00:00')")
conn.execute("INSERT INTO test VALUES ('2024-01-01 23:59:00')")
conn.execute("INSERT INTO test VALUES ('2024-01-01 23:59:59')")
conn.commit()

q1 = "SELECT TimeStamp FROM test WHERE TimeStamp >= '2024-01-01 00:00' AND TimeStamp <= '2024-01-01 23:59'"
print("BETWEEN 00:00 AND 23:59 ->", conn.execute(q1).fetchall())

# Test with trailing format
q2 = "SELECT TimeStamp FROM test WHERE TimeStamp >= '2024-01-01 00:00' AND TimeStamp <= '2024-01-01 23:59:59'"
print("BETWEEN 00:00 AND 23:59:59 ->", conn.execute(q2).fetchall())

q3 = "SELECT TimeStamp FROM test WHERE TimeStamp BETWEEN '2024-01-01 00:00' AND '2024-01-01 23:59'"
print("Using BETWEEN text text ->", conn.execute(q3).fetchall())

# What about appending :59 in Python dynamically if len == 16?
