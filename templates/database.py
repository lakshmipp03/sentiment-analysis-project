import sqlite3

# Connect database
conn = sqlite3.connect("history.db")

# Cursor
cursor = conn.cursor()

# Create table
cursor.execute("""

CREATE TABLE IF NOT EXISTS history (

    id INTEGER PRIMARY KEY AUTOINCREMENT,

    review TEXT,

    prediction TEXT,

    emotion TEXT

)

""")

# Save changes
conn.commit()

# Close connection
conn.close()

print("Database Created Successfully")