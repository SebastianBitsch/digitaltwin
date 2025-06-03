import sqlite3

conn = sqlite3.connect("data/tracking.db")
c = conn.cursor()

# Create tables
c.execute('''
CREATE TABLE IF NOT EXISTS people_tracks (
    id INTEGER PRIMARY KEY,
    first_seen TIMESTAMP,
    last_seen TIMESTAMP
)
''')

c.execute('''
CREATE TABLE IF NOT EXISTS positions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    track_id INTEGER,
    timestamp TIMESTAMP,
    x REAL,
    y REAL,
    FOREIGN KEY(track_id) REFERENCES people_tracks(id)
)
''')

conn.commit()
