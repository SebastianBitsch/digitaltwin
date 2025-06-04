import sqlite3

def make_db(db_path: str):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()

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
        camera_id INTEGER,
        u REAL,
        v REAL,
        x REAL,
        y REAL,
        size REAL,
        zone_id INTEGER,
        FOREIGN KEY(track_id) REFERENCES people_tracks(id)
    )
    ''')

    conn.commit()
    conn.close()
