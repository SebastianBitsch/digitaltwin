import os
import threading
import sqlite3

from dataclasses import dataclass
from datetime import datetime
from digitaltwin.database.make_db import make_db


@dataclass
class DetectionEvent:
    camera_id: int 
    frame_index: int
    size: float
    timestamp: datetime
    track_id: int # person
    u: float
    v: float
    x: float
    y: float
    zone_id: int = -1

class EventListener:
    def handle_detection(self, event: DetectionEvent):
        raise NotImplementedError


class DatabaseLogger(EventListener):
    def __init__(self, db_path: str):
        self.db_path = db_path
        if not os.path.exists(self.db_path):
            print("Not SQL database found, making one")
            make_db(db_path)
        self.track_seen = set()
        self._local = threading.local()

    def get_cursor(self):
        if not hasattr(self._local, "conn"):
            self._local.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        return self._local.conn.cursor(), self._local.conn

    def handle_detection(self, event: DetectionEvent):
        cursor, conn = self.get_cursor()

        if event.track_id not in self.track_seen:
            cursor.execute('''
                INSERT OR IGNORE INTO people_tracks (id, first_seen, last_seen)
                VALUES (?, ?, ?)
            ''', (event.track_id, event.timestamp, event.timestamp))
            self.track_seen.add(event.track_id)
        else:
            cursor.execute('''
                UPDATE people_tracks SET last_seen = ? WHERE id = ?
            ''', (event.timestamp, event.track_id))

        cursor.execute('''
            INSERT INTO positions (track_id, timestamp, camera_id, u, v, x, y, zone_id, size)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (event.track_id, event.timestamp, event.camera_id, event.u, event.v, event.x, event.y, event.zone_id, event.size))

        conn.commit()
