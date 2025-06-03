import threading
import sqlite3

from dataclasses import dataclass
from datetime import datetime


@dataclass
class DetectionEvent:
    # camera_id: int
    frame_index: int
    timestamp: datetime
    track_id: int
    x: float
    y: float


class EventListener:
    def handle_detection(self, event: DetectionEvent):
        raise NotImplementedError


class DatabaseLogger(EventListener):
    def __init__(self, db_path: str):
        self.db_path = db_path
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
            INSERT INTO positions (track_id, timestamp, x, y)
            VALUES (?, ?, ?, ?)
        ''', (event.track_id, event.timestamp, event.x, event.y))

        print("logged")
        conn.commit()
