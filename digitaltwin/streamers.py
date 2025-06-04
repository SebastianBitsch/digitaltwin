import os
import io
import time
import sqlite3
from datetime import datetime, timedelta, UTC
from abc import ABC, abstractmethod

import cv2
import matplotlib
matplotlib.use('Agg') # mac bullshit
import matplotlib.pyplot as plt
import numpy as np
from natsort import natsorted
# from ultralytics import solutions

from digitaltwin.database.db_logger import EventListener, DetectionEvent
from digitaltwin.objects import Camera


class Streamer(ABC):

    def __init__(self, id: int):
        self.id = id

    @abstractmethod
    def __iter__(self):
        pass

    @abstractmethod
    def __next__(self):
        pass


class CameraStreamer(Streamer):
    
    def __init__(self, id: int, cam_id: int = 0):
        super().__init__(id=id)
        self.cam_id = cam_id
        self.cap = cv2.VideoCapture(cam_id)

    def __iter__(self):
        self.index = 0  # Reset for fresh iteration
        return self

    def __next__(self):
        success, frame = self.cap.read()
        if success:
            return frame

        else:
            print("Frame failed")
            raise StopIteration


class DirectoryStreamer(Streamer):

    def __init__(self, id: int, images_dir: str) -> None:
        super().__init__(id=id)
        if not os.path.isdir(images_dir):
            raise FileNotFoundError(f"Directory for camera not found: {images_dir}")
        
        image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        self.image_paths: list[str] = natsorted([os.path.join(images_dir, f) for f in image_files])
        self.index = 0

    def __iter__(self):
        self.index = 0
        return self

    def __next__(self):
        if self.index >= len(self.image_paths):
            raise StopIteration

        frame_path = self.image_paths[self.index]
        self.index += 1

        frame = cv2.imread(frame_path)
        if frame is None:
            return self.__next__()  # Skip unreadable frame

        return frame



class VideoStreamer(Streamer):

    def __init__(self, id: int, video_path: str) -> None:
        super().__init__(id=id)
        if not os.path.isfile(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)

        if not self.cap.isOpened():
            raise IOError(f"Failed to open video file: {video_path}")


    def __iter__(self):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # Rewind to start if needed
        return self


    def __next__(self):
        success, frame = self.cap.read()
        if not success:
            raise StopIteration
        return frame


    def __del__(self):
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()


class HeatmapStreamer(Streamer):
    def __init__(self, id: int, db_path: str, im_size, interval_sec: float = 1.0):
        self.id = id
        self.im_size = im_size
        self.interval_sec = interval_sec
        self.db_path = db_path
        self.last_update = 0
        self.cached_frame = None  # store last generated frame

    def __iter__(self):
        return self

    def __next__(self):
        now = time.time()
        if self.cached_frame is None or (now - self.last_update) >= self.interval_sec:
            self.last_update = now
            self.cached_frame = self._generate_plot_frame()
        return self.cached_frame


    def _generate_plot_frame(self):
        # Connect to DB and query positions from last N seconds
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        ts_cutoff = datetime.now(UTC) - timedelta(seconds=30) # todo
        cursor.execute("""
            SELECT u, v FROM positions
            WHERE camera_id = ? AND timestamp >= ?
        """, (0, ts_cutoff, ))
        data = cursor.fetchall()
        conn.close()

        if not data:
            x, y = np.array([]), np.array([])
        else:
            x, y = zip(*data)
            x, y = np.array(x), np.array(y)

        fig, ax = plt.subplots(figsize=(6.4, 3.6)) # 16/9 aspect 

        if len(x) > 0:
            ax.scatter(x,y)
            # hb = ax.hexbin(x, y, gridsize=30, cmap='inferno', extent=(0, 1920, 0, 1080))
            # fig.colorbar(hb, ax=ax)
        else:
            ax.text(0.5, 0.5, "No data", ha='center', va='center', fontsize=12)
        
        ax.set_title("Heatmap of Positions in imagespace from Camera 1")
        ax.set_xlim(0, self.im_size[0])
        ax.set_ylim(self.im_size[1], 0)

        # Convert to image
        buf = io.BytesIO()
        fig.savefig(buf, format='jpg')
        buf.seek(0)
        plt.close(fig)

        nparr = np.frombuffer(buf.read(), np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        return frame


class YOLOStreamer(Streamer):
    def __init__(self, base_streamer: Streamer, model, camera: Camera = None, zones: list = None, conf: float = 0.3, tracker: str = "botsort.yaml"):
        super().__init__(id=base_streamer.id)
        self.base_streamer = base_streamer
        self.model = model
        self.camera = camera
        self.conf = conf
        self.tracker = tracker
        self.listeners: list[EventListener] = []
        self.frame_idx = 0
        # if zones is not None:
        #     self.zones = [
        #         solutions.TrackZone(
        #             show = False,
        #             region = z,
        #             model = "models/yolo11n.pt" # TODO
        #         )
        #         for z in zones
        #     ]
        #     print("ZEONES", self.zones)
        # else:
        #     self.zones = None


    def add_listener(self, listener: EventListener):
        self.listeners.append(listener)


    def notify_listeners(self, event: DetectionEvent):
        for listener in self.listeners:
            listener.handle_detection(event)


    def __iter__(self):
        self.frame_iter = iter(self.base_streamer)
        self.frame_idx = 0
        return self

    def __next__(self):
        frame = next(self.frame_iter)

        ts = datetime.now()
        results = self.model.track(
            frame, 
            conf=self.conf, tracker=self.tracker, 
            persist=False, stream=False, verbose=False,
            classes=[0] # people
        )
        # zone_finds = {}
        # # shit code sorry
        # if self.zones is not None:
        #     for zone_id, zone in enumerate(self.zones):
        #         zone_results = zone(frame)
        #         print("res", zone_results)
        #         if zone.trackzone.track_ids:
        #             for r in zone_results:
        #                 print("r", r, zone_id)
        #                 zone_finds[r] = zone_id

        # Get annotated image from the results
        annotated_frame = results[0].plot()

        if results[0].boxes.id is not None:
            for i, (box, track_id) in enumerate(zip(results[0].boxes.xyxy, results[0].boxes.id)):
                x1, y1, x2, y2 = box.tolist()
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

                xy = self.camera.project_2d(cx,cy) if self.camera is not None else (0.0, 0.0)

                event = DetectionEvent(
                    camera_id=self.id,
                    frame_index=self.frame_idx,
                    timestamp=ts,
                    track_id=int(track_id),
                    u=cx,
                    v=cy,
                    x=xy[0],
                    y=xy[1],
                    zone_id=-1, # zone_finds.get(i, -1), # sorry
                    size=float((x2 - x1) * (y2 - y1)),
                )
                self.notify_listeners(event)

        self.frame_idx += 1
        return annotated_frame
