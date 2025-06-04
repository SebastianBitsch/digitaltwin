import os
from datetime import datetime
from abc import ABC, abstractmethod

import cv2
from natsort import natsorted

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
            raise StopIteration


class DirectoryStreamer(Streamer):

    def __init__(self, id: int, images_dir: str) -> None:
        super().__init__(id=id)
        if not os.path.isdir(images_dir):
            raise FileNotFoundError(f"Directory for camera not found: {images_dir}")
        
        image_files = [f for f in os.listdir(images_dir) if f.endswith('.png')]
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


class YOLOStreamer(Streamer):
    def __init__(self, base_streamer: Streamer, model, camera: Camera = None, conf: float = 0.3, tracker: str = "botsort.yaml"):
        super().__init__(id=base_streamer.id)
        self.base_streamer = base_streamer
        self.model = model
        self.camera = camera
        self.conf = conf
        self.tracker = tracker
        self.listeners: list[EventListener] = []
        self.frame_idx = 0


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
            persist=False, stream=False, verbose=False
        )

        # Get annotated image from the results
        annotated_frame = results[0].plot()

        if results[0].boxes.id is not None:
            for box, track_id in zip(results[0].boxes.xyxy, results[0].boxes.id):
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
                    zone_id=-1,
                    size=float((x2 - x1) * (y2 - y1)),
                )
                self.notify_listeners(event)

        self.frame_idx += 1
        return annotated_frame
