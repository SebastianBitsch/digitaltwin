import os
from abc import ABC, abstractmethod

import cv2
from natsort import natsorted


class Streamer(ABC):

    @abstractmethod
    def __iter__(self):
        pass

    @abstractmethod
    def __next__(self):
        pass

    def to_bytes(self, frame):
        _, buffer = cv2.imencode('.jpg', frame)
        return (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n'
        )

class CameraStreamer(Streamer):
    
    def __init__(self, cam_id: int = 0):
        self.cam_id = cam_id
        self.cap = cv2.VideoCapture(cam_id)

    def __iter__(self):
        self.index = 0  # Reset for fresh iteration
        return self

    def __next__(self):
        success, frame = self.cap.read()
        if success:
            return self.to_bytes(frame)

        else:
            raise StopIteration


class DirectoryStreamer(Streamer):

    def __init__(self, images_dir: str) -> None:
        super().__init__()
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

        return self.to_bytes(frame)



class VideoStreamer(Streamer):

    def __init__(self, video_path: str) -> None:
        super().__init__()
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
        return self.to_bytes(frame)


    def __del__(self):
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()


from ultralytics import YOLO
import numpy as np

class YOLOStreamer(Streamer):
    def __init__(self, base_streamer: Streamer, model, conf: float = 0.3, tracker: str = "botsort.yaml"):
        self.base_streamer = base_streamer
        self.model = model
        self.conf = conf
        self.tracker = tracker

    def __iter__(self):
        self.stream_iter = iter(self.base_streamer)
        return self

    def __next__(self):
        # Get next raw frame from base_streamer (as jpeg-encoded bytes)
        frame_bytes = next(self.stream_iter)  # May raise StopIteration

        # Decode JPEG bytes back into numpy image
        jpeg_bytes = frame_bytes.split(b'\r\n\r\n', 1)[1].rsplit(b'\r\n', 1)[0]
        nparr = np.frombuffer(jpeg_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Run YOLOv8 tracking
        results = self.model.track(frame, conf=self.conf, tracker=self.tracker, persist=False, stream=False, verbose=False)

        # Get the annotated image from the results
        annotated_frame = results[0].plot()

        return self.base_streamer.to_bytes(annotated_frame)
