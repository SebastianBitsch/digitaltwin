import os
import sys
from contextlib import contextmanager

import cv2

def to_bytes(stream): # type Streamer
    for frame in stream:
        _, buffer = cv2.imencode('.jpg', frame)
        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n'
        )


@contextmanager
def quiet():
    """
    Temporarily redirect stdout and stderr to devnull to suppress OpenCV's warnings.
    Otherwise OpenCV prints an error (`OpenCV: out device of bound (0-1): 2`, etc.) 
    whenever we try to access a port or camera that isnt available.
    This is (i think) the nicest way of supressing that.
    """
    try:
        devnull = os.open(os.devnull, os.O_WRONLY)
        old_stderr = os.dup(2)
        sys.stderr.flush()
        os.dup2(devnull, 2)
        os.close(devnull)
        try:
            yield
        finally:
            os.dup2(old_stderr, 2)
            os.close(old_stderr)
    except Exception:
        yield # If any of the OS-level operations fail, fall back to doing nothing


def list_cameras(max_ports: int = 10) -> list[dict]:
    """ Returns list of available camera ports and the size of their images """
    available = []
    for port in range(max_ports):
        with quiet():
            cap = cv2.VideoCapture(port)
            if not cap.isOpened():
                continue
            
            ret, frame = cap.read()
            if not ret:
                continue
            
            available.append({
                "port" : port,
                "im_h" : frame.shape[0],
                "im_w" : frame.shape[1],
                "im_c" : frame.shape[2] if 2 < frame.ndim else 1,
            })
            cap.release()

    return available


if __name__ == "__main__":
    cameras = list_cameras()
    print(f"Available cameras found {len(cameras)}: {[cam for cam in cameras]}")