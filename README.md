# Multi-Source Object Tracking and Visualization System

This project is a modular Python-based system for object tracking, data logging, and live visualization. It supports streaming from multiple video sources, performs real-time detection and tracking using YOLO and BoT-SORT, logs results to a local database, and generates live-updating plots of tracked positions.

The data can be queried through an LLM to get a quick grasp of what has happened on the streams 

The system is designed with a consistent interface for all stream types and a composable architecture that makes it easy to extend or modify individual components.

---

## Overview

The goal of the project is to build a reusable and extensible pipeline for computer vision tasks involving object tracking, data storage, and visualization. This could be for CCTV, crowd control. 

The main goal is to make visual data from multiple sources more manageable at a glance by having a LLM to query the data with

### Use Cases

* Tracking and analyzing movement in retail spaces or public areas
* Lightweight video analytics for robotics or edge devices
* Visualizing activity over time for research or experimentation
* Prototyping real-time systems with computer vision backends

---

## Features

* Unified `Streamer` interface for cameras, videos, image directories, and plots
* Real-time tracking with Ultralytics YOLOv11 and BoT-SORT
* LLM interface to query about detections in the streams
* Position and ID logging to a local SQLite database
* Modular event-based logging architecture
* Live heatmap visualization of recent detections
* MJPEG frame output for web or GUI integration

---

## Components

### Streamers

Each `Streamer` subclass implements Python's iterator protocol and produces MJPEG-compatible byte frames:

* `CameraStreamer`: Streams from a webcam
* `VideoStreamer`: Reads frames from a video file
* `DirectoryStreamer`: Streams sorted images from a directory
* `PlotStreamer`: Streams matplotlib plots that update periodically
* `YoloStreamer`: Wraps another streamer and applies object detection and tracking

### Trackers

The `YoloStreamer` class runs YOLOv11 for detection and BoT-SORT for identity tracking. It can annotate frames and pass detection events to listeners.

### Database Logger

Implements a listener interface that logs each detection to a SQLite database. It maintains a `people_tracks` table for tracking IDs and a `positions` table for storing spatial-temporal data.

### Visualization

`PlotStreamer` queries the `positions` table and renders a heatmap using matplotlib, showing where tracked objects have recently been located in the scene.

---

## Example Usage

Tracking from a video file and logging to the database:

```python
import sqlite3
streamer = YoloStreamer(VideoStreamer("video.mp4"), "../models/yolo11m.pt")
streamer.add_listener(DatabaseLogger(sqlite3.connect("people.db")))

for frame in streamer:
    # send frame as MJPEG
    ...
```

Displaying a live heatmap of tracked positions:

```python
streamer = PlotStreamer(id=0, db_path="people.db")

for frame in streamer:
    # stream or display frame
    ...
```
