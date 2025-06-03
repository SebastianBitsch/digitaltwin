import open3d as o3d
import numpy as np
import xml.etree.ElementTree as ET
import cv2
import json
import glob
import os

# --- CONFIGURATION ---
CALIB_DIR = "data/processed/Wildtrack_dataset/calibrations"
INTRINSIC_DIR = os.path.join(CALIB_DIR, "intrinsic_zero")
EXTRINSIC_DIR = os.path.join(CALIB_DIR, "extrinsic")
ANNOTATION_FILE = "data/processed/Wildtrack_dataset/annotations_positions/00000000.json"

CAMERA_COUNT = 7
CAMERA_COLOR = [0, 0, 1]
PERSON_COLOR = [1, 0, 0]

# --- HELPERS ---
def read_intrinsics(file_path):
    fs = cv2.FileStorage(file_path, cv2.FILE_STORAGE_READ)
    camera_matrix = fs.getNode("camera_matrix").mat()
    dist_coeffs = fs.getNode("distortion_coefficients").mat()
    fs.release()
    return camera_matrix#, dist_coeffs


def read_extrinsics(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()
    
    # Extract rvec and tvec
    rvec_element = root.find('rvec')
    tvec_element = root.find('tvec')
    
    if rvec_element is None or tvec_element is None:
        raise ValueError("Could not find rvec or tvec in XML file")
    
    # Convert text to numpy arrays
    rvec = np.fromstring(rvec_element.text.strip(), sep=' ')
    tvec = np.fromstring(tvec_element.text.strip(), sep=' ')

    # Convert rotation vector to rotation matrix
    # R, _ = cv2.Rodrigues(rvec)
    # # return R, tvec
    tvec /= 1000.0
    # T = np.eye(4)
    # T[:3, :3] = R
    # T[:3, 3] = tvec
    R, _ = cv2.Rodrigues(rvec)
    Rt = np.eye(4)
    Rt[:3, :3] = R
    Rt[:3, 3] = tvec

    # Invert to get camera-to-world transform
    Rt_inv = np.linalg.inv(Rt)

    # Apply OpenCV to Open3D axis correction (X, -Y, -Z)
    flip = np.diag([1, -1, -1, 1])
    T = Rt_inv @ flip

    # GLOBAL ROTATION: Rotate camera rig down by -90 deg around X (clockwise)
    rot_x_neg_90 = np.eye(4)
    theta = -np.pi / 2
    rot_x_neg_90[1, 1] = np.cos(theta)
    rot_x_neg_90[1, 2] = -np.sin(theta)
    rot_x_neg_90[2, 1] = np.sin(theta)
    rot_x_neg_90[2, 2] = np.cos(theta)

    T = rot_x_neg_90 @ T

    return T


def create_camera_frustum(K, T, scale=0.5):
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    width = 1920
    height = 1080

    # Project corners in image plane to camera coordinates
    corners = np.array([
        [0, 0, 1],
        [width, 0, 1],
        [width, height, 1],
        [0, height, 1]
    ])
    rays = np.linalg.inv(K) @ corners.T
    rays *= scale

    # Add camera origin
    rays = np.concatenate([np.zeros((3, 1)), rays], axis=1)

    # Transform to world
    rays = T[:3, :3] @ rays + T[:3, 3:4]

    # Create lines
    lines = [[0, i] for i in range(1, 5)] + [[1, 2], [2, 3], [3, 4], [4, 1]]
    colors = [CAMERA_COLOR for _ in lines]

    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(rays.T),
        lines=o3d.utility.Vector2iVector(lines)
    )
    line_set.colors = o3d.utility.Vector3dVector(colors)
    return line_set

def position_id_to_world(pid):
    x = -3.0 + 0.025 * (pid % 480)
    y = -9.0 + 0.025 * (pid // 480)
    return np.array([x, 0.0, y])

def load_annotations(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data

# --- MAIN ---
vis_objects = []

# Load and visualize cameras
for i in range(CAMERA_COUNT):
    intr_path = os.path.join(INTRINSIC_DIR, f"{i}.xml")
    extr_path = os.path.join(EXTRINSIC_DIR, f"{i}.xml")

    K = read_intrinsics(intr_path)
    T = read_extrinsics(extr_path)

    cam = create_camera_frustum(K, T, scale=1.0)
    vis_objects.append(cam)

# Load and visualize people
detections = load_annotations(ANNOTATION_FILE)
for person in detections:
    pid = person["positionID"]
    pos = position_id_to_world(pid)
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.1)
    sphere.paint_uniform_color(PERSON_COLOR)
    sphere.translate(pos)
    vis_objects.append(sphere)

# Start Open3D viewer
vis_objects.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0))
o3d.visualization.draw_geometries(vis_objects)
