import json
import glob
import time
import os

import open3d as o3d
import numpy as np
import xml.etree.ElementTree as ET
import cv2

# --- CONFIGURATION ---
CALIB_DIR = "data/processed/Wildtrack_dataset/calibrations"
INTRINSIC_DIR = os.path.join(CALIB_DIR, "intrinsic_zero")
EXTRINSIC_DIR = os.path.join(CALIB_DIR, "extrinsic")
CAMERA_COUNT = 7
CAMERA_COLOR = [0, 0, 1]
PERSON_COLOR = [1, 0, 0]
FRAME_RATE = 10

# --- HELPERS ---
def read_intrinsics(file_path):
    fs = cv2.FileStorage(file_path, cv2.FILE_STORAGE_READ)
    camera_matrix = fs.getNode("camera_matrix").mat()
    dist_coeffs = fs.getNode("distortion_coefficients").mat()
    fs.release()
    return camera_matrix

def read_extrinsics(file_path):
    """
    Read extrinsics and convert to world-to-camera transform for Open3D
    """
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
    
    # Convert to meters (assuming input is in mm)
    tvec = tvec / 1000.0
    
    # Convert rotation vector to rotation matrix
    R, _ = cv2.Rodrigues(rvec)
    
    # Create world-to-camera transformation matrix
    T_wc = np.eye(4)
    T_wc[:3, :3] = R
    T_wc[:3, 3] = tvec
    
    # For visualization, we need camera-to-world (inverse)
    T_cw = np.linalg.inv(T_wc)
    
    # OpenCV uses right-handed coordinate system: X-right, Y-down, Z-forward
    # Open3D uses right-handed coordinate system: X-right, Y-up, Z-out-of-screen
    # Apply coordinate system conversion
    opencv_to_open3d = np.array([
        [1,  0,  0, 0],
        [0, -1,  0, 0],
        [0,  0, -1, 0],
        [0,  0,  0, 1]
    ])
    
    T_final = T_cw @ opencv_to_open3d
    
    return T_final

def create_camera_frustum(K, T, scale=0.5):
    """Create camera frustum for visualization"""
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    
    # Assume standard image dimensions (adjust if different)
    width = 1920
    height = 1080
    
    # Image corners in homogeneous coordinates
    corners_2d = np.array([
        [0, 0, 1],
        [width, 0, 1],
        [width, height, 1],
        [0, height, 1]
    ]).T
    
    # Project to camera coordinates
    K_inv = np.linalg.inv(K)
    corners_3d_cam = K_inv @ corners_2d
    corners_3d_cam *= scale  # Scale for visualization
    
    # Add camera center
    points_cam = np.concatenate([
        np.zeros((3, 1)),  # Camera center
        corners_3d_cam
    ], axis=1)
    
    # Transform to world coordinates
    points_world = (T[:3, :3] @ points_cam + T[:3, 3:4]).T
    
    # Define frustum edges
    lines = [
        [0, 1], [0, 2], [0, 3], [0, 4],  # Center to corners
        [1, 2], [2, 3], [3, 4], [4, 1]   # Rectangle edges
    ]
    colors = [CAMERA_COLOR for _ in lines]
    
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points_world),
        lines=o3d.utility.Vector2iVector(lines)
    )
    line_set.colors = o3d.utility.Vector3dVector(colors)
    
    return line_set

def position_id_to_world(pid, ground_height=0.0):
    """
    Convert position ID to world coordinates
    Based on paper: 480x1440 grid, totaling 691,200 positions
    
    Args:
        pid: Position ID from annotation (0-691199)
        ground_height: Height of the ground plane
    """
    if pid < 0 or pid >= 691200:
        raise ValueError(f"Position ID {pid} out of range [0, 691199]")
    
    x = -3.0 + 0.025 * (pid % 480)
    z = -9.0 + 0.025 * (pid // 480)  # pid // 480 gives row (0-1439)
    y = ground_height  # Ground plane height
    
    return np.array([x, y, z])

def load_annotations(json_file):
    """Load person annotations from JSON file"""
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data

def create_ground_plane():
    """Create a ground plane for reference based on 480x1440 grid"""
    # Grid dimensions from paper: 480x1440 = 691,200 positions
    x_min, x_max = -3.0, -3.0 + 0.025 * 480  # 480 columns
    z_min, z_max = -9.0, -9.0 + 0.025 * 1440  # 1440 rows
    
    print(f"Ground plane: X=[{x_min:.1f}, {x_max:.1f}], Z=[{z_min:.1f}, {z_max:.1f}]")
    
    vertices = np.array([
        [x_min, 0, z_min],
        [x_max, 0, z_min],
        [x_max, 0, z_max],
        [x_min, 0, z_max]
    ])
    
    triangles = np.array([
        [0, 3, 2],
        [0, 2, 1]
    ])
    
    ground = o3d.geometry.TriangleMesh()
    ground.vertices = o3d.utility.Vector3dVector(vertices)
    ground.triangles = o3d.utility.Vector3iVector(triangles)
    ground.paint_uniform_color([0.5, 0.5, 0.5])  # Gray ground
    
    return ground

# --- MAIN ---
def main():
    vis_objects = []
    
    # Add coordinate frame for reference
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
    vis_objects.append(coord_frame)
    
    # Add ground plane
    ground = create_ground_plane()
    vis_objects.append(ground)
    
    # Load and visualize cameras
    print("Loading cameras...")
    for i in range(CAMERA_COUNT):
        intr_path = os.path.join(INTRINSIC_DIR, f"{i}.xml")
        extr_path = os.path.join(EXTRINSIC_DIR, f"{i}.xml")
        
        if not os.path.exists(intr_path) or not os.path.exists(extr_path):
            print(f"Warning: Camera {i} calibration files not found")
            continue
            
        try:
            K = read_intrinsics(intr_path)
            T = read_extrinsics(extr_path)
            cam = create_camera_frustum(K, T, scale=1.0)
            vis_objects.append(cam)
            print(f"Camera {i} loaded successfully")
        except Exception as e:
            print(f"Error loading camera {i}: {e}")
    
    # Load and visualize people
    ANNOTATION_FILE = "data/processed/Wildtrack_dataset/annotations_positions/00000000.json"
    if os.path.exists(ANNOTATION_FILE):
        print("Loading person annotations...")
        detections = load_annotations(ANNOTATION_FILE)
        
        for person in detections:
            pid = person["positionID"]
            pos = position_id_to_world(pid)
            
            # Create person as cylinder (as mentioned in paper)
            # Average human height ~1.7m, radius ~0.2m
            cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=0.2, height=1.7)
            cylinder.paint_uniform_color(PERSON_COLOR)
            
            # Position cylinder with base on ground
            cylinder.rotate(cylinder.get_rotation_matrix_from_xyz((np.pi / 2, 0, 0)), center=(0, 0, 0))
            cylinder.translate(pos + np.array([0, 0.85, 0]))  # Lift by half height
            cylinder.rotate
            vis_objects.append(cylinder)
        
        print(f"Loaded {len(detections)} person detections")
    else:
        print(f"Annotation file not found: {ANNOTATION_FILE}")
    
    # Visualize
    print("Starting visualization...")
    o3d.visualization.draw_geometries(
        vis_objects, 
        window_name="WILDTRACK Dataset Visualization",
        width=1200, 
        height=800
    )

def animate():
    # Create Open3D visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="WILDTRACK Playback", width=1200, height=800)
    
    # Ground + coordinate frame
    ground = create_ground_plane()
    coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
    vis.add_geometry(ground)
    vis.add_geometry(coord)

    # Load camera frustums
    camera_frustums = []
    for i in range(CAMERA_COUNT):
        intr_path = os.path.join(INTRINSIC_DIR, f"{i}.xml")
        extr_path = os.path.join(EXTRINSIC_DIR, f"{i}.xml")
        if not os.path.exists(intr_path) or not os.path.exists(extr_path):
            continue
        try:
            K = read_intrinsics(intr_path)
            T = read_extrinsics(extr_path)
            cam = create_camera_frustum(K, T, scale=1.0)
            camera_frustums.append(cam)
            vis.add_geometry(cam)
        except:
            pass

    # Load all annotation frames
    annotation_files = sorted(glob.glob("data/processed/Wildtrack_dataset/annotations_positions/*.json"))

    person_meshes = []
    for _ in range(30):  # Assume max 30 people
        person = o3d.geometry.TriangleMesh.create_cylinder(radius=0.2, height=1.7)
        person.paint_uniform_color(PERSON_COLOR)
        person.rotate(person.get_rotation_matrix_from_xyz((np.pi / 2, 0, 0)))
        person.translate(np.array([0, 0, 0]))  # Start off-screen
        vis.add_geometry(person)
        person_meshes.append(person)

    # Animate each frame
    for json_path in annotation_files:
        data = load_annotations(json_path)

        for i, mesh in enumerate(person_meshes):
            if i < len(data):
                pos = position_id_to_world(data[i]["positionID"])
                translation = pos + np.array([0, 0.85, 0])
                mesh.translate(translation - np.asarray(mesh.get_center()))
            # else:
            #     # Hide unused meshes far away
            #     mesh.translate(np.array([0, 0, 0]) - np.asarray(mesh.get_center()))
        
        vis.update_geometry(ground)
        vis.update_geometry(coord)
        for cam in camera_frustums:
            vis.update_geometry(cam)
        for person in person_meshes:
            vis.update_geometry(person)

        vis.poll_events()
        vis.update_renderer()
        time.sleep(1 / FRAME_RATE)  # ~20 FPS

    vis.destroy_window()


if __name__ == "__main__":
    # main()
    animate()