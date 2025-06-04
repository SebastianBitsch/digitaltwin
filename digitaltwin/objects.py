import json
import os
import yaml
from dataclasses import dataclass, asdict, field

import numpy as np
import matplotlib.pyplot as plt
from cv2 import Rodrigues

@dataclass
class Chessboard:
    n_cols: int
    n_rows: int
    cell_size: int = 1 # in mm

    @property
    def corners(self) -> np.ndarray:
        """
        Generate a numpy array with shape (rows * cols, 3) where each row represents a 3D coordinate
        in a grid pattern that matches the classic openCV chessboard corners. This represents the true
        geometry of the chessboard that we can compare images to. Third column is all 0. i.e. it lies flat on the floor
        Chessboard cell size only matters if doing extrinsic calibration.
        """
        return self.cell_size * np.array([(i % self.n_cols, i // self.n_cols, 0) for i in range(self.n_rows * self.n_cols)], dtype=np.float32)


@dataclass
class Camera:
    """ """
    name: str

    im_w: int
    im_h: int
    camera_model: str = "unknown"
    distortion_model: str = "rational" # 8 coeffs
    frame_rate: int = 60

    R_matrix: np.ndarray = field(default_factory = lambda: np.eye(3))
    t_vector: np.ndarray = field(default_factory = lambda: np.zeros(3))

    intrinsic_matrix: np.ndarray = field(default_factory = lambda: np.zeros((3, 3)))  # fx, 0, cx; 0, fy, cy; 0, 0, 1
    distortion_coeffs: np.ndarray = field(default_factory = lambda: np.zeros(8))      # k1, k2, p1, p2, k3, k4, k5, k6

    @property
    def R_vector(self) -> np.ndarray:
        """ Return the (3,) rotation vector from the (3,3) rotation matrix """
        return Rodrigues(self.R_matrix)[0].reshape(-1)

    @property
    def homogenous_transformation_matrix(self) -> np.ndarray:
        """ Return the 4x4 rotation matrix with no rotation """
        out = np.eye(4)
        out[:3, :3] = self.R_matrix
        return out

    @property
    def extrinsic_matrix(self) -> np.ndarray:
        """ Return a (3,4) extrinsic matrix of the camera / transformation matrix """
        return np.column_stack([self.R_matrix, self.t_vector.T])
    
    @property
    def projection_matrix(self) -> np.ndarray:
        return self.intrinsic_matrix @ self.extrinsic_matrix

    @property
    def focal(self) -> np.ndarray:
        """ (2,) array """
        return np.diagonal(self.intrinsic_matrix)[:2]
    
    @property
    def center(self) -> np.ndarray:
        return self.intrinsic_matrix[:2, 2]


    def project_2d(self, u: float, v: float) -> tuple[float, float]:
        """
        Projects 2D image coordinates (u, v) to world (x, y) assuming z=0 ground plane.
        Returns (x, y) in world coordinates.
        """
        # Inverse intrinsic
        K_inv = np.linalg.inv(self.intrinsic_matrix)

        # Image point in homogeneous coords
        uv1 = np.array([u, v, 1.0])

        # Ray in camera coordinates
        ray_camera = K_inv @ uv1  # shape (3,)

        # Transform ray into world coordinates
        ray_world = self.R_matrix.T @ ray_camera
        cam_origin_world = -self.R_matrix.T @ self.t_vector.reshape(3)

        # Line: origin + s * direction
        # Intersect with z=0 plane in world frame
        s = -cam_origin_world[2] / ray_world[2]
        world_point = cam_origin_world + s * ray_world

        return float(world_point[0]), float(world_point[1])


    def to_json(self, file_name: str) -> None:
        """ Serialize to JSON """
        if os.path.sep in file_name:
            os.makedirs(os.path.dirname(file_name), exist_ok=True)
        with open(file_name, 'w', encoding='utf-8') as file:
            json.dump(asdict(self), file, ensure_ascii=False, indent=4, cls=Numpy2ListEncoder)


    @classmethod
    def from_json(cls, file_name: str):
        """ Deserialize from JSON """
        with open(file_name, encoding='utf-8') as file:
            data = json.load(file, cls=List2NumpyDecoder)
        return cls(**data)


    def plot(self, ax: plt.Axes, scale: float = 1.0, color: str = 'tab:blue') -> np.ndarray:
        """
        Plot a camera as the classic viewport wireframe icon (as in blender etc. i dont know the name)
        With a origin and a cone marking a square in the fov and small triangle pointing up
            Z-axis pointing forward (towards the scene, with positive depth)
            Y-axis pointing down (note it is plotted as pointing -y)
            X-axis pointing to the right

        Returns the camera position in world space
        """
        fov_x = np.rad2deg(2 * np.arctan2(self.im_w, 2 * self.focal[0]))
        fov_y = np.rad2deg(2 * np.arctan2(self.im_h, 2 * self.focal[1]))

        f = scale
        w = 2 * f * np.tan(np.deg2rad(fov_x) / 2)
        h = 2 * f * np.tan(np.deg2rad(fov_y) / 2)

        # Construct points
        ps_c = np.array([
            [   0,         0,   0], # 0 Origin 
            [ w/2,       h/2,   f], # 1 Bottom right coner
            [ w/2,      -h/2,   f], # 2 Top right corner
            [-w/2,      -h/2,   f], # 3 Top left corner
            [-w/2,       h/2,   f], # 4 Bottom left corner
            [   0,   h/2+h/4,   f]  # 5 Triangle up direction indicator
        ])

        # Convert to world coordinates
        # Negate the translation vector for visualization to reflect actual camera position in the world
        R_world = self.R_matrix.T  # Inverse of rotation (transpose for orthogonal matrices)
        t_world = -self.R_matrix.T @ self.t_vector  # Camera center in world coordinates

        # Transform points from camera to world coordinates
        ps_w = np.array([(R_world @ (scale * p)) + t_world for p in ps_c])

        # Construct the lines
        L01   = np.array([ps_w[0], ps_w[1]])
        L02   = np.array([ps_w[0], ps_w[2]])
        L03   = np.array([ps_w[0], ps_w[3]])
        L04   = np.array([ps_w[0], ps_w[4]])
        L1234 = np.array([ps_w[1], ps_w[2], ps_w[3], ps_w[4], ps_w[1]])
        L154  = np.array([ps_w[1], ps_w[5], ps_w[4]])

        ax.plot(L01[:,   0],   L01[:, 1],   L01[:, 2], "-", color=color)
        ax.plot(L02[:,   0],   L02[:, 1],   L02[:, 2], "-", color=color)
        ax.plot(L03[:,   0],   L03[:, 1],   L03[:, 2], "-", color=color)
        ax.plot(L04[:,   0],   L04[:, 1],   L04[:, 2], "-", color=color)
        ax.plot(L1234[:, 0], L1234[:, 1], L1234[:, 2], "-", color=color)
        ax.plot(L154[:,  0],  L154[:, 1],  L154[:, 2], "-", color=color)

        return ps_w[0] # return camera position


class Numpy2ListEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
    
class List2NumpyDecoder(json.JSONDecoder):
    def decode(self, s):
        def convert(obj):
            if isinstance(obj, list):
                return np.asarray(obj)
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            else:
                return obj
            
        return convert(super().decode(s))

