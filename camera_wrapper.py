import os
import carb
import numpy as np
import torch
import math
from typing import List, Dict, Optional, Sequence, Union

from omni.isaac.orbit.sensors.camera.camera import Camera

import omni.replicator.core as rep
from omni.isaac.orbit.robots.config.anymal import ANYMAL_C_CFG

from omni.isaac.orbit.sensors.camera import Camera, PinholeCameraCfg, FisheyeCameraCfg
from omni.isaac.orbit.sensors.camera.utils import create_pointcloud_from_rgbd
from omni.isaac.orbit.utils import convert_dict_to_backend


# !!!!!! TODO:Find camera parameters for the depth and wide angle camera sensor from ETH  !!!!
ANYMAL_WIDE_ANGLE_CAMERA_CFG = PinholeCameraCfg(
    sensor_tick=1/30.0,
    height=540,
    width=720,
    data_types=["rgb"],
    usd_params=PinholeCameraCfg.UsdCameraCfg(
        focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
    ),
)

ANYMAL_DEPTH_CAMERA_CFG = PinholeCameraCfg(
    sensor_tick=1/30.0,
    height=240,
    width=360,
    data_types=["rgb", "distance_to_camera"],
    usd_params=PinholeCameraCfg.UsdCameraCfg(
        focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
    ),
)

ANYMAL_D_CFG = PinholeCameraCfg(
    sensor_tick=1/30.0,
    height=540,
    width=720,
    data_types=["rgb", "distance_to_camera"],
    usd_params=PinholeCameraCfg.UsdCameraCfg(
        focal_length=1.0, horizontal_aperture=1.25086, clipping_range=(0.1, 1.0e5)
    ),
)


WILL_DEPTH_CFG = PinholeCameraCfg(
    sensor_tick=1/30.0,
    height=540,
    width=720,
    data_types=["rgb", "distance_to_camera"],
    usd_params=PinholeCameraCfg.UsdCameraCfg(
        focal_length=1.0, horizontal_aperture=1.25086, clipping_range=(0.1, 1.0e5)
    ),
)


k_wide_angle_rosbag = [575.6050407221768, 0.0, 745.7312198525915, 0.0, 578.564849365178, 519.5207040671075, 0.0, 0.0, 1.0]
k_depth_rosbag = [423.54608154296875, 0.0, 427.6981506347656, 0.0, 423.54608154296875, 240.177734375, 0.0, 0.0, 1.0]

def ha_from_K(K, w):
    # inverse of Camera.compute_k()
    return .5*math.tan(math.atan(w / (2*K[0])))

ha_wide_angle = ha_from_K(k_wide_angle_rosbag, 1440)
ha_depth = ha_from_K(k_depth_rosbag, 848)

WILL_WIDE_ANGLE_CFG = PinholeCameraCfg(
    sensor_tick=1/30.0,
    height=1080,
    width=1440,
    data_types=["rgb", "distance_to_camera"],
    usd_params=PinholeCameraCfg.UsdCameraCfg(
        focal_length=1.0, horizontal_aperture=ha_wide_angle, clipping_range=(0.1, 1.0e5)
    ),
)

WILL_DEPTH_CFG = PinholeCameraCfg(
    sensor_tick=1/30.0,
    height=480,
    width=848,
    data_types=["rgb", "distance_to_camera"],
    usd_params=PinholeCameraCfg.UsdCameraCfg(
        focal_length=1.0, horizontal_aperture=ha_depth, clipping_range=(0.1, 1.0e5)
    ),
)


class CameraWrapper(Camera):
    """
    TO get LATEST frame use:
        - Camera.data.output
            or 
        - call get_most_recent_frame()
    """

    def __init__(self, cfg: Union[PinholeCameraCfg, FisheyeCameraCfg], device: str = "cpu"):
        """
        cfg is currently just being used to initialize the Camera superclass. It doesn't actually matter;
        All the intrinsics are set by K
        """
        super().__init__(cfg, device)
        
        self.camera_name = "WideAngleCamera"
        self._output_path = None
        self._rep_writer = None
        self._frame_count = 0
        self._orientation_offset = [-0.5, -0.5, 0.5, 0.5]
        


    def spawned_already(self)->bool:
        return self._is_spawned

    def get_most_recent_frame(self):
        return self.data.output

    def get_output_path(self) -> str:
        return self._output_path

    def get_frame_count(self) -> int:
        return self._frame_count

    def get_camera_name(self) -> str:
        if self._output_path is None:
            carb.log_warn("Warning: Trying to get camera name before camera has spawned!!!")
        return self.camera_name

    def spawn(self, parent_prim_path: str, translation: Sequence[float] = None, orientation: Sequence[float] = None):
        orientation = orientation if orientation is not None else self._orientation_offset

        super().spawn(parent_prim_path, translation, orientation)
        self.camera_name = os.path.basename(parent_prim_path)
        self._output_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "output", self.camera_name)
        self.K = self.compute_K()

    def compute_K(self):
        K: torch.tensor
        K = self._compute_intrinsic_matrix()
        K = torch.from_numpy(K)
        K = torch.vstack((K, torch.zeros(3)))
        zeros_column = torch.zeros((4, 1), dtype=K.dtype)
        K = torch.cat((K, zeros_column), axis=1)
        K[3,3] = 1.0
        K = K.float()
        return K

    def get_sensor_local_pose(self):
        """This is a helper method added for setting the spawn translation and location correctly"""

        return self._sensor_xform.get_local_pose()

    def setup_image_data_logger(self, frame_padding=3):
        self._rep_writer = rep.BasicWriter(output_dir=self._output_path, frame_padding=frame_padding)

    def save_latest_data(self, backend="numpy"):
        if self._rep_writer != None:
            ()
            #self._rep_writer.write(convert_dict_to_backend(self.data.output, backend=backend))
        else:
            carb.log_error("Error: Attempting to save latest camera data before setting up image data logger")
