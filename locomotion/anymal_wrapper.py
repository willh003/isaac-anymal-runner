"""General useful imports"""
import omni
import io
import carb
import torch
import math
import copy
import numpy as np
from typing import List, Dict, Optional, Sequence, Union


from omni.isaac.core.simulation_context import SimulationContext  # used to make sure simulation is running


from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.prims import get_prim_at_path, get_prim_type_name, is_prim_path_valid
from omni.isaac.core.utils.rotations import (
    quat_to_rot_matrix,
    quat_to_euler_angles,
    euler_to_rot_matrix,
    euler_angles_to_quat,
)

"""imports for the leggedrobot base class and the debug visual wrappers"""
from pxr import Gf
from omni.isaac.orbit.robots.legged_robot import LeggedRobot
from omni.isaac.orbit.markers import PointMarker, StaticMarker
from omni.isaac.orbit.utils.assets import ISAAC_ORBIT_NUCLEUS_DIR, ISAAC_NUCLEUS_DIR


from omni.isaac.sensor import LidarRtx
from omni.isaac.sensor import RotatingLidarPhysX
from omni.isaac.core.objects import DynamicCuboid
from omni.isaac.orbit.utils.math import quat_from_euler_xyz
from omni.isaac.orbit.robots.config.anymal import ANYMAL_C_CFG, quat_from_euler_rpy


"""Methods/Classes being locally referenced"""
from camera_wrapper import CameraWrapper, ANYMAL_WIDE_ANGLE_CAMERA_CFG, ANYMAL_DEPTH_CAMERA_CFG
from helper_methods import path_with_respect_to_helper_method,spacer
from omni_isaac_rough_locomotion import OmniLocomotionModel




""" Updated the USD to be the  ANYbotics usd instead of orbit. This resolves the meshes not being properly labeled """

ANYMAL_C_CFG.meta_info.usd_path = path_with_respect_to_helper_method("anymal-assets/AnymalC/anymal_c.usda")


class AnymalC(LeggedRobot):
    """
    Useful methods inherited from LeggedRobot:

           @property
           def data(self) -> LeggedRobotData:

           @property
           def feet_names(self) -> Sequence[str]:
    """

    def __init__(self, ground_prim="/World/GroundPlane", cfg=ANYMAL_C_CFG):
        super().__init__(cfg)
       
        self.cfg = cfg
       
        self._ground_prim = ground_prim
        self.robot_name = "Anymal_C"
        self._prim_path = ""

        # Used to display feet markers
        self._display_feet_marker = False
        self._feet_markers: List[StaticMarker] = list()
        self._feet_contact_markers: List[PointMarker] = list()

        # setting up locomotion to command policy
        self._previous_action = None
        self._command = np.zeros(3)
        self._new_action_count = 0 #used to apply new actions 4 times 



        # height sample point visualizers
        self._ground_height_visualizer= []

        print("MADE IT TO ANYMAL C LAUNCHING")
        carb.log_info("MADE IT TO ANYMAL C LAUNCHING")

        #Locomotion Wrapper
        self.locomotion_wrap = OmniLocomotionModel()

        #Height Scanners
        self.physx_query_interface = omni.physx.get_physx_scene_query_interface()
        self._query_info = []
        self.states = None

        # Sensors Setup/ Configuration

        # 1) Lidar
        self._lidar = None

        # 2) Depth Cams
        self._front_depth_camera = CameraWrapper(ANYMAL_DEPTH_CAMERA_CFG)
        self._rear_depth_camera = CameraWrapper(ANYMAL_DEPTH_CAMERA_CFG)
        self._right_depth_camera = CameraWrapper(ANYMAL_DEPTH_CAMERA_CFG)
        self._left_depth_camera = CameraWrapper(ANYMAL_DEPTH_CAMERA_CFG)

        # 3) Wide angle Cam
        self._front_wide_angle = CameraWrapper(ANYMAL_WIDE_ANGLE_CAMERA_CFG)
        self._rear_wide_angle = CameraWrapper(ANYMAL_WIDE_ANGLE_CAMERA_CFG)
    
 
    def update_height_sample_marker(self,locations):
        
        if len(self._ground_height_visualizer) != 0:
            markers = self._ground_height_visualizer[0]

            markers.set_world_poses(positions=locations.view(-1,3))
            

    def update_feet_markers(self):
        """Assuming that there any feet markers this method udpates the position of them"""
        for foot_marker, contact_marker in zip(self._feet_markers, self._feet_contact_markers):
            # feet
            foot_marker.set_world_poses(
                self.data.feet_state_w[..., 0:3].view(-1, 3), self.data.feet_state_w[..., 3:7].view(-1, 4)
            )
            # contact sensors
            contact_marker.set_world_poses(
                self.data.feet_state_w[..., 0:3].view(-1, 3), self.data.feet_state_w[..., 3:7].view(-1, 4)
            )
            contact_marker.set_status(torch.where(self.data.feet_air_time.view(-1) > 0.0, 1, 2))

    def _spawn_height_sample_position_markers(self):
        
        marker = PointMarker(f"/World/Visuals/{self.robot_name}/Locomotion_height_sample", self.num_height_points, radius=0.05)
        self._ground_height_visualizer.append(marker)
        

    def _spawn_feet_markers(self):
        """Spawns feet markers"""
        marker = StaticMarker(f"/World/Visuals/{self.robot_name}/feet", 4 * self.count, scale=(0.1, 0.1, 0.1))
        self._feet_markers.append(marker)
        # contact
        marker = PointMarker(f"/World/Visuals/{self.robot_name}/feet_contact", 4 * self.count, radius=0.035)
        self._feet_contact_markers.append(marker)


    def _setup_height_sampling_points(self):

        # This method sets up the grid of points that are used by the height scanner
     
        # 1mx1.6m rectangle (without center line)
        y = 0.1 * torch.tensor(
            [-5, -4, -3, -2, -1, 1, 2, 3, 4, 5], device=self.device , requires_grad=False
        )  # 10-50cm on each side

        x = 0.1 * torch.tensor(
            [-8, -7, -6, -5, -4, -3, -2, 2, 3, 4, 5, 6, 7, 8], device=self.device , requires_grad=False
        )  # 20-80cm on each side
        
        grid_x, grid_y = torch.meshgrid(x, y)

        self.num_height_points = grid_x.numel()
        self._scan_points = torch.zeros(self.num_height_points, 3, device=self.device , requires_grad=False)
        self._scan_points[ :, 0] = grid_x.flatten()
        self._scan_points[ :, 1] = grid_y.flatten()

        self.heights = torch.zeros(self.num_height_points,device=self.device ,requires_grad=False)

    def spawn(
        self,
        prim_path: str,
        translation: Sequence[float] = None,
        orientation: Sequence[float] = None,
        with_sensors=False,
    ):
        """spawns the robot using the given prim path and also spawns all sensors"""
        super().spawn(prim_path, translation, orientation)


      
        """Saving the prim_path and the robot_name"""
        self._prim_path = prim_path
        self.robot_name = prim_path.split("/")[-1]

        

        if with_sensors:
            self.spawn_sensors()
    
    def initialize(self, prim_paths_expr: Optional[str] = None):
        output = super().initialize(prim_paths_expr)
        
        self._default_dof_pos, self._default_dof_vel = self.get_default_dof_state()

        # TODO: double check if this optimal location for this 
        self.locomotion_wrap.build(self.device,self._default_dof_pos) 

        
        self._setup_height_sampling_points()
        return output

    def _debug_on_every_frame_calls(self):
        """This is a method for adding debug print outs to be displayed at every advance call"""
        # translation, orientation = self._right_depth_camera.get_sensor_local_pose()

        # print(f'Translation: {translation} , Orientation: {orientation}')

        # print(self._lidar.get_current_frame())
        pass

    def spawn_sensors(self):
        self._front_depth_camera.spawn(f"{self._prim_path}/depth_camera_front_camera")
        self._right_depth_camera.spawn(f"{self._prim_path}/depth_camera_right_camera")
        self._left_depth_camera.spawn(f"{self._prim_path}/depth_camera_left_camera")
        self._rear_depth_camera.spawn(f"{self._prim_path}/depth_camera_rear_camera")

        self._front_wide_angle.spawn(f"{self._prim_path}/wide_angle_camera_front_camera")
        self._rear_wide_angle.spawn(f"{self._prim_path}/wide_angle_camera_rear_camera")

        self._lidar = RotatingLidarPhysX(
            prim_path=f"{self._prim_path}/lidar/rtx_lidar", name="rtx_lidar", translation=np.array([0, 0, 0])
        )

        self._lidar.add_point_cloud_data_to_frame()

        self._lidar.enable_visualization()
        self._lidar.initialize()

    def _update_feet_marker_visibility(self):
        """updates the feet marker visiblity"""
        for foot_marker, contact_marker in zip(self._feet_markers, self._feet_contact_markers):
            foot_marker.set_visibility(self._display_feet_marker)
            contact_marker.set_visibility(self._display_feet_marker)

    def apply_zero_action(self):
        """Applies a sequence of zero actions"""
        actions = torch.zeros(self.count,self.num_actions)

        # print(actions.device,actions)
        self.apply_action(actions.to('cpu'))
        self._previous_action = actions.clone()

    def reset_pose(self) -> None:
        """Method to reset robot default standing up pose"""
       
        self.set_dof_state(self._default_dof_pos, self._default_dof_vel)
        self.reset_buffers()
        self.apply_zero_action()

    def enable_feet_marker(self) -> None:
        """Spawns feet markers if not spawned already and enables the visiblity"""
        self._display_feet_marker = True
        if len(self._feet_markers) == 0:
            self._spawn_feet_markers()
        self._update_feet_marker_visibility()

    def enable_ground_height_sample_marker(self)->None:
        if len(self._ground_height_visualizer) == 0:
            self._spawn_height_sample_position_markers()
        self._ground_height_visualizer[0].set_visibility(True)
    
    def disable_feet_marker(self) -> None:
        """Currently only turns off the visiblity of the feet markers"""
        self._display_feet_marker = False
        self._update_feet_marker_visibility()

    def _hit_report_callback(self, hit):
        """used to get the hit between the groundplane and the sample points"""

        # assert is_prim_path_valid(self._ground_prim)  # Note if this isn't set right locomotion will fail to work
        # print(hit)
        current_hit_body = hit.rigid_body
        if current_hit_body:
            self._query_info.append(hit.distance)
        return True
    
    def _get_heights(self)->None:
        # height_scanner
        self.heights.zero_()

        rpy = -quat_to_euler_angles(self.data.root_quat_w.view(-1).numpy())
        rpy[:2] = 0.0
        yaw_rot = np.array(Gf.Matrix3f(euler_to_rot_matrix(rpy)))
        base_location = self.data.root_pos_w.view(-1).numpy()

        world_scan_points = np.matmul(yaw_rot, self._scan_points.T).T + base_location

      
        # for k,v in zip(['projected_gravity_b','feet_state_w'],[self.data.projected_gravity_b, self.data.feet_state_w]):
        #     print(f'{k}: \n{v}\n')

        for i in range(world_scan_points.shape[0]):
            self._query_info.clear()
            self.physx_query_interface.raycast_all(
                tuple(world_scan_points[i]), (0.0, 0.0, -1.0), 100, self._hit_report_callback
            )
            if self._query_info:
                distance = min(self._query_info)
            
                self.heights[i] = np.clip(distance, -1.0, 1.0)
            else:

                print("No hit")
        
        world_scan_points[:,2] = self.heights
        self.update_height_sample_marker(world_scan_points)
        

    def _compute_observation_and_action(self, command):
        """[summary]

        compute the observation vector for the policy

        Argument:
        command {np.ndarray} -- the robot command (v_x, v_y, w_z)

        Returns:
        torch.Tensor -- The observation vector.

        """

        assert self.locomotion_wrap is not None
        
        self._get_heights()

        return self.locomotion_wrap.get_action(self.data,self.heights,torch.from_numpy(command))

    def advance(self, dt, command):
        
        print(dt,command)
        # self.update_buffers(dt) # TODO THIS MIGHT BE TOO MANY CALLS 

        # self._debug_on_every_frame_calls()  # TODO make this into a flag

        if self.locomotion_wrap is not None:
        
            self._command = np.array(command)

            with torch.no_grad():
                new_action = self._compute_observation_and_action(self._command)

            print(new_action)
            self.reset_pose()     
            self.apply_action(new_action.to('cpu'))
            self._previous_action = new_action.clone()
            self.update_feet_markers()
            self._new_action_count += 1

        else:
            carb.log_error("\nATTEMPTING TO ADVANCE ANYMAL WITHOUT SPAWNING A LOCOMOTION WRAPPER INSTANCE!!!\n")
