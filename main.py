# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from typing import Tuple
from omni.isaac.kit import SimulationApp

simulation_app = SimulationApp({"headless": False})

# Omni Imports
from omni.isaac.core import World
from omni.isaac.quadruped.robots import Anymal
from omni.isaac.core.utils.prims import define_prim, get_prim_at_path
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.usd import set_prop_val
from pxr import Gf, UsdGeom, Sdf
import omni.appwindow  # Contains handle to keyboard

# Standard imports
import os
import dataclasses
import numpy as np
from PIL import Image
import cv2
import carb

# Stage Builder Imports
from isaac_stage.utils import open_stage, save_stage, get_context, get_stage
from isaac_stage.prims import create_scope, get_pose, delete, is_defined, transform


# Camera Imports (Sean)
from camera_wrapper import CameraWrapper, ANYMAL_D_CFG

# Planning Imports
from ws_isaac_planner.planner import PlanningAgent

# General 
from states import RobotState, LearningState
from config import SimConfig, StageConfig, DebugConfig
from utils import *


class AnymalRunner(object):
    """This class contains the necessary components to run Isaac Sim with the Orbit Framework."""
    _anymal_direct : Anymal
    _world : World
    physics_dt : float

    def __init__(self, physics_dt=1/200, render_dt=1/60) -> None:
        """Creates the simulation world with preset physics_dt and render_dt.an anymal robot

        Args:
            physics_dt (float): Physics downtime of the scene (milliseconds)
            render_dt  (float): Render downtime of the scene (milliseconds)

        NOTE:
            All callbacks need to have the step_size argument, even if unused. 
        """
        self.prepare_stage()
        self._world = World(stage_units_in_meters=1.0, physics_dt=physics_dt, rendering_dt=render_dt, backend="numpy") #instance SimulationContext
        self.physics_dt = physics_dt

        self.spawn_anymal()
        self.define_fields()

    def prepare_stage(self):
        """Opens a stage for simulation if it exists, otherwise a new one is generated."""
        create_scope(DebugConfig.debug_prim_directory)
        
        if StageConfig.default_stage:
            open_stage(StageConfig.default_stage)
        else:
            StageConfig.stage_builder.build_stage(spawn_assets=StageConfig.spawn_assets, global_offset=(0,0,0), asset_density=StageConfig.asset_density)

    def define_fields(self):
        self._base_command = np.array([0.0,0.0,0])
        #self.log_dir = unique_log_dir(SimConfig.log_dir)
        self.lock = 0
        self.reset_count = 0
                
        # Data classes
        self.robot_state = RobotState()
        # Initialize traversability generator to process velocity commands
        self.planner = PlanningAgent(start=(0,0), goal=(0,0), env_dim=[SimConfig.env_x, SimConfig.env_y], 
                                     edge_radius = 1, draw_markers = SimConfig.draw_markers) 
        
        self.planner.random_new_goal()

        # Data collection
        self.recording = True

    def spawn_anymal(self):
        # spawn anymal
        self.anymal_prim_path = "/World/AnymalC/Robot_1"
        self._anymal_direct = Anymal(prim_path=self.anymal_prim_path,
                name="Anymal",
                usd_path=get_assets_root_path() + "/Isaac/Robots/ANYbotics/anymal_c.usd",
                position=np.array([0, 0, 0]),)
        self._anymal = self._world.scene.add(self._anymal_direct)   

        orientation = quat_of_euler([68.92165,-6.36384, -85.90091], rotation_order='xyz')
        orientation = orientation[[3,0,1,2]] # NOTE: camera.spawn convention is wxyz, quat_of_euler returns xyzw

        # set up camera
        self._anymal_direct.front_depth_camera = CameraWrapper(cfg=ANYMAL_D_CFG)
        self._anymal_direct.front_depth_camera.spawn(f'{self.anymal_prim_path}/wide_angle_camera_front_camera', # put the depth camera where the wide angle one should be (because its convenient)
                                                      #orientation=orientation)
                                                      orientation=[.50603, .42113, -.48007, -.57642]) # NOTE: Old orientation for old camera.
        param_attr = getattr(self._anymal_direct.front_depth_camera._sensor_prim, f"GetVerticalApertureAttr")
        set_prop_val(param_attr(), .93)

        # initialize camera
        self._anymal_direct.front_depth_camera.initialize()
        self._anymal_direct.front_depth_camera.setup_image_data_logger()
        self._anymal_direct.front_depth_camera.reset()

        transform(self.anymal_prim_path+"/base",translation=[0,0,SimConfig.robot_height])

    def setup(self) -> None:
        """
        [Summary]

        Set up keyboard listener and add physics callback

        """
        self._appwindow = omni.appwindow.get_default_app_window()
        self._input = carb.input.acquire_input_interface()
        self._keyboard = self._appwindow.get_keyboard()
        self._sub_keyboard = self._input.subscribe_to_keyboard_events(self._keyboard, self._sub_keyboard_event)


        # add callbacks (executes from BOTTOM FIRST, last-in-first-out):
        self._world.add_physics_callback("IsaacRunner_Physics", self.on_physics_step)
        self._world.add_physics_callback("inference", callback_fn = self.save_image_callback)
        self._world.add_physics_callback("anymal_advance", callback_fn=self.main_physics_callback)
        #self._world.add_physics_callback("logging", callback_fn = self.logging_callback)

        

    def main_physics_callback(self, step_size):
        """Callback to run on each physics step (anymal_advance should be here)"""

        if self.lock > 0:
            self.lock -= 1

        if not self.robot_state.is_moving(threshold=.4) or not self.robot_state.is_standing():
            self.respawn_anymal()

        assert SimConfig.control_type in ['manual', 'action_model', 'goal_driven']

        if self._anymal_direct is not None:
            if  SimConfig.control_type == 'manual' or SimConfig.control_type == 'action_model':
                self._anymal.advance(step_size, self._base_command)
            elif SimConfig.control_type == 'goal_driven':
                pose = self.robot_state.get_xyt_pose()
                self._base_command = self.planner.calculate_action(self.planner.path, pose,
                                                        fwd_vel=SimConfig.robot_max_velocity, look_ahead=SimConfig.look_ahead)
                self._anymal.advance(step_size, self._base_command)
        
                
    def planning_callback(self, step_size):
        '''Calls upon the planner to update, defers to user input if enabled.'''
        if self._world.current_time_step_index % SimConfig.planner_callback_rate == 0: 
            if SimConfig.control_type == 'goal_driven':
                pose = self.robot_state.get_xyt_pose()
                self.planner.calculate_path(pose)

    def logging_callback(self, step_size):

        if self._world.current_time_step_index % SimConfig.image_callback_rate != 0 or self.lock > 0:
                return
            
        self._anymal_direct.front_depth_camera.buffer()
        self._anymal_direct.front_depth_camera.save_latest_data()
        image = self._anymal_direct.front_depth_camera.data.output
        rgb_image = image['rgb'] 
        depth = image['distance_to_camera']

        torch_image = torch.from_numpy(rgb_image).float().to(SimConfig.device)
        torch_image = torch_image.permute(2, 0, 1)[:3] # change to channel first: (3, n, m)
        torch_depth = torch.from_numpy(depth)[None].to(SimConfig.device)


        camera_parent = 'wide_angle_camera_front_camera'
        T_PIW = transformation_matrix_of_pose(get_pose(f"{self.anymal_prim_path}/{camera_parent}"), ordering="wxyz").to(SimConfig.device)
        T_CIP = transformation_matrix_of_pose(get_pose(f"{self.anymal_prim_path}/{camera_parent}/Camera"), ordering="wxyz").to(SimConfig.device)
        T_CIW = T_PIW @ T_CIP @ torch.Tensor([[-1,0,0,0],[0,1,0,0],[0,0,-1,0],[0,0,0,1]]).to(device='cuda') 
 
        self.log_data(img = rgb_image[:,:,:3], depth = torch_depth, pose_cam_in_world=T_CIW)

        self.cur_image = rgb_image

    __image_inference_callback_counter = 0

    def image_inference_callback(self, step_size):
        if self._world.current_time_step_index % SimConfig.inference_callback_rate != 0: 
            return
        
        self._anymal_direct.front_depth_camera.buffer()
        self._anymal_direct.front_depth_camera.save_latest_data()
        image = self._anymal_direct.front_depth_camera.data.output
        rgb_image = image['rgb'] 
        theta = self.action_model(rgb_image)
        self._base_command = np.array([1, 0, theta])

        torch_image = torch.from_numpy(rgb_image).float().to(SimConfig.device)
        torch_image = torch_image.permute(2, 0, 1)[:3] # change to channel first: (3, n, m)
        img = torch_image / 255.0
        size = (224,224)

    def save_image_callback(self, step_size):
        if not self.recording or self._world.current_time_step_index % SimConfig.inference_callback_rate != 0: 
            return
        
        self._anymal_direct.front_depth_camera.buffer()
        self._anymal_direct.front_depth_camera.save_latest_data()
        image = self._anymal_direct.front_depth_camera.data.output
        rgb_image = image['rgb'] 
        
        img = self.clean_image_for_model(rgb_image)

        #img = torchvision.transforms.functional.center_crop(img, size)
        img = F.interpolate(img[None], (224,224), mode='bilinear').squeeze(0)

        # Save Result
        num = self.__image_inference_callback_counter; 
        self.__image_inference_callback_counter+=1
        cmd = F"{self._base_command[0]},{self._base_command[1]},{self._base_command[2]}"

         
        if not os.path.exists("image_command"):
            os.mkdir("image_command")
        
        if StageConfig.default_stage:
            stage_name = StageConfig.default_stage.split("/")[-1].split(".")[0]
            if not os.path.exists(F"image_command/{stage_name}"):
                os.mkdir(F"image_command/{stage_name}")
                print("Created Directory")
            torch.save(img,F"image_command/{stage_name}/{num},{cmd}.pt" if StageConfig.default_stage != None else F"image_command/{num},{cmd}.pt")
            print("Saved Image")

    def image_and_proprio_callback(self, step_size):
        """Processes image"""
        self.last_image_ts = self._world.current_time_step_index

        camera_parent = 'wide_angle_camera_front_camera'
        use_for_training = True

        if self._world.current_time_step_index % SimConfig.image_callback_rate != 0 or self.lock > 0:
            return
        
        self._anymal_direct.front_depth_camera.buffer()
        self._anymal_direct.front_depth_camera.save_latest_data()
        image = self._anymal_direct.front_depth_camera.data.output
        rgb_image = image['rgb'] 
        #depth = image['distance_to_camera']

        torch_image = torch.from_numpy(rgb_image).float().to(SimConfig.device)
        torch_image = torch_image.permute(2, 0, 1)[:3] # change to channel first: (3, n, m)
        torch_image_normalized = torch_image / 255.0
        #torch_depth = torch.from_numpy(depth)[None].to(SimConfig.device)

        # Get the intrinsics matrix and modify it according to crop/interpolations that will be performed 
        K = self._anymal_direct.front_depth_camera.K.clone()
        
        torch_image_resized, intrinsics = crop_and_interpolate(torch_image_normalized, SimConfig.network_input_image_height, SimConfig.network_input_image_width, K)
        #torch_depth_resized, _ = crop_and_interpolate(torch_depth, SimConfig.network_input_image_height, SimConfig.network_input_image_width, torch.eye(4,4)) # not using K here
        intrinsics = intrinsics.unsqueeze(0).to(SimConfig.device)

        # Get Poses
        pose_base_in_world = transformation_matrix_of_pose(get_pose(self.anymal_prim_path + "/base"), ordering="xyzw").to(SimConfig.device)
        T_PIW = transformation_matrix_of_pose(get_pose(f"{self.anymal_prim_path}/{camera_parent}"), ordering="wxyz").to(SimConfig.device)
        T_CIP = transformation_matrix_of_pose(get_pose(f"{self.anymal_prim_path}/{camera_parent}/Camera"), ordering="wxyz").to(SimConfig.device)
        T_CIW = T_PIW @ T_CIP @ torch.Tensor([[-1,0,0,0],[0,1,0,0],[0,0,-1,0],[0,0,0,1]]).to(device='cuda') 
 
        H = torch.IntTensor([SimConfig.network_input_image_height]).to(SimConfig.device)
        W = torch.IntTensor([SimConfig.network_input_image_width]).to(SimConfig.device)


    def log_data(self, **kwargs):
        '''
        Logs data from image_callback, according to SimConfig.logging_rate.

        kwargs:
            img: image to log

            masked_conf: confidence masked over image

            masked_trav: traversability masked over image

            pose_cam_in_world: torch.tensor, saved as .pt

            depth: torch.tensor, save as .pt

            depth_np: numpy array, save as .png
        '''
        if 'masked_trav' in kwargs:
            masked_trav = kwargs.get('masked_trav')
        if 'masked_conf' in kwargs:
            masked_conf = kwargs.get('masked_conf')

        if 'masked_conf' in kwargs and 'masked_trav' in kwargs:
            im1 = Image.fromarray(masked_trav)
            im2 =Image.fromarray(masked_conf) 
            both = get_concat_h(im1, im2)
            both.save(f'live.jpg') # show real time masked traversability

        if (self._world.current_time_step_index // SimConfig.image_callback_rate) % SimConfig.logging_rate != 0:
            return
    
        i = self._world.current_time_step_index // SimConfig.image_callback_rate
        relative_path = os.path.join(os.getcwd(),self.log_dir)

        if not os.path.exists(relative_path):
            print('making dir')
            os.mkdir(relative_path)
        print(relative_path)
        if 'img' in kwargs:
            ground_truth = Image.fromarray(kwargs.get('img'))
            ground_truth.save(os.path.join(relative_path, f'img_{i}.jpg')) 
        if 'masked_conf' in kwargs and 'masked_trav' in kwargs:
            both.save(os.path.join(relative_path,f'masks_{i}.jpg'))
        if 'pose_cam_in_world' in kwargs:
            pose_cam_in_world = kwargs.get('pose_cam_in_world')
            torch.save(pose_cam_in_world, os.path.join(relative_path,f'cam_loc_{i}.pt'))
        # if 'depth' in kwargs:
        #     depth = kwargs.get('depth')
        #     torch.save(depth, os.path.join(relative_path,f'depth_{i}.pt'))
        # if 'depth_np' in kwargs:
        #     depth = kwargs.get('depth_np')
        #     mask = np.isinf(depth)
        #     depth[mask] = 0
        #     depth = depth / np.max(depth)
        #     depth[mask] = 1
        #     depth *= 255
        #     depth = depth.astype('uint8')

        #     image = Image.fromarray(depth)
            
        #     image.save(os.path.join(relative_path, f'depth_img_{i}.jpg'))


    # NOTE: Required for raytracing calls.
    def on_physics_step(step, step_size : float):
        pass

    def run(self) -> None:
        """Step simulation."""
        self._world.reset()
    
        self.respawn_anymal() # NOTE: Necessary to spawn anymal above ground.

        # Main execution loop
        while simulation_app.is_running():
            
            self._world.step(render=True)

            if SimConfig.max_resets and self.reset_count == SimConfig.max_resets:
                break #NOTE: Changed from simulation_app.close() because that gets called anyway after run() 
                 
        if not StageConfig.default_stage: # So we don't save a stage that we opened.
            if "Y" in input("Save stage? (Y/n) ").capitalize():
                # TODO: Clean up the stage
                if is_defined(DebugConfig.debug_prim_directory):
                    delete(DebugConfig.debug_prim_directory)
                if is_defined(self.anymal_prim_path):
                    delete(self.anymal_prim_path)
                if is_defined("/physicsScene"):
                    delete("/physicsScene")
                if is_defined("/Render/PostProcess/SDGPipeline"):
                    delete("/Render/PostProcess/SDGPipeline")  
                stage_name = input('Enter stage name (saving to ./stages/NAME.usd): ')
                stage_path = F"./stages/{stage_name}.usd"
                save_stage(stage_path)

    def _sub_keyboard_event(self, event, *args, **kwargs) -> bool:  
        """Keyboard subscriber callback to when kit is updated."""
        # reset event
        self._event_flag = False
        # when a key is pressed for released  the command is adjusted w.r.t the key-mapping
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            # on pressing, the command is incremented
            if event.input.name in SimConfig.input_keyboard_mapping:
                if SimConfig.teamviewer:
                    self._base_command[0:3] == np.array(SimConfig.input_keyboard_mapping[event.input.name])
                else:
                    self._base_command[0:3] += np.array(SimConfig.input_keyboard_mapping[event.input.name])

                print(self._base_command)
            elif event.input.name == "R":
                self.respawn_anymal(SimConfig.robot_spawn_location)
            elif event.input.name == "P":
                self.recording = not self.recording
                print(f"Recording: {self.recording}")
            elif event.input.name == "G":
                self.planner.random_new_goal()


        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            # on release, the command is decremented
            if event.input.name in SimConfig.input_keyboard_mapping:
                if SimConfig.teamviewer:
                    self._base_command[0:3] += np.array(SimConfig.input_keyboard_mapping[event.input.name])
                else:
                    self._base_command[0:3] -= np.array(SimConfig.input_keyboard_mapping[event.input.name])
                print(self._base_command)

        return True

    def respawn_anymal(self, loc=(0,0,0)):
        #TODO: clear the proprio and mission node graphs on respawn
        # maybe the robot prim itself stays in place, while its parts move after a reset
        # watch the anymal itself position before and after reset

        self.robot_state.reset()
        self.lock_proprio(250) # resself._input_keyboard_mapping
        # rotate robot to face towards goal node from the start
        #theta = (360 + 180 * np.arctan2(self.planner.goal[1], self.planner.goal[0]) / np.pi) % 360
        theta = 360*np.random.random()
        
        transform(F"{self.anymal_prim_path}/base", translation=[loc[0],loc[1] + SimConfig.robot_height,loc[2]], rotation=[-90,theta,0],rotation_order=[0,2,1])

        self.planner.calculate_path(self.robot_state.get_xyt_pose())
        self.reset_count += 1
        print(f'RESET #{self.reset_count}')

    def lock_proprio(self, steps):
        self.lock = steps

    def clean_image_for_model(self, rgb_image):
        """
        Converts rgb_image to a format that can be used by the model

        Inputs: 
            rgb_image: a numpy array with color values in (0, 255)
        """

        torch_image = torch.from_numpy(rgb_image).float().to(SimConfig.device)
        torch_image = torch_image.permute(2, 0, 1)[:3] # change to channel first: (3, n, m)
        img = torch_image / 255.0
        size = (224,224)

        if img.size()[1] < size[0] or img.size()[2] < size[1]:
            img = F.interpolate(img[None], size, mode='bilinear').squeeze(0)

        #img = torchvision.transforms.functional.center_crop(img, size)
        img = F.interpolate(img[None], size, mode='bilinear').squeeze(0)
        return img


def main():
    """Parse arguments and instantiate/run Orbit."""
    physics_dt = 1 / 200.0
    render_dt = 1 / 60.0

    runner = AnymalRunner(physics_dt=physics_dt, render_dt=render_dt)
    # simulation_app.update() NOTE: Included in demo code but seems to have no effect.
    runner.setup()

    runner.run()
    simulation_app.close()


if __name__ == "__main__":
    main()
