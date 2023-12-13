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
import torch
import torch.nn.functional as F
import torchvision.transforms
from PIL import Image
import cv2
import carb
import yaml

# Stage Builder Imports
from isaac_stage.utils import open_stage, save_stage, get_context, get_stage
from isaac_stage.prims import create_scope, get_pose, delete, is_defined, transform, create_sphere
from isaac_stage.appliers import apply_color_to_prim


# Camera Imports (Sean)
from camera_wrapper import CameraWrapper, ANYMAL_D_CFG

# Planning Imports
from ws_isaac_planner.planner import PlanningAgent
from ws_isaac_planner.utils import pure_pursuit_step

# General 
from states import RobotState, LearningState
from config import SimConfig, StageConfig, DebugConfig, ModelConfig
from utils import *

from bc_trav.model_factories import trav_prior_factory, get_factory_from_type
from kornia.geometry.camera.pinhole import PinholeCamera


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
        self.log_dir = unique_log_dir(SimConfig.log_dir)
        self.lock = 0
        self.reset_count = 0
                
        # Data classes
        self.robot_state = RobotState()
        # Initialize traversability generator to process velocity commands
        self.planner = PlanningAgent(start=(0,0), goal=(0,0), env_dim=[SimConfig.env_x, SimConfig.env_y], 
                                     edge_radius = 1, draw_markers = SimConfig.draw_markers) 
        
        self.planner.random_new_goal()


        
        with open(ModelConfig.trav_cfg_path, 'r') as file:
            trav_cfg = yaml.safe_load(file)
        trav_checkpoint = trav_cfg['traversability']['checkpoint_path']

        self.trav_model = trav_prior_factory(trav_checkpoint, ModelConfig.model_image_size )
        
        if ModelConfig.model_type in ['trav', 'image', 'fusion']:
            factory = get_factory_from_type(ModelConfig.model_type)
            bc_model = factory(trav_cfg_path = ModelConfig.trav_cfg_path,
                                           train_cfg_path= ModelConfig.train_cfg_path, 
                                           ckpt= ModelConfig.action_checkpoint,
                                           device = SimConfig.device)            
            self.action_model = self.discrete_action_model(bc_model, theta_scale=.4)
        else:
            self.action_model = self.heuristic_trav(theta_scale=.4)
        self.cur_image = None

        # Data collection
        self.recording = False
        self.obs_num=0
        self.rollout_num=0

    def heuristic_trav(self, theta_scale=.4):
        """
        A heuristic which guides the robot by guiding it towards the most traversable patch.
        """
        def action_fn(x : torch.Tensor):
            if ModelConfig.heuristic_type == "COLS":
                # x is a (1,6,3,244,244)
                # Split image into 8 regions, act on most traversable.
                # [0-2] ->  LEFT
                # [3-4]   ->  STRAIGHT
                # [5-7] ->  RIGHT
                # ._______________.
                # |0|1|2|3|4|5|6|7|
                # | | | | | | | | | 
                # | | | | | | | | | (224 x 224) split into (224x28) patches
                # | | | | | | | | | 
                # | | | | | | | | | 
                # ^^^^^^^^^^^^^^^^^
                trav = self.trav_model(x[0][-1][None])[0][0] # trav(x) -> (1,2,224,224) -> (224,224)

                regions=[]
                [regions.append(r) for r in trav.split(28,dim=1)]

                traversability_signals = [region[:,:].mean() for region in regions]

                max_traversable_index = torch.argmax(torch.tensor(traversability_signals))
                self.__traversability_target_region = max_traversable_index.item()

                if max_traversable_index in [0,1,2]:
                    return 1 # NOTE: This looks backwards but does the right thing in practice??
                elif max_traversable_index in [3,4]:
                    return 0 
                elif max_traversable_index in [5,6,7]:
                    return -1
            elif ModelConfig.heuristic_type == "OCTS":
                # x is a (1,6,3,244,244)
                # Split image into 8 regions, act on most traversable.
                # [0,4,5] ->  LEFT
                # [1,2]   ->  STRAIGHT
                # [3,6,7] ->  RIGHT
                # .___________.
                # |0 |1 |2 |3 |
                # |  |  |  |  |
                # |--|--|--|--|  (224 x 224) split into (112x56) patches
                # |4 |5 |6 |7 |
                # |  |  |  |  |
                # ^^^^^^^^^^^^^
                trav = self.trav_model(x[0][-1][None])[0][0] # trav(x) -> (1,2,224,224) -> (224,224)

                regions=[]
                [regions.append(reg) for r in trav.split(112,dim=0) for reg in r.split(56,dim=1)]

                traversability_signals = [region[:,:].mean() for region in regions]

                max_traversable_index = torch.argmax(torch.tensor(traversability_signals))
                self.__traversability_target_region = max_traversable_index.item()

                if max_traversable_index in [0, 4, 5]:
                    return 1 # NOTE: This looks backwards but does the right thing in practice??
                elif max_traversable_index in [1, 2]:
                    return 0 
                elif max_traversable_index in [3, 6, 7]:
                    return -1
            elif ModelConfig.heuristic_type == "BH-OCTS": #Bottom-Heavy Octants
                # x is a (1,6,3,244,244)
                # Split image into 8 regions, act on most traversable.
                # [0,4,5] ->  LEFT
                # [1,2]   ->  STRAIGHT
                # [3,6,7] ->  RIGHT
                # .___________.
                # |0 |1 |2 |3 |
                # |--|--|--|--|
                # |4 |5 |6 |7 |  (224 x 224) split into (56x56)-top and (168x56)-bottom patches
                # |  |  |  |  |
                # |  |  |  |  |
                # ^^^^^^^^^^^^^
                trav = self.trav_model(x[0][-1][None])[0][0] # trav(x) -> (1,2,224,224) -> (224,224)

                regions=[]
                [regions.append(reg) for r in trav.split([56,168],dim=0) for reg in r.split(56,dim=1)]

                traversability_signals = [region[:,:].mean() for region in regions]

                max_traversable_index = torch.argmax(torch.tensor(traversability_signals))
                self.__traversability_target_region = max_traversable_index.item()

                if max_traversable_index in [0, 4, 5]:
                    return 1 # NOTE: This looks backwards but does the right thing in practice??
                elif max_traversable_index in [1, 2]:
                    return 0 
                elif max_traversable_index in [3, 6, 7]:
                    return -1
            elif ModelConfig.heuristic_type == "BHP-OCTS": #Bottom-Heavy Octants Pairwise
                # x is a (1,6,3,244,244)
                # Split image into 8 regions, act on most traversable PAIR
                # [(0,1),(4,5)] -> [0,3] -> LEFT
                # [(1,2),(5,6)] -> [1,4] -> STRAIGHT
                # [(2,3),(6,7)] -> [2,5] -> RIGHT
                # .___________.
                # |0 |1 |2 |3 |
                # |--|--|--|--|
                # |4 |5 |6 |7 |  (224 x 224) split into (56x56)-top and (168x56)-bottom patches
                # |  |  |  |  |
                # |  |  |  |  |
                # ^^^^^^^^^^^^^
                trav = self.trav_model(x[0][-1][None])[0][0] # trav(x) -> (1,2,224,224) -> (224,224)

                regions=[]
                [regions.append(reg) for r in trav.split([56,168],dim=0) for reg in r.split(56,dim=1)]

                solo_traversability_signals = [region[:,:].mean() for region in regions]

                traversability_signals = []
                [traversability_signals.append(solo_traversability_signals[4*r + c]+solo_traversability_signals[4*r + c+1]) for r in [0,1] for c in [0,1,2]]


                max_traversable_index = torch.argmax(torch.tensor(traversability_signals))
                self.__traversability_target_region = max_traversable_index.item()

                if max_traversable_index in [0,3]:
                    return 1 # NOTE: This looks backwards but does the right thing in practice??
                elif max_traversable_index in [1, 4]:
                    return 0 
                elif max_traversable_index in [2,5]:
                    return -1
        
        return action_fn

                

    def discrete_action_model(self, model, theta_scale=.4):
        """
        Given a model that outputs a vector of size 3, containing probs for left turn, straight, right turn:
        Return a rotation value to apply (scaled by theta_scale) 
        """
        def action_fn(x):
            probs = model(x)

            amax = torch.argmax(probs).item()
            if amax==0:
                return -1*theta_scale
            elif amax==1:
                return 0.0
            else:
                return theta_scale

        return action_fn
    
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
        self.intrinsics = self._anymal_direct.front_depth_camera.compute_K()

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
        self._world.add_physics_callback("inference", callback_fn = self.image_inference_callback)
        self._world.add_physics_callback("observe", callback_fn = self.save_image_callback)
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


        T_CIW = self.get_cam_in_world()
        self.log_data(img = rgb_image[:,:,:3], depth = torch_depth, pose_cam_in_world=T_CIW)

        self.cur_image = rgb_image

    __image_inference_callback_counter = 0

    def image_inference_callback(self, step_size):
        if self._world.current_time_step_index % SimConfig.inference_callback_rate != 0 or self.lock > 0: 
            return

        if len(self.robot_state.img_memory) < 6:
            return
        obs = torch.stack([self.clean_image_for_model(image) for image in self.robot_state.img_memory])
        obs = obs.to(SimConfig.device)
        theta = self.action_model(obs[None]) # expand dims because it should be batched 
        self._base_command = np.array([1.5, 0, theta])
        print(f'theta pred: {theta}')

    def save_image_callback(self, step_size):
        if self._world.current_time_step_index % SimConfig.inference_callback_rate != 0: 
            return
        
        self._anymal_direct.front_depth_camera.buffer()
        self._anymal_direct.front_depth_camera.save_latest_data()
        image = self._anymal_direct.front_depth_camera.data.output
        rgb_image = image['rgb'] 

        self.robot_state.add_image(rgb_image)

        if self.recording:
            rgb_image = self.clean_image_for_model(rgb_image).permute(1,2,0).cpu()
            path = f'bc_data/{self.rollout_num},{self.obs_num},{self._base_command[0]},{self._base_command[1]},{self._base_command[2]}.pt'
            torch.save(rgb_image, path)
            self.obs_num+=1

        else:
            img = self.clean_image_for_model(rgb_image)

            stage_name = StageConfig.default_stage.split("/")[-1]

            cu_img = img.cuda()[None]
            probs = self.trav_model(cu_img)
            pred = torch.argmax(probs[0], dim=0).cpu()
            
            # Update Mask to show target square if Using Heuristic
            if ModelConfig.model_type == 'heuristic':
                if ModelConfig.heuristic_type == "COLS":
                    col = self.__traversability_target_region
                    img[1,:,28*col:28*(col+1)] += 0.25
                elif ModelConfig.heuristic_type == "OCTS":
                    row = self.__traversability_target_region // 4
                    col = self.__traversability_target_region % 4
                    img[1,112*row:112*(row+1),56*col:56*(col+1)] += 0.25
                elif ModelConfig.heuristic_type == "BH-OCTS":
                    row = self.__traversability_target_region // 4
                    col = self.__traversability_target_region % 4
                    if row == 0:
                        img[1,:56,56*col:56*(col+1)] += 0.25
                    else:
                        img[1,56:,56*col:56*(col+1)] += 0.25
                elif ModelConfig.heuristic_type == "BHP-OCTS":
                    row = self.__traversability_target_region // 3
                    col = self.__traversability_target_region % 3
                    if row == 0:
                        img[1,:56,56*col:56*(col+2)] += 0.25
                    else:
                        img[1,56:,56*col:56*(col+2)] += 0.25

            save_mask(img, pred, 'live_updates/live.png')

            # W
            rgbt = torch.zeros((224,224,4))
            rgbt[:,:,:3] = img.permute(1,2,0)
            rgbt[:,:, 3] = pred

            # Save Result
            num = self.__image_inference_callback_counter; 
            self.__image_inference_callback_counter+=1
            cmd = F"{self._base_command[0]},{self._base_command[1]},{self._base_command[2]}"

            if StageConfig.default_stage != None:
                stage_name = StageConfig.default_stage.split("/")[-1]
                if not os.path.exists(F"image_command/{stage_name}"):
                    os.mkdir(F"image_command/{stage_name}")
            #torch.save(rgbt.detach(),F"image_command/{stage_name}/{num},{cmd}.pt" if StageConfig.default_stage != None else F"image_command/{num},{cmd}.pt")
            
            self.obs_num += 1

            # Track Pathing Information
            if not "GLASS" in stage_name:
                if not os.path.exists("rollout_paths"):
                    os.mkdir(F"rollout_paths")
                if not os.path.exists(F"rollout_paths/{stage_name}"):
                    os.mkdir(F"rollout_paths/{stage_name}")
                
                rollout_kind = ModelConfig.model_type
                if rollout_kind == "heuristic":
                    rollout_kind += F"_{ModelConfig.heuristic_type}"
                x,y,z = get_pose(self.anymal_prim_path + '/base')[:3]
                with open(F"rollout_paths/{stage_name}/{rollout_kind}.dcsv",'a') as file:
                    file.write(F"{self.rollout_num},,{self.obs_num},,{x},{y},{z}\n")

                #self.respawn_anymal()
                print(F"Rollout: {self.reset_count}/{SimConfig.max_resets}")
            print('Inference complete')


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
                #if self.recording: #NOTE: For mapping out rollouts we still need these to increment.
                self.rollout_num += 1
                self.obs_num = 0
            elif event.input.name == "P":
                self.recording = not self.recording
                print(f"Recording: {self.recording}")
            elif event.input.name == "G":
                self.planner.random_new_goal()
            elif event.input.name == "Q": #
                self.quit_and_plot_rollouts()
                

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
        self._base_command = np.array([0,0,0])
        self.lock_n_steps(100) # allow the robot to get settled before doing anything else
        # rotate robot to face towards goal node from the start
        #theta = (360 + 180 * np.arctan2(self.planner.goal[1], self.planner.goal[0]) / np.pi) % 360
        spawn_angle_offset = 0 #2*(np.random.random()-0.5)*60
        rx,ry,rz = SimConfig.robot_spawn_rotation
        spawn_angle = [rx, ry, rz+spawn_angle_offset]
        transform(F"{self.anymal_prim_path}/base", translation=[loc[0], loc[1], loc[2] + SimConfig.robot_height], rotation=spawn_angle)

        self.planner.calculate_path(self.robot_state.get_xyt_pose())
        self.reset_count += 1
        print(f'RESET #{self.reset_count}')

    def lock_n_steps(self, steps):
        self.lock = steps

    def clean_image_for_model(self, rgb_image, depth_image = None):
        """
        Converts rgb_image to a format that can be used by the model
        If depth_image is provided, scales it in the same way as rgb_image

        Inputs: 
            rgb_image: a numpy array with color values in (0, 255)
        """

        torch_image = torch.from_numpy(rgb_image).float().to(SimConfig.device)
        torch_image = torch_image.permute(2, 0, 1)[:3] # change to channel first: (3, n, m)
        img = torch_image / 255.0

        img = F.interpolate(img[None], ModelConfig.model_image_size, mode='bilinear').squeeze(0)

        if torch.is_tensor(depth_image):
            torch_depth = torch.from_numpy(depth_image).float().to(SimConfig.device)
            depth_interp= F.interpolate(torch_depth[None],
                                        ModelConfig.model_image_size,
                                        mode='bilinear').squeeze(0)
            return img, depth_interp 
        
        return img
    
    def unproject(self, point, depth, intrinsics):
        T_CIW = self.get_cam_in_world()
        P = intrinsics @ T_CIW
        P_inv = torch.inverse(P) # (4, 4)
        homogenous = torch.stack(depth * point, torch.as_tensor([depth,1]))
        transform = P_inv @ homogenous
        
        return transform
        

    def get_cam_in_world(self):
        camera_parent = 'wide_angle_camera_front_camera'
        T_PIW = transformation_matrix_of_pose(get_pose(f"{self.anymal_prim_path}/{camera_parent}"), ordering="wxyz").to(SimConfig.device)
        T_CIP = transformation_matrix_of_pose(get_pose(f"{self.anymal_prim_path}/{camera_parent}/Camera"), ordering="wxyz").to(SimConfig.device)
        T_CIW = T_PIW @ T_CIP @ torch.Tensor([[-1,0,0,0],[0,1,0,0],[0,0,-1,0],[0,0,0,1]]).to(device='cuda') 
        return T_CIW

    def quit_and_plot_rollouts(self):
        def hsv_to_rgb(h,s,v) -> tuple:
            if s:
                if h == 1.0: h = 0.0
                i = int(h*6.0); f = h*6.0 - i
                w = v * (1.0 - s)
                q = v * (1.0 - s * f)
                t = v * (1.0 - s * (1.0 - f))
                if i==0: return (v, t, w)
                if i==1: return (q, v, w)
                if i==2: return (w, v, t)
                if i==3: return (w, q, v)
                if i==4: return (t, w, v)
                if i==5: return (v, w, q)
            else: return (v, v, v)

        stage_name = StageConfig.default_stage.split("/")[-1]    
        #stage_name = "Lincolns_Inn_Chapel_Undercroft.usd"
        rollout_kind = ModelConfig.model_type
        rollout_kind += F"_{ModelConfig.heuristic_type}"
        #rollout_filepath = "/home/pcgta/Documents/cs6670finalproject/anymalrunner/rollout_paths/Lake_Shore_Drone_Scan.usd/__saved_lakeheuristic_BH-OCTS.dcsv"
        create_scope("/World/Data")
        with open(rollout_filepath,'r') as file:
            max_color = 1
            for line in file.readlines():
                color = int(line.split(",,")[0])
                if color > max_color:
                    max_color = color
        with open(rollout_filepath,'r') as file:       
            for line in file.readlines():
                color = int(line.split(",,")[0])
                x_str, y_str, z_str = line.split(",,")[-1].split(",")
                x,y,z = float(x_str), float(y_str), float(z_str)
                create_sphere([x,y,z],radius=0.15,parent_prim_path="/World/Data",applier=apply_color_to_prim((1,0,0)))

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


    # def log_data(self, **kwargs):
    #     '''
    #     Logs data from image_callback, according to SimConfig.logging_rate.

    #     kwargs:
    #         img: image to log

    #         masked_conf: confidence masked over image

    #         masked_trav: traversability masked over image

    #         pose_cam_in_world: torch.tensor, saved as .pt

    #         depth: torch.tensor, save as .pt

    #         depth_np: numpy array, save as .png
    #     '''
    #     if 'masked_trav' in kwargs:
    #         masked_trav = kwargs.get('masked_trav')
    #     if 'masked_conf' in kwargs:
    #         masked_conf = kwargs.get('masked_conf')

    #     if 'masked_conf' in kwargs and 'masked_trav' in kwargs:
    #         im1 = Image.fromarray(masked_trav)
    #         im2 =Image.fromarray(masked_conf) 
    #         both = get_concat_h(im1, im2)
    #         both.save(f'live.jpg') # show real time masked traversability

    #     if (self._world.current_time_step_index // SimConfig.image_callback_rate) % SimConfig.logging_rate != 0:
    #         return
    
    #     i = self._world.current_time_step_index // SimConfig.image_callback_rate
    #     relative_path = os.path.join(os.getcwd(),self.log_dir)
