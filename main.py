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
        # self.log_dir = unique_log_dir(SimConfig.log_dir)
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

        transform(self.anymal_prim_path+"/base",translation=[0,SimConfig.robot_height,0])

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


    __save_image_callback_counter = 0
    __save_image_callback_rollout = 0
    def save_image_callback(self, step_size):
        if not self.recording or self._world.current_time_step_index % SimConfig.inference_callback_rate != 0: 
            return

        self._anymal_direct.front_depth_camera.buffer()
        self._anymal_direct.front_depth_camera.save_latest_data()
        image = self._anymal_direct.front_depth_camera.data.output
        rgb_image = image['rgb'] 
        
        img = self.clean_image_for_model(rgb_image)

        live_preview_img = img.permute(1,2,0)
        live_preview_img = torch.flip(live_preview_img, dims=[2])
        live_preview_img = live_preview_img.cpu().numpy() * 255
        cv2.imwrite("live.jpg", live_preview_img)

        rgb = torch.zeros((224,224,3))
        rgb[:,:,:3] = img.permute(1,2,0)

        # Save Result
        rollout = self.__save_image_callback_rollout
        num = self.__save_image_callback_counter; 
        cmd = F"{self._base_command[0]},{self._base_command[1]},{self._base_command[2]}"

        # if num % 12 == 0:
        #     cv2.imwrite(F"for_paper_{num}.jpg", live_preview_img)

        if not os.path.exists("image_command"):
            os.mkdir("image_command")
        
        if StageConfig.default_stage:
            stage_name = StageConfig.default_stage.split("/")[-1].split(".")[0]
            if not os.path.exists(F"image_command/{stage_name}"):
                os.mkdir(F"image_command/{stage_name}")
                print("Created Directory")
            torch.save(rgb,F"image_command/{stage_name}/{rollout},{num},{cmd}.pt" if StageConfig.default_stage != None else F"image_command/{num},{cmd}.pt")
            #print("Saved Image")

            if self.__save_image_callback_counter % SimConfig.frames_per_rollout == SimConfig.frames_per_rollout - 1:
                self.respawn_anymal() # NOTE: updates the counters
            else:
                self.__save_image_callback_counter += 1

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

        self.robot_state.reset()
        self.lock_proprio(250) # resself._input_keyboard_mapping
        
        theta = 0 # 90 #-25 + np.sign((np.random.random()-0.5))*8.5
        print(theta)
        transform(F"{self.anymal_prim_path}/base", translation=[loc[0],loc[1] + SimConfig.robot_height,loc[2]], rotation=[-90,theta,0])

        self.planner.calculate_path(self.robot_state.get_xyt_pose())
        self.reset_count += 1
        print(f'RESET #{self.reset_count}/{SimConfig.max_resets}')

        self.__save_image_callback_rollout += 1
        self.__save_image_callback_counter = 0

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
