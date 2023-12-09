import os
from typing import List, Optional, Tuple, Union
import numpy as np

# Stage Builder Imports
from isaac_stage.utils import open_stage, save_stage
from isaac_stage.appliers import apply_appliers, apply_default_dirt_texture, apply_default_ground_physics_material, apply_color_to_prim
from isaac_stage.prims import create_sphere, create_triangle, create_scope, get_pose, delete, is_defined
from isaac_stage.terrain import Terrain, PathsTerrain
from isaac_stage.assets import Asset, AssetManager
from isaac_stage.stage_builder import StageBuilder, ForestStageBuilder

class SimConfig:
    # Relevant frames
    fixed_frame= 'odom'
    base_frame= 'base'
    footprint_frame= 'footprint'

    # Robot specs (comments are what I get from orbit bounding box)
    robot_length= 1.0 # 1.05
    robot_width= 0.6 # .67
    robot_height= 0.65 # .92- maybe this one is just robot height without the legs?

    # Traversability estimation params      # NOTE : Default Values
    traversability_radius= 5*5.0 # meters     # 5.0m
    image_graph_dist_thr= 5*0.2 # meters      # 0.2m 
    proprio_graph_dist_thr= 0.1 # meters    # 0.1m
    network_input_image_height= 224 # 448
    network_input_image_width= 224 # 448
    segmentation_type= "slic"
    feature_type= "dino"
    dino_patch_size= 16 # DINO only
    confidence_std_factor= 4.0
    scale_traversability= True
    scale_traversability_max_fpr= 0.05
    min_samples_for_training= 5

    # Optical flow params
    optical_flow_estimator_type= "none"

    # Supervision Generator
    untraversable_thr= -1 #0.01 NOTE: This should be factored out, the untraversable plane is deprecated. -1 Sets it to disabled.
    mission_name= "test"
    mission_timestamp= True

    # Threads (period in physics timesteps)
    image_callback_rate= 8 #10 hz
    proprio_callback_rate= 20  #4 hz
    learning_callback_rate= 8 #10 hz
    inference_callback_rate = 40
    planner_callback_rate = 400

    # Runtime options
    device= "cuda"
    mode= "online" # check out comments in the class WVNMode
    colormap= "RdYlBu"
    print_image_callback_time= False
    print_proprio_callback_time= False
    log_time= False
    log_trav = True
    log_confidence= False
    verbose= False
    debug_supervision_node_index_from_last= 10
    extraction_store_folder=  "nan"
    exp= "nan"

    # Image seg params
    loss_method = None
    traversability_threshold = .5
    threshold_update_period = image_callback_rate * 100 # every n images, update the confidence and traversability thresholds

    # Logging params
    log_dir = 'image_features'
    logging_rate = 20 # this one is w.r.t. to image_callback_rate (perform every n image callbacks)
    log = True

    # Simulation and Planner params
    sim_mode = 'log' # either 'log' or 'wvn'. If 'log', just collects images and camera locations. If 'wvn', runs the full wvn pipeline
    control_type = 'manual' # manual, goal_driven, or action_model
    teamviewer=False

    # Rollout Settings

    robot_spawn_location = (0, 0, 0)
    env_x = 80
    env_y = 80
    look_ahead = 10
    draw_markers = '' # '' for no markers, 'sparse' for just the current path, or 'dense' for the whole graph
    max_resets = 30 # number of resets to perform before shutdown. None to run forever
    frames_per_rollout = 53
    input_keyboard_mapping = {
            # forward command
            "W": [1.5, 0.0, 0.0],
            "UP": [1.5, 0.0, 0.0],
            # back command
            "S": [-1.0, 0.0, 0.0],
            "DOWN": [-1.0, 0.0, 0.0],
            # left command
            "D": [0.0, -1.0, 0.0],
            "RIGHT": [0.0, -1.0, 0.0],
            # right command
            "A": [0.0, 1.0, 0.0],
            "LEFT": [0.0, 1.0, 0.0],
            # yaw command (positive)
            "NUMPAD_7": [0.0, 0.0, 0.4],
            "N": [0.0, 0.0, 0.4],
            # yaw command (negative)
            "NUMPAD_9": [0.0, 0.0, -0.4],
            "M": [0.0, 0.0, -0.4],
        }

class StageConfig:
    #----------------------#
    #   stage parameters   #
    #----------------------#
    default_stage : Optional[str] =  "/home/sean/Desktop/Lincolns_Inn_Chapel_Undercroft.usd"

    xdim : int = 100 # NOTE: Has some relationship with env_x and env_y from the StaticParams
    ydim : int = 100
    #terrain_unit : float = 4

    asset_density : float = 0.75
    spawn_assets : bool = False

    #------------------------#
    #   terrain generation   #
    #------------------------#
    terrain : Terrain = PathsTerrain(terrain_unit=1, xdim=xdim, ydim=ydim, amp=0.1, spawn_radius=3,road_min_width=1
                            ,road_max_width=3, road_num=15, border_threshold=7, border_height=2.0, bowl_amplitude=5
                            ,applier=apply_appliers([apply_default_ground_physics_material, apply_default_dirt_texture]))

    #-------------------#
    #   asset manager   #
    #-------------------#
    asset_manager : AssetManager = AssetManager()

    # asset registration
    if spawn_assets:
        ()
        # FOREST ASSETS
        #asset_manager.register("./assets/forest/grass",recurse=False,asset_scale=0.035,applier=None)
        #asset_manager.register("./assets/forest/trees",recurse=False,asset_scale=0.0285,area_factor=0.6,applier=apply_default_ground_physics_material)
        #asset_manager.register("./assets/forest/rocks/Rock_Stone_11_RAWscan.usdz",asset_scale=0.035, area_factor=1.0,applier=apply_default_ground_physics_material)
        #asset_manager.register
    
    #-------------------#
    #   stage builder   #
    #-------------------#
    stage_builder : StageBuilder = ForestStageBuilder(xdim=xdim, ydim=ydim, terrain=terrain, asset_manager=asset_manager)

class DebugConfig:
    """This class contains flags and methods for debugging."""
    debug_prim_directory : str = "/World/Debug"

    # Flags
    check_axis : bool =  False
    enable_footpath : bool = True
    watch_supervision_signal : bool = True
    watch_res_state_dict : bool = False
    unproject : bool = False
    unproject_calls : int = 0
