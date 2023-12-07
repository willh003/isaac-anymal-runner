
import torch
from rl_games.algos_torch import torch_ext
from rl_games.algos_torch import model_builder

from omni.isaac.orbit.robots.legged_robot import LeggedRobotData


from helper_methods import path_with_respect_to_helper_method,load_yaml,spacer




class OmniLocomotionModel:
        
        def __init__(self,num_envs=1,num_actions=12) -> None:

            self._locomotion_policy_weights_path = path_with_respect_to_helper_method('anymal-assets/AnymalC/anymal_c_locomotion_omniisaacgymenvs/weight.pth')
            self._policy_cfg_path = path_with_respect_to_helper_method('anymal-assets/AnymalC/anymal_c_locomotion_omniisaacgymenvs/config.yaml')
            self._policy_cfg_dict = load_yaml(self._policy_cfg_path)
            self._task_cfg = self._policy_cfg_dict['task']
            self._train_cfg = self._policy_cfg_dict['train']

            # normalization for policy
            self.lin_vel_scale = self._task_cfg["env"]["learn"]["linearVelocityScale"]
            self.ang_vel_scale = self._task_cfg["env"]["learn"]["angularVelocityScale"]
            self.dof_pos_scale = self._task_cfg["env"]["learn"]["dofPositionScale"]
            self.dof_vel_scale = self._task_cfg["env"]["learn"]["dofVelocityScale"]
            self.action_scale = self._task_cfg["env"]["control"]["actionScale"]


            self._previous_action = torch.zeros(num_envs,num_actions)

        def _build_policy(self, device)->None:
            """This method builds a saved locomotion policy """
            self.builder = model_builder.ModelBuilder()

            self._model_builder = self.builder.load(self._policy_cfg_dict['train']['params'])
            self.build_config = {'actions_num': 12, 'input_shape': (188,), 'num_seqs': 1,
                    'value_size': 1, 'normalize_value': True, 'normalize_input': True}


            self.model = self._model_builder.build(self.build_config).to(device)

            checkpoint = torch_ext.load_checkpoint(self._locomotion_policy_weights_path)


            self.model.load_state_dict(checkpoint['model'])
            self.model.eval()

            spacer()
            print("FINISHED LOADING LOCOMOTION MODEL : ")
            print(self.model)
            spacer()

        def build(self,device,default_positions)->None:
            """TODO ADD SOME LOGGERS and default initalizers"""
            self._build_policy(device)

            self._dof_pos = default_positions
            self.model.running_mean_std.training = False # Double check if this is correct 
        

            self.commands_scale = torch.tensor(
                 [self.lin_vel_scale, self.lin_vel_scale, self.ang_vel_scale],
                 device=device ,
                 requires_grad=False,
                 )
        

        def get_action(self,data:LeggedRobotData,heights,command):
             
            assert self.model is not None


            self.obs_buf = torch.cat(
                (
                    data.root_lin_vel_b * self.lin_vel_scale,
                    data.root_ang_vel_b * self.ang_vel_scale,
                    data.projected_gravity_b,
                    (command * self.commands_scale).unsqueeze(0),
                    (data.dof_pos -self._dof_pos)* self.dof_pos_scale,
                    data.dof_vel * self.dof_vel_scale,
                    heights.unsqueeze(0),
                    self._previous_action,
                ),
                dim=-1
            ).to(torch.float32)
            
         
            input_args = {'is_train':False,'obs':self.obs_buf}

            self._previous_action = self.model(input_args)['actions'].detach()*self.action_scale # determenistic approach

            return self._previous_action
