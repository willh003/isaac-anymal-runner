import torch    
import functools
from helper_methods import path_with_respect_to_helper_method,load_yaml,spacer,obs_generator


class ETH_WildNavModel:
    def _load_rnn_policy(self)->None:
        
        self._model = torch.jit.load(self._policy_path).to(self.device)
        self._hidden = torch.zeros(1, 100).to(self.device)
        self._gen = functools.partial(obs_generator, keys=["obs", "priv_exte"])
    
    def rnn_policy(self, obs, info, hidden):
        
        self.policy_obs = self._gen(obs, info)
        action, self._hidden = self._model(self.policy_obs, hidden)
        
        return action
    
'''
        @property
        def gc(self) -> torch.Tensor:
            """Returns the generalized coordinates for an articulated system.

            Returns:
                torch.Tensor: Generalized coordinates `[root_pos, root_quat, dof_pos]` in simulation world frame.
                    Shape=(num_envs, 7 + num_dof).
            """
            return torch.cat([self.root_pos_w, self.root_quat_w, self.dof_pos], dim=1)

        @property
        def gv(self) -> torch.Tensor:
            """Returns the generalized velocities for an articulated system.

            Returns:
                torch.Tensor: Generalized velocities `[root_lin_vel, root_ang_vel, dof_vel]` in simulation world frame.
                    Shape=(num_envs, 6 + num_dof).
            """
            return torch.cat([self.root_lin_vel_w, self.root_ang_vel_w, self.dof_vel], dim=1)
        '''
