import numpy as np
from isaacgym import torch_utils
from numpy.core.fromnumeric import nonzero

import torch
from torch.distributions.normal import Normal

# from rlgpu.utils.torch_jit_utils import *
# from isaacgym_anymal.anymal_common_rl.python.observation.observation_base import ObservationBase
from .observation_base import ObservationBase
from .raisim_conversion import isaac_raisim_joint_conversion, isaac_raisim_foot_conversion
from .cpg import CPG
from .ik_chimera import solve_ik, get_foot_height


class ProprioceptiveObservation(ObservationBase):
    def __init__(self, default_joint_pos=None, simulation_dt=0.0025, control_dt=0.02, num_envs=1, device="cuda:0"):
        super().__init__(simulation_dt, control_dt)
        self.num_envs = num_envs
        self.device = device
        # self.default_joint_pos = default_joint_pos.to(device)
        self.default_joint_pos = torch.tensor(
            [
                -0.13859,
                0.480936,
                -0.761428,
                -0.13859,
                -0.480936,
                0.761428,
                0.13859,
                0.480936,
                -0.761428,
                0.13859,
                -0.480936,
                0.761428,
            ],
            device=device,
        )
        self.base_frequency = 1.3

        self.cpg = CPG(shape=(num_envs, 4), base_frequency=1.3, dt=control_dt, device=device)
        self.joint_pos_history = torch.zeros(num_envs, 14, 12, dtype=torch.float, requires_grad=False).to(device)
        self.joint_vel_history = torch.zeros(num_envs, 14, 12, dtype=torch.float, requires_grad=False).to(device)
        self.joint_pos_scale = torch.tensor(
            [0.5, 0.4, 0.3, 0.5, 0.4, 0.3, 0.5, 0.4, 0.3, 0.5, 0.4, 0.3], requires_grad=False
        )
        self.joint_vel_scale = torch.tensor(
            [6.5, 4.5, 3.5, 6.5, 4.5, 3.5, 6.5, 4.5, 3.5, 6.5, 4.5, 3.5], requires_grad=False
        )
        self.freqency_scale = self.simulation_dt * 2 * np.pi
        self.history_randomizer = Normal(
            torch.zeros(*self.joint_pos_history.shape, requires_grad=False).to(device),
            torch.ones(*self.joint_pos_history.shape, requires_grad=False).to(device),
        )
        gravity_vec = torch.tensor([0.0, 0.0, 1.0]).to(device)
        self.gravity_vecs = gravity_vec.repeat((self.num_envs, 1))
        self.joint_target = torch.tile(self.default_joint_pos, (num_envs, 1)).to(device)
        self.prev_joint_target = torch.tile(self.default_joint_pos, (num_envs, 1)).to(device)
        self.prev2_joint_target = torch.tile(self.default_joint_pos, (num_envs, 1)).to(device)
        self.gravitational_axis_in_base = torch.tile(torch.tensor([0, 0, 1]), (num_envs, 1)).to(device)
        self.default_foot_height = 0.20
        self.feet_id = torch.tile(torch.tensor([0, 1, 2, 3], dtype=torch.int64), (num_envs,)).view(-1).to(device)
        self.foot_default_position = torch.tile(
            torch.tensor([0.4, 0.2, -0.55, -0.4, 0.2, -0.55, 0.4, -0.2, -0.55, -0.4, -0.2, -0.55]), (num_envs, 1)
        ).to(
            device
        )  # isaac order for now
        self.foot_default_z = -0.55
        self.action_mean = torch.zeros(16, requires_grad=False).to(device)
        self.action_std = torch.ones(16, requires_grad=False).to(device) * 0.2
        self.action_std[:4] = 0.5 * self.freqency_scale
        self.projected_gravity = torch.tile(torch.tensor([0.0, 0.0, 1.0]), (self.num_envs, 1)).to(self.device)
        self.init()

        # self.is_stand = False # TODO: Temporal
        self.is_stand = torch.zeros(num_envs, dtype=torch.bool).to(device)
        self.prev_is_stand = torch.zeros(num_envs, dtype=torch.bool).to(device)

    def init(self):
        self.mean = (
            torch.cat(
                [
                    torch.zeros(3),  # command
                    torch.tensor([0, 0, 1]),  # gravity axis
                    torch.zeros(6),  # base lin & ang vel
                    self.default_joint_pos.cpu().reshape(-1),  # current joint pos
                    torch.zeros(12),  # drive error history
                    torch.zeros(12),
                    torch.zeros(4),  # cpg info
                    torch.zeros(8),
                    torch.zeros(24),  # joint error history
                    torch.zeros(24),
                    self.default_joint_pos.cpu().reshape(-1),  # previous joint pos
                    self.default_joint_pos.cpu().reshape(-1),
                    torch.tensor([0]),
                ]
            )
            .to(self.device)
            .detach()
        )  # stance bool
        self.std = (
            torch.cat(
                [
                    torch.ones(3) * 1.5,
                    torch.ones(3) * 5,
                    torch.ones(3) * 2,
                    torch.ones(3) * 2,
                    torch.ones(12) * 2,
                    self.joint_pos_scale,
                    self.joint_vel_scale,
                    torch.ones(4) * 2 / self.freqency_scale,
                    torch.ones(8) * 1.5,
                    torch.ones(24) * 5.0,
                    self.joint_pos_scale,
                    self.joint_pos_scale,
                    torch.ones(12) * 2,
                    torch.ones(12) * 2,
                    torch.tensor([2 / self.freqency_scale]),
                ]
            )
            .to(self.device)
            .detach()
        )
        noise_std = (
            torch.cat(
                [
                    torch.zeros(6),
                    torch.ones(3) * 1e-2,
                    torch.ones(3) * 2e-2,
                    torch.ones(12) * 1e-3,
                    torch.ones(12) * 2e-2,
                    torch.ones(12) * 1e-3,
                    torch.zeros(12),
                    torch.ones(24) * 1e-3,
                    torch.ones(12) * 1e-3,
                    torch.ones(12) * 2e-2,
                    torch.zeros(12),
                    torch.zeros(12),
                    torch.zeros(1),
                ]
            )
            .to(self.device)
            .detach()
        )
        self.noise_dist = torch.distributions.Normal(0, noise_std + 1e-10)
        self.std = 1.0 / self.std
        self.obs = torch.tile(self.mean.detach().clone(), (self.num_envs, 1)).to(self.device)
        self.joint_start_indices = [12, 24, 36] + [12 * i for i in range(5, 11)]
        self.phase_start_indices = [48]

    def update(self, gc, gv, command, base_lin_v=None, base_ang_v=None, projected_gravity=None):
        # If local variables are not available, calculate here.
        # Use local variables if they are already calculated to reduce computation.

        self.is_stand = torch.norm(command, dim=1) < 0.01
        # self.update_stand_trot_phase()
        base_quat = gc[:, 3:7]
        if base_lin_v is None:
            base_lin_v = torch_utils.quat_rotate_inverse(base_quat, gv[:, 0:3])
        if base_ang_v is None:
            base_ang_v = torch_utils.quat_rotate_inverse(base_quat, gv[:, 3:6])
        if projected_gravity is None:
            projected_gravity = torch_utils.quat_rotate_inverse(base_quat, self.gravity_vecs)
        self.projected_gravity = projected_gravity
        phase = self.cpg.get_phase()
        phase_sin_cos = torch.cat([torch.sin(phase).view(-1, 4, 1), torch.cos(phase).view(-1, 4, 1)], dim=2).view(-1, 8)
        self.obs = torch.cat(
            [
                command,
                projected_gravity,
                base_lin_v,
                base_ang_v,
                gc[:, 7:19],
                gv[:, 6:18],
                self.joint_pos_history[:, -1, :],
                self.cpg.get_d_phase(),
                phase_sin_cos,
                self.joint_pos_history[:, -4, :],
                self.joint_pos_history[:, -9, :],
                self.joint_vel_history[:, -4, :],
                self.joint_vel_history[:, -9, :],
                self.prev_joint_target,
                self.prev2_joint_target,
                self.is_stand_tensor(),
            ],
            dim=1,
        )
        return self.obs

    def get_obs(self, use_raisim_order=False) -> torch.Tensor:
        obs = (self.obs - self.mean) / self.std
        if use_raisim_order:
            obs = isaac_raisim_joint_conversion(obs, self.joint_start_indices)
            # obs = isaac_raisim_foot_conversion(obs, self.phase_start_indices)
            obs = self.convert_observation_phase(obs)
        return obs

    def get_noisy_obs(self, use_raisim_order=False) -> torch.Tensor:
        obs = self.get_obs(use_raisim_order=use_raisim_order)
        return obs + self.noise_dist.sample((obs.shape[0],))

    def store_action_and_get_joint_target(self, action, use_raisim_order=False):
        if use_raisim_order:
            # raisim order to isaac
            action = self.convert_action_order(action)
        self.prev2_joint_target = self.prev_joint_target
        self.prev_joint_target = self.joint_target
        normalized_action = action.detach() * self.action_std + self.action_mean
        self.joint_target = self.apply_cpg_action_and_get_joint_target(normalized_action)
        return self.joint_target

    def reset(self, env_idx):
        self.joint_target[env_idx] = self.default_joint_pos
        self.prev_joint_target[env_idx] = self.default_joint_pos
        self.prev2_joint_target[env_idx] = self.default_joint_pos

        self.joint_pos_history[env_idx] = 0.1 * self.history_randomizer.sample()[env_idx]
        self.joint_vel_history[env_idx] = self.history_randomizer.sample()[env_idx]
        self.set_trot_phase(env_idx)

    def is_stand_tensor(self):
        return (~self.is_stand * self.base_frequency * self.freqency_scale).reshape(self.num_envs, 1)

    def convert_action_order(self, action):
        action = isaac_raisim_joint_conversion(action, [4])
        action = isaac_raisim_foot_conversion(action, [0])
        return action

    def convert_observation_phase(self, x):
        indices = torch.arange(x.shape[1]).to(self.device)
        s = 48
        indices[s + 1] = s + 2
        indices[s + 2] = s + 1
        for s in [52]:
            indices[s + 2] = s + 4
            indices[s + 3] = s + 5
            indices[s + 4] = s + 2
            indices[s + 5] = s + 3
        return torch.index_select(x, 1, indices)

    def apply_cpg_action_and_get_joint_target(self, normalized_action):
        dphase = normalized_action[:, :4]
        self.cpg.update_phase(dphase)
        h = self.cpg.get_cubic_height()
        foot_pos_target = self.foot_default_position.detach().clone()  # (num_env, 12)
        foot_pos_target[:, 2:12:3] = 0.0
        gravitational_axis_in_base = torch.unsqueeze(self.projected_gravity, 1)
        gravitational_axis_in_base = torch.tile(gravitational_axis_in_base, (1, 4, 1))
        foot_pos_target += torch.einsum(
            ("efd,ef->efd"), gravitational_axis_in_base, (self.foot_default_z + h * self.default_foot_height)
        ).view(-1, 12)
        joint_targets = solve_ik(foot_pos_target.view(-1, 3), self.feet_id)
        self.joint_target = joint_targets.view(-1, 12) + normalized_action[:, 4:16]
        return self.joint_target

    def update_stand_trot_phase(self):
        changed = ~(self.prev_is_stand == self.is_stand)
        stand_idx = ((self.is_stand > 0.5) & changed).nonzero().flatten()
        trot_idx = ((self.is_stand < 0.5) & changed).nonzero().flatten()
        self.set_stand_phase(stand_idx)
        self.set_trot_phase(trot_idx)
        self.prev_is_stand = self.is_stand

    def set_stand_phase(self, idx):
        self.cpg.reset_stance(idx)

    def set_trot_phase(self, idx):
        self.cpg.reset_trot(idx)

    def reset_stand_phase(self):
        return None

    def substep_update(self, gc, gv):
        if self.num_envs > 1:
            self.joint_pos_history[:, :-1, :] = self.joint_pos_history[:, 1:, :]
            self.joint_vel_history[:, :-1, :] = self.joint_vel_history[:, 1:, :]
            self.joint_pos_history[:, -1, :] = self.joint_target - gc[:, 7:19]
            self.joint_vel_history[:, -1, :] = gv[:, 6:18]
        else:
            self.joint_pos_history[:, :-1, :] = self.joint_pos_history[:, 1:, :].clone()
            self.joint_vel_history[:, :-1, :] = self.joint_vel_history[:, 1:, :].clone()
            self.joint_pos_history[:, -1, :] = self.joint_target - gc[:, 7:19]
            self.joint_vel_history[:, -1, :] = gv[:, 6:18]

    def get_base_frequency(self):
        return self.base_frequency

    def get_phase(self):
        return self.cpg.get_phase()
