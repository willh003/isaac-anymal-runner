import torch
import torch.nn as nn
import numpy as np


class CPG(nn.Module):
    def __init__(self, shape, decimation=8, base_frequency=1.0, dt=1 / 60.0, device="cuda:0"):
        self.shape = shape
        self.base_frequency = base_frequency * torch.ones(shape, dtype=torch.float, requires_grad=False).to(device)
        self.base_frequency_f = base_frequency
        self.dt = dt
        self.base_phase_step = dt * self.base_frequency
        self.phases = torch.zeros(shape, dtype=torch.float, requires_grad=False).to(device)
        self.dphases = torch.zeros(shape, dtype=torch.float, requires_grad=False).to(device)
        self.uniform_dist = torch.distributions.uniform.Uniform(
            torch.zeros(*shape).to(device).detach(), torch.ones(*shape).to(device).detach()
        )
        self.device = device
        self.decimation = decimation
        self.trot_phase = torch.tensor([0, 0.5, 0.5, 0], requires_grad=False).to(device) - 0.25
        self.stance_phase = torch.tensor([0, 0.0, 0.0, 0], requires_grad=False).to(device) - 0.25

    def reset(self, idx):
        self.reset_random(idx)
        self.reset_dphase(idx)

    def reset_trot(self, idx):
        self.phases[idx] = self.trot_phase
        self.phases = self.round_phase(self.phases)
        self.reset_dphase(idx)
        self.set_base_frequency(idx, self.base_frequency_f)

    def reset_stance(self, idx):
        self.phases[idx] = self.stance_phase
        self.phases = self.round_phase(self.phases)
        self.reset_dphase(idx)
        self.set_base_frequency(idx, 0.0)

    def set_base_frequency(self, idx, frequency):
        self.base_frequency[idx] = frequency
        self.base_phase_step = self.base_frequency * self.dt

    def reset_random(self, idx):
        self.phases[idx] = self.uniform_dist.sample()[idx]

    def reset_dphase(self, idx):
        self.dphases[idx] = self.base_phase_step[idx]

    def update_phase(self, phase_increment):
        self.dphases = self.base_phase_step + phase_increment * self.decimation / (2 * np.pi)
        self.phases += self.dphases
        self.phases = self.round_phase(self.phases)

    def get_cubic_height(self):
        # return h=0 for stance and h=1 as highest for swing phase using cubic spline.
        t = self.phases * 4
        h_lift = torch.where(t < 1.0, -2 * t**3 + 3 * t**2, 2 * (t - 1) ** 3 - 3 * (t - 1) ** 2 + 1)
        h = torch.where(self.phases < 0.5, h_lift, torch.zeros_like(h_lift))
        return h

    def get_phase(self):
        return self.angle_mod(self.phases * 2 * np.pi)

    def get_d_phase(self):
        return self.dphases * 2 * np.pi / self.decimation

    def round_phase(self, phases):
        phases -= torch.floor(phases)
        return phases

    def wrap_angle(self, angle):
        angle -= 2 * np.pi * torch.floor(angle / (2 * np.pi))
        return angle

    def angle_mod(self, angle):
        return self.wrap_angle(angle + np.pi) - np.pi
