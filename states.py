'''
Classes to replace ros messages (maybe make these monads/promises?)

'''

import torch
import numpy as np
from utils import euler_of_quat, quat_to_ang_vel
from collections import defaultdict, deque

class LearningState:
    def __init__(self, items = {'loss_total', 'loss_trav', 'loss_reco'}):
        '''
        A class that maintains relevant data over training
        @param items: the items from learning to include in the class
        '''
        self.items = items 
        self.data = defaultdict(list)
    
    def add_one_step(self, res):
        '''
        Add one training step to the state.

        res is a dictionary, output by traversability_estimator.train()
        '''
        res: dict or dict[str,int]
        for k in res.keys():
            if k in self.items:
                self.data[k].append(res[k])

class RobotState:
    def __init__(self):
        self.base_dim = 7 + 6 # pose + twist
        self.transform = torch.zeros(7)
        self.transform[-1] = 1 # w quaternion should be 1
        
        # memory of robot, where 4:7th elements of each item contain the quaternions of robot's rotation
        self.pose_memory = deque(maxlen=100)

        # previous 6 images for robot
        self.img_memory = deque(maxlen=6)

        self.base_lin_vel = torch.zeros(3)
        self.base_ang_vel = torch.zeros(3)
        self.transform = torch.zeros(7)

    def reset(self):
        self.transform = torch.zeros(7)
        self.transform[-1] = 1 # w quaternion should be 1
        self.pose_memory.clear()

    def add_image(self, img):
        self.img_memory.append(img)

    def set_pose(self, transform):
        '''
        transform: [x,y,z,qx,qy,qz, qw]
        '''
        self.transform = torch.FloatTensor(transform)
        self.pose_memory.append(self.transform)

    def set_twist(self, obs):
        '''
        obs given by Anymal._compute_observation
        '''

        self.base_lin_vel = torch.from_numpy(obs[:3])
        self.base_ang_vel = torch.from_numpy(obs[3:6])
        self.cmd_vel = torch.from_numpy(obs[9:12])
        self.joint_pos = torch.from_numpy(obs[12:24])
        self.joint_vel = torch.from_numpy(obs[24:36])

    def set_velocities(self, dt):
        '''
        set the linear and angular velocity properties of the robot from its last two poses
        '''
        if len(self.pose_memory) < 2:
            return

        cur_pos = self.pose_memory.pop()
        last_pos = self.pose_memory.pop()

        self.base_lin_vel  = (cur_pos[:3] - last_pos[:3]) / dt
        self.base_ang_vel = quat_to_ang_vel(cur_pos[3:], last_pos[3:], dt)

        self.pose_memory.append(last_pos)
        self.pose_memory.append(cur_pos)

    def to_torch(self, device='cpu'):
        torch_state = torch.zeros(self.base_dim, dtype=torch.float32)
        torch_state[:3] = self.transform[:3]

        torch_state[3:7] = self.transform[3:]
        torch_state[7:10] = self.base_lin_vel
        torch_state[10:13] = self.base_ang_vel
        return torch_state.to(device)
    
    def is_standing(self):

        mem = list(self.pose_memory)

        if len(mem) < 20:
            return True

        for l in mem:
            qx, qy, qz, qw = l[3], l[4], l[5], l[6]
            gz = qx*qx - qy*qy - qz*qz + qw*qw
            if gz > 0:
                return True
            
        return False

    def is_moving(self, threshold, n=20):

        locs = list(self.pose_memory)
        if len(locs) < n + 1:
            return True

        recents = locs[-(n+1):]
        displacement = torch.norm(recents[0][:2] - recents[-1][:2]) # only consider displacement along xy plane
        return displacement > threshold

    def _get_theta(self):
        '''
        return rotation of the robot in the xy plane 
        '''
        return euler_of_quat(self.transform[3:])[0]

    def get_xyt_pose(self):
        return [self.transform[0].item(), self.transform[1].item(), self._get_theta()]

