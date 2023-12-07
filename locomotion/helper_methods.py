import os
import yaml
import inspect
import torch
import numpy as np 



def obs_generator(obs, info, keys={}, post_process_fns={}):
    """Note this is code code is taken from Legged_GYM repo from ETH 
            TODO: add source
    """
    obs_l = []
    for k in keys:
        if k == "obs":
            obs_l.append(obs.reshape(obs.shape[0], -1))
        elif k in info.keys():
            obs_l.append(info[k].reshape(obs.shape[0], -1))
        if k in post_process_fns.keys():
            obs_l[-1] = post_process_fns[k](obs_l[-1])
    return torch.cat(obs_l, dim=1)



def to_torch(x, dtype=torch.float, device='cuda:0', requires_grad=False):
    return torch.tensor(x, dtype=dtype, device=device, requires_grad=requires_grad)

@torch.jit.script
def wrap_to_pi(angles):
    angles %= 2 * np.pi
    angles -= 2 * np.pi * (angles > np.pi)
    return angles


def spacer(marker='~', skips=2):
    print('\n' * skips)
    print(marker * 40,'|')
    print('\n' * skips)


def load_yaml(file_location):
    with open(file_location, 'r') as file:
        output = yaml.safe_load(file)
    return output

def path_with_respect_to_helper_method(file_loc):

    parent_dir = os.path.dirname(os.path.abspath(inspect.getfile(spacer)))
    if len(parent_dir) == 0:
        parent_dir = '.'

    output_dir = ''.join([parent_dir, '/', file_loc])

    spacer()
    print('File Location: ', output_dir)
    return output_dir
