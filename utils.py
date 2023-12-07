import math
from typing import Callable, Sequence, Tuple
import torch
import carb
import os
import pathlib
import numpy as np
import torch
import torch.nn.functional as F
import cv2
import PIL
from PIL import Image

from scipy.spatial.transform import Rotation as Rotation
from liegroups import SE3, SO3


from config import SimConfig, DebugConfig

# Stage Builder Imports
from isaac_stage.prims import create_sphere, create_triangle, translate, delete, is_defined
from isaac_stage.appliers import apply_color_to_prim

################# Robot #####################
def linear_xy_error(twist, command):
    """
    Inputs:
        twist: torch.tensor, [pos, rot, linear velocity, angular velocity], shape (13,)

        command: torch.tensor, where the first element is the command given to robot
    Output:
        error: torch.tensor, MSE of commanded and actual velocity in the xy plane
        assigns a lower error to a robot that moves faster in the commanded direction
    """
    cmd_fwd = command[0]
    quat = twist[3:7]
    quat = quat[[1,2,3,0]]
    euler = euler_of_quat(quat.cpu(),degrees=False)

    z_rot = torch.from_numpy(euler)[2]

    # NOTE: multiplying these by 10 to ensure that they are always larger than the real velocities
    # This must be done because we don't know the scale of the commanded velocities
    # Otherwise, a higher error could be assigned to the robot moving faster, which would be bad
    cmd_x = 10 * torch.cos(z_rot)
    cmd_y = 10 * torch.sin(z_rot)

    real_x = twist[7]
    real_y = twist[8]

    error = torch.sqrt((real_x - cmd_x) ** 2 + (real_y - cmd_y) ** 2)
    return error


################# Logging ####################

def apply_trav_overlay(img, trav):
    '''
    @param img: (n, m, 3) torch tensor
    @param trav: (n,m) torch tensor
    @return: numpy array, with trav overlayed on img
    '''
    trav_np = trav.cpu().numpy()*255.0
    img_np = img.cpu().numpy()*255.0

    color_map = cv2.applyColorMap(trav_np.astype(np.uint8), cv2.COLORMAP_JET)

    # Overlay the color map on the original image
    output = cv2.addWeighted(img_np.astype(np.uint8), 0.7, color_map, 0.3, 0)
    return output

def unique_log_dir(log_dir):
    i=0
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    while os.path.exists(os.path.join(log_dir, f'test_{i}')):
        i+=1
    return os.path.join(log_dir, f'test_{i}')

def get_concat_h(im1, im2):
    '''
    horizontally concatenate two images
    @param im1, im2: PIL Image objects
    @return PIL Image
    '''
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst

################## Images ##################
def crop_and_interpolate(image, new_height, new_width, K):
    '''
    Input:
        image: torch.tensor, shape (channels, height, width)
        K: torch.tensor, shape (4,4), camera intrinsinces

    Output:
        interp: torch.tensor, shape (channels, new_height, new_width), where image has first been cropped to square and then downsampled 
        K: new intrinsics matrix
    
    Requires:
        If using K, new_height must equal new_width
    '''
    image, K = crop_to_square(image, K)
    interp, K = interpolate(image, new_height, new_width, K)

    K[0,0] = -K[0,0] # Rationale below.
    K[1,1] = -K[1,1] 
    # NOTE: We had to turn the camera around 180 degrees about the y-axis
    # This caused footpaths to appear on the 'ceiling', which is a result of them being 'behind' the camera.
    # BEFORE:                       ___-->  | 
    #                      ___---'''        |   Changing the focal length negative turns the intrinsics
    #           <camera[===  (FIELD OF VIEW)|   backwards; this correctly orients them.
    #                      '''---___        |    
    #|#|#|#|#|#|#|...                '''--> |
    #  <-- FOOTPATHS

    return interp, K

def crop_to_square(image, K ):
    '''
    Input:
        image: torch.tensor, shape (channels, height, width)
        K: torch.tensor, shape (4,4), camera intrinsics

    Output:
        cropped: torch.tensor, shape (channels, side, side), where side is the smaller of height and width
        K: torch.tensor, shape (4,4), new camera intrinsics
    '''

    channels, height, width = image.size()
    if width == height:
        return image
    elif width > height:
        to_crop = (width - height) // 2  
        cropped = image[:, :, to_crop : width - to_crop] 
        K[0,2] -= to_crop
    else:
        to_crop = (height - width) // 2
        cropped = image[:, to_crop : height - to_crop, :]
        K[1,2] -= to_crop

    return cropped, K

def interpolate(image, new_height, new_width, K):
    '''
    Input:
        image: square torch.tensor, shape (channels, height, width)
        K: torch.tensor, shape (4,4), camera intrinsics

    
    Output:
        interp: square torch.tensor, shape (channels, new_height, new_width)
        K: torch.tensor, shape (4,4), new camera intrinsics

    '''
    _, oh, ow = image.size()
    interp = F.interpolate(image[None], size=(new_height, new_width))
    scale = new_width / ow

    K[[0,1,0,1],[0,1,2,2]] *= scale

    return torch.squeeze(interp), K

def save_mask(img, pred, f="live_updates/live.png"):
    """
    Writes an image with a binary mask to the file given by f
    Inputs:
        img: a torch tensor representing an rgb image
        pred: a torch tensor representing binary predictions
        f: a file path
    """
    img_to_save = img.permute(1,2,0)
    img_to_save = torch.flip(img_to_save, dims=[2])
    img_to_save = img_to_save.cpu().numpy() * 255
    mask_rgb = np.ones_like(img_to_save)
    mask_rgb[:, :, 2] = pred.numpy()*255 


    masked = cv2.addWeighted(img_to_save.astype(np.float32), .6, mask_rgb.astype(np.float32), .4, 0)
    cv2.imwrite(f,masked)

################## Math ####################

def quat_to_ang_vel(q1, q2, dt):
    '''
    Inputs: 
        q1, q2: arraylike, (4,), representing two subsequent quaternions

        dt: the time in between the quaternion measurements

    Returns: torch.tensor, representing [w_x, w_y, w_z]
    '''
    return (2 / dt) * torch.FloatTensor([
        q1[0]*q2[1] - q1[1]*q2[0] - q1[2]*q2[3] + q1[3]*q2[2],
        q1[0]*q2[2] + q1[1]*q2[3] - q1[2]*q2[0] - q1[3]*q2[1],
        q1[0]*q2[3] - q1[1]*q2[2] + q1[2]*q2[1] - q1[3]*q2[0]])

def bresenham_line(x0,y0,x1,y1):
    '''
    Requires: all inputs are integers

    Returns: the grid cells intersected by a line through the coordinates given by (x0,y0), (x1, y1)
    '''
    dx = abs(x1 - x0)
    sx =  1 if x0 < x1 else -1
    dy = -abs(y1 - y0)
    sy = 1 if y0 < y1 else -1
    error = dx + dy
    
    out = []

    while True:
        out.append((x0, y0))
        if x0 == x1 and y0 == y1:
           return out
        e2 = 2 * error
        if e2 >= dy:
            if x0 == x1:
                return out
            error = error + dy
            x0 = x0 + sx
        if e2 <= dx:
            if y0 == y1:
               return out
            error = error + dx
            y0 = y0 + sy

def get_all_pixels(img_height, img_width, device='cuda'):
    '''
    img_height, img_width: int
    
    Output:
        all_pixels: torch float tensor of size (h*w, 2), corresponding to indices for each pixel
    '''

    all_pixels = torch.meshgrid(torch.arange(img_height), torch.arange(img_width), indexing='ij')
    all_pixels = torch.stack(all_pixels, dim=-1).reshape(-1, 2)

    return all_pixels.float().to(device)


'''
when image is taken:
    project image to global traversability map -- O(pixels)
    store traversabilities in each cell (exponential average)
when edge cost queried:
    find the cells crossed by the edge-- O(resolution)
    take the average of values for cells crossed by the edge -- O(grid resolution)
'''

def cost_from_grid(edge, grid, grid_transform):
    '''
    @param edge: (tuple(2), tuple(2))
    @param grid: array-like, 2 dimensional
    @param grid_transform: tuple(2) -> tuple(2): 
    a function from world coordinates to grid indices
    '''
    v1, v2 = edge
    c1, c2 = grid_transform(v1), grid_transform(v2)
    i1, j1 = c1
    i2, j2 = c2

    cells = bresenham_line(i1, j1, i2, j2)
    total_cost = sum([grid[i][j] for i, j in cells])
    return total_cost / len(cells)

def euler_of_quat(quats, degrees=False):
    '''
    @param quats: arraylike, shape (4,), representing a quaternion in xyzw format
    @return: [roll, pitch, yaw]
    '''
    rot = Rotation.from_quat(quats)
    return rot.as_euler('xyz', degrees=degrees)

def quat_of_euler(euler, degrees : bool=True, rotation_order : str='xyz'):
    '''
    @param euler: arraylike, shape (3,), representing euler angles according to rotation_order
    @param degrees: True if the euler angles are expressed as degrees
    '''
    rot = Rotation.from_euler(rotation_order, euler,degrees=degrees)
    return rot.as_quat()

def transformation_matrix_of_pose(pose : Sequence[float], ordering : str):
    """
    Convert a translation and rotation into a 4x4 transformation matrix.
 
    Args:
        pose (float subscriptable @ 0..6): A 7 element array representing the pose [t0,t1,t2,q0,q1,q2,q3] with the order specified by ordering.
        ordering (str): Either wxyz or xyzw.
            NOTE: Orbit uses 'wxyz', ROS-WVN uses 'xyzw' internally.

    Returns:
        4x4 Transformation Matrix \in SE3 following the ordering convention specified.

    """
    assert ordering == "xyzw" or ordering == "wxyz"
    quat = np.array(pose[3:])
    quat = quat / np.linalg.norm(quat)
    matrix = SE3(rot=SO3.from_quaternion(quat, ordering=ordering), trans=pose[:3]).as_matrix() # Check order (wxyz looks correct for orbit footpath)
    matrix = matrix.astype(np.float32)
    return torch.from_numpy(matrix)

def unproject(intrinsics, extrinsics, pixels, depth):
    """
    Inputs:
        intrinsics: (B, 4, 4)

        extrinsics: (B, 4, 4)

        pixels: (B, 2, H, W)

        depth: (B, H, W)
    Returns:
        The world coordinates of pixels
    """
    batches, height, width = depth.size()
    #extrinsics[0] = extrinsics[0].permute(0, 2, 1, 3)

    depth = depth.unsqueeze(-1) # (B, H, W, 1)
    pixels = pixels.permute(0, 2, 3, 1) # (B, H, W, 2)

    P = torch.bmm(intrinsics, extrinsics) # (B, 4, 4) @ (B, 4, 4) = (B, 4, 4) 
    P_inv = torch.inverse(P) # (B, 4, 4)
    print(extrinsics)
    print(torch.inverse(extrinsics))
    # homogenous_pixels = F.pad(pixels, [0, 1], "constant", 1.0) # (B, H, W, 2) -> (B, H, W, 3) 
    # pixels_for_transform = F.pad(homogenous_pixels, [0,1], "constant", 1.0) # (B, H, W, 3) -> (B, H, W, 4)

    # pixels_for_transform = pixels_for_transform.permute(0, 3, 1, 2).flatten(start_dim=2, end_dim=3)
    #world_unscaled = torch.bmm(P_inv, pixels_for_transform) # (B, 4, 4) @ (B, 4, H*W) -?> (B, 4, H*W)

    # depth_for_transform = depth.permute(0, 3, 1, 2).flatten(start_dim=2, end_dim=3)
    # world_coords = world_unscaled[:, :3, :] * depth_for_transform # (B,3, H*W) * (B, 1, H*W) -> (B, 3, H*W)
    # return world_coords.view(batches, -1, height, width)

    world_unscaled = torch.bmm(P_inv, torch.ones(1,4,1, device='cuda'))
    print(world_unscaled)
    #breakpoint()
    return world_unscaled[:,:3,:]

def get_all_pixels_ov(img_height, img_width, batches, device='cuda'):
    '''
    Inputs:
        img_height, img_width: int
    
    Returns:
        torch.FloatTensor (B, 2, H, W), corresponding to indices for each pixel
    '''
    
    all_pixels = torch.meshgrid(torch.arange(img_height), torch.arange(img_width), indexing='ij')
    all_pixels = torch.stack(all_pixels, dim=-1).permute(2, 0, 1)
    batched_pixels = expand_to_batch_size(all_pixels, batches)

    return batched_pixels.float().to(device)

def expand_to_batch_size(tensor, batches):
    """
    Inputs:
        tensor: shape S

        batches: int

    Returns:
        torch.tensor, of shape (batches, *S)
    """

    return torch.repeat_interleave(tensor[None], batches, dim=0) 

#-------------------#
#   visualization   #
#-------------------#
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class LineGraph:
    def __init__(self):
        # Create figure for plotting
        matplotlib.use( 'tkagg')

        self.value = 0
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(1, 1, 1)
        self.xs = []
        self.ys = []
    
    def add_point(self, value):
        self.value = value

    # This function is called periodically from FuncAnimation
    def animate(self, i, xs, ys):

        # Add x and y to lists
        xs.append(i)
        ys.append(self.value)

        # Limit x and y lists to 20 items
        xs = xs[-20:]
        ys = ys[-20:]

        # Draw x and y lists
        self.ax.clear()
        self.ax.plot(xs, ys)

        # Format plot
        plt.xticks(rotation=45, ha='right')
        plt.subplots_adjust(bottom=0.30)
        plt.title('TMP102 Temperature over Time')
        plt.ylabel('Temperature (deg C)')

    def start_animation(self):
        # Set up plot to call animate() function periodically
        self.ani = animation.FuncAnimation(self.fig, self.animate, fargs=(self.xs, self.ys), interval=1000)
        print('ANIMATING')
        plt.show()

def save_raw_img(img, name):
    '''
    @param img: (n, m, 3) torch tensor
    @return: image
    '''
    np_array = img.cpu().numpy()*255.0
    image = Image.fromarray(np_array.astype(np.uint8))
    image.save(name) if name else ()
    return image



