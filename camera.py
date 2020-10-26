import numpy as np
import os,sys,time
import torch
import torch.nn.functional as torch_F
import collections
from easydict import EasyDict as edict

import util

class Pose():

    def __call__(self,R=None,t=None):
        assert(R is not None or t is not None)
        if R is None:
            if not isinstance(t,torch.Tensor): t = torch.tensor(t)
            R = torch.eye(3,device=t.device).repeat(*t.shape[:-1],1,1)
        elif t is None:
            if not isinstance(R,torch.Tensor): R = torch.tensor(R)
            t = torch.zeros(R.shape[:-1],device=R.device)
        else:
            if not isinstance(R,torch.Tensor): R = torch.tensor(R)
            if not isinstance(t,torch.Tensor): t = torch.tensor(t)
        assert(R.shape[:-1]==t.shape and R.shape[-2:]==(3,3))
        R = R.float()
        t = t.float()
        pose = torch.cat([R,t[...,None]],dim=-1) # [...,3,4]
        assert(pose.shape[-2:]==(3,4))
        return pose

    def invert(self,pose,use_inverse=False):
        R,t = pose[...,:3],pose[...,3:]
        R_inv = R.inverse() if use_inverse else R.transpose(-1,-2)
        t_inv = (-R_inv@t)[...,0]
        pose_inv = self(R=R_inv,t=t_inv)
        return pose_inv

    def compose(self,pose_list):
        # pose_new(x) = poseN(...(pose2(pose1(x)))...)
        pose_new = pose_list[0]
        for pose in pose_list[1:]:
            pose_new = self.compose_pair(pose_new,pose)
        return pose_new

    def compose_pair(self,pose_a,pose_b):
        # pose_new(x) = pose_b(pose_a(x))
        R_a,t_a = pose_a[...,:3],pose_a[...,3:]
        R_b,t_b = pose_b[...,:3],pose_b[...,3:]
        R_new = R_b@R_a
        t_new = (R_b@t_a+t_b)[...,0]
        pose_new = self(R=R_new,t=t_new)
        return pose_new

pose = Pose()

def to_hom(X):
    X_hom = torch.cat([X,torch.ones_like(X[...,:1])],dim=-1)
    return X_hom

def world2cam(X,pose): # [B,N,3]
    X_hom = to_hom(X)
    return X_hom@pose.transpose(-1,-2)
def cam2img(X,cam_intr):
    return X@cam_intr.transpose(-1,-2)
def img2cam(X,cam_intr):
    return X@cam_intr.inverse().transpose(-1,-2)
def cam2world(X,pose):
    X_hom = to_hom(X)
    pose_inv = Pose().invert(pose)
    return X_hom@pose_inv.transpose(-1,-2)

def angle_to_rotation_matrix(a,axis):
    roll = dict(X=1,Y=2,Z=0)[axis]
    O = torch.zeros_like(a)
    I = torch.ones_like(a)
    M = torch.stack([torch.stack([a.cos(),-a.sin(),O],dim=-1),
                     torch.stack([a.sin(),a.cos(),O],dim=-1),
                     torch.stack([O,O,I],dim=-1)],dim=-2)
    M = M.roll((roll,roll),dims=(-2,-1))
    return M

def get_camera_grid(opt,batch_size,intr=None):
    # compute image coordinate grid
    if opt.camera.model=="perspective":
        y_range = torch.arange(opt.H,dtype=torch.float32,device=opt.device).add_(0.5)
        x_range = torch.arange(opt.W,dtype=torch.float32,device=opt.device).add_(0.5)
        Y,X = torch.meshgrid(y_range,x_range) # [H,W]
        xy_grid = torch.stack([X,Y],dim=-1).view(-1,2) # [HW,2]
    elif opt.camera.model=="orthographic":
        assert(opt.H==opt.W)
        y_range = torch.linspace(-1,1,opt.H,device=opt.device)
        x_range = torch.linspace(-1,1,opt.W,device=opt.device)
        Y,X = torch.meshgrid(y_range,x_range) # [H,W]
        xy_grid = torch.stack([X,Y],dim=-1).view(-1,2) # [HW,2]
    xy_grid = xy_grid.repeat(batch_size,1,1) # [B,HW,2]
    if opt.camera.model=="perspective":
        grid_3D = img2cam(to_hom(xy_grid),intr) # [B,HW,3]
    elif opt.camera.model=="orthographic":
        grid_3D = to_hom(xy_grid) # [B,HW,3]
    return xy_grid,grid_3D

def get_center_and_ray(opt,pose,intr=None,offset=None): # [HW,2]
    batch_size = len(pose)
    xy_grid,grid_3D = get_camera_grid(opt,batch_size,intr=intr) # [B,HW,3]
    # compute center and ray
    if opt.camera.model=="perspective":
        if offset is not None:
            grid_3D[...,:2] += offset
        center_3D = torch.zeros(batch_size,1,3,device=opt.device) # [B,1,3]
    elif opt.camera.model=="orthographic":
        center_3D = torch.cat([xy_grid,torch.zeros_like(xy_grid[...,:1])],dim=-1) # [B,HW,3]
    # transform from camera to world coordinates
    grid_3D = cam2world(grid_3D,pose) # [B,HW,3]
    center_3D = cam2world(center_3D,pose) # [B,HW,3]
    ray = grid_3D-center_3D # [B,HW,3]
    return center_3D,ray

def get_3D_points_from_depth(opt,center,ray,depth,multi_samples=False):
    if multi_samples: center,ray = center[:,:,None],ray[:,:,None]
    # x = c+dv
    points_3D = center+ray*depth # [B,HW,3]/[B,HW,N,3]/[N,3]
    return points_3D

def get_depth_from_3D_points(opt,center,ray,points_3D):
    # d = ||x-c||/||v|| (x-c and v should be in same direction)
    depth = (points_3D-center).norm(dim=-1,keepdim=True)/ray.norm(dim=-1,keepdim=True) # [B,HW,1]
    return depth
