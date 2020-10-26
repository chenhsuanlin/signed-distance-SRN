import numpy as np
import os,sys,time
import torch
import threading,queue
import mcubes
import trimesh
from easydict import EasyDict as edict

import chamfer_3D
import util,util_vis

@torch.no_grad()
def get_dense_3D_grid(opt,var,N=None):
    batch_size = len(var.idx)
    N = N or opt.eval.vox_res
    range_min,range_max = opt.eval.range
    grid = torch.linspace(range_min,range_max,N+1,device=opt.device)
    points_3D = torch.stack(torch.meshgrid(grid,grid,grid),dim=-1) # [N,N,N,3]
    points_3D = points_3D.repeat(batch_size,1,1,1,1) # [B,N,N,N,3]
    return points_3D

@torch.no_grad()
def compute_level_grid(opt,impl_func,points_3D,N=None):
    # process points in batches
    level_all = []
    N = points_3D.shape[1]
    slice_batch_size = 1
    for i in range(0,N,slice_batch_size):
        points_3D_batch = points_3D[:,i:i+slice_batch_size] # [B,N',N,N,3]
        level_batch = impl_func.forward(opt,points_3D_batch) # [B,N',N,N,1]
        level_all.append(level_batch)
    level = torch.cat(level_all,dim=1)[...,0] # [B,N,N,N]
    return level

@torch.no_grad()
def compute_chamfer_dist(opt,var):
    batch_size = len(var.idx)
    points_3D = get_dense_3D_grid(opt,var) # [B,N,N,N,3]
    level_vox = compute_level_grid(opt,var.impl_func,points_3D=points_3D) # [B,N,N,N,K]
    *level_grids, = level_vox.cpu().numpy() # numpy equivalent to torch.unbind
    meshes,pointclouds = convert_to_explicit(opt,level_grids,isoval=0.,to_pointcloud=True)
    var.mesh_pred = meshes
    var.dpc_pred = torch.tensor(pointclouds,dtype=torch.float32,device=opt.device)
    if opt.eval.icp:
        var.dpc_pred = ICP(opt,var.dpc_pred,var.dpc.points)
    dist_acc,dist_comp,_,_ = chamfer_distance(opt,X1=var.dpc_pred,X2=var.dpc.points)
    var.cd_acc = dist_acc.mean(dim=1)
    var.cd_comp = dist_comp.mean(dim=1)
    return dist_acc.mean(),dist_comp.mean()

def convert_to_explicit(opt,level_grids,isoval=0.,to_pointcloud=False):
    N = len(level_grids)
    meshes = [None]*N
    pointclouds = [None]*N if to_pointcloud else None
    threads = [threading.Thread(target=convert_to_explicit_worker,
                                args=(opt,i,level_grids[i],isoval,meshes),
                                kwargs=dict(pointclouds=pointclouds),
                                daemon=False) for i in range(N)]
    for t in threads: t.start()
    for t in threads: t.join()
    if to_pointcloud:
        pointclouds = np.stack(pointclouds,axis=0)
        return meshes,pointclouds
    else: return meshes

def convert_to_explicit_worker(opt,i,level_vox_i,isoval,meshes,pointclouds=None):
    # use marching cubes to convert implicit surface to mesh
    vertices,faces = mcubes.marching_cubes(level_vox_i,isovalue=isoval)
    assert(level_vox_i.shape[0]==level_vox_i.shape[1]==level_vox_i.shape[2])
    S = level_vox_i.shape[0]
    range_min,range_max = opt.eval.range
    vertices = vertices/S*(range_max-range_min)+range_min
    mesh = trimesh.Trimesh(vertices,faces)
    meshes[i] = mesh
    if pointclouds is not None:
        # randomly sample on mesh to get uniform dense point cloud
        if len(mesh.triangles)!=0:
            points = mesh.sample(opt.eval.num_points)
        else: points = np.zeros([opt.eval.num_points,3])
        pointclouds[i] = points

def chamfer_distance(opt,X1,X2):
    B = len(X1)
    N1 = X1.shape[1]
    N2 = X2.shape[1]
    assert(X1.shape[2]==3)
    dist_1 = torch.zeros(B,N1,device=opt.device)
    dist_2 = torch.zeros(B,N2,device=opt.device)
    idx_1 = torch.zeros(B,N1,dtype=torch.int32,device=opt.device)
    idx_2 = torch.zeros(B,N2,dtype=torch.int32,device=opt.device)
    chamfer_3D.forward(X1,X2,dist_1,dist_2,idx_1,idx_2)
    return dist_1.sqrt(),dist_2.sqrt(),idx_1,idx_2

def ICP(opt,X1,X2,num_iter=50): # [B,N,3]
    assert(len(X1)==len(X2))
    for it in range(num_iter):
        d1,d2,idx,_ = chamfer_distance(opt,X1,X2)
        X2_corresp = torch.zeros_like(X1)
        for i in range(len(X1)):
            X2_corresp[i] = X2[i][idx[i].long()]
        t1 = X1.mean(dim=-2,keepdim=True)
        t2 = X2_corresp.mean(dim=-2,keepdim=True)
        U,S,V = ((X1-t1).transpose(1,2)@(X2_corresp-t2)).svd(some=True)
        R = V@U.transpose(1,2)
        R[R.det()<0,2] *= -1
        X1 = (X1-t1)@R.transpose(1,2)+t2
    return X1

