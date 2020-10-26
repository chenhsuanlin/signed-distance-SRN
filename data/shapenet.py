import numpy as np
import os,sys,time
import torch
import torch.nn.functional as torch_F
import torchvision
import torchvision.transforms.functional as torchvision_F
import PIL
import pickle
from easydict import EasyDict as edict

from . import base
import camera
import util

class Dataset(base.Dataset):

    def __init__(self,opt,split="train",subset=None):
        super().__init__(opt,split)
        assert(opt.H==64 and opt.W==64)
        assert(opt.camera.model=="perspective")
        self.cat_id_all = dict(
            car="02958343",
            chair="03001627",
            plane="02691156",
        )
        self.cat_id = list(self.cat_id_all.values()) if opt.data.shapenet.cat is None else \
                      [v for k,v in self.cat_id_all.items() if k in opt.data.shapenet.cat.split(",")]
        self.path = "data/NMR_Dataset"
        self.list_cads = self.get_list(opt,split)
        if subset: self.list_cads = self.list_cads[:subset]
        self.list = self.get_list_with_viewpoints(opt,split)
        # preload dataset
        if opt.data.preload:
            self.images = self.preload_threading(opt,self.get_image)
            self.cameras = self.preload_threading(opt,self.get_camera,data_str="cameras")
            self.pointclouds = self.preload_threading(opt,self.get_pointcloud,data_str="point clouds")

    def get_list(self,opt,split):
        cads = []
        for c in self.cat_id:
            list_fname = "data/shapenet_{}_{}.list".format(c,split)
            cads += [(c,m,i) for i,m in enumerate(open(list_fname).read().splitlines())]
        return cads

    def get_list_with_viewpoints(self,opt,split):
        self.num_views = 24
        view = (opt.data.shapenet.train_view if split=="train" else opt.data.shapenet.test_view) or self.num_views
        with open("data/shapenet_view.pkl","rb") as file:
            view_idx = pickle.load(file)
        view_idx = { k:v for k,v in view_idx.items() if k in self.cat_id }
        samples = [(c,m,i,v) for c,m,i in self.list_cads for v in view_idx[c][split][m][:view]]
        return samples

    def __getitem__(self,idx):
        opt = self.opt
        cad_idx = self.list[idx][2]
        sample = dict(
            idx=idx,
            cad_idx=cad_idx,
        )
        aug = self.generate_augmentation(opt) if self.augment else None
        # load camera
        pose_cam = camera.pose(R=[[-1,0,0],
                                  [0,-1,0],
                                  [0,0,-1]],
                               t=[0,0,opt.camera.dist])
        intr,pose = self.cameras[idx] if opt.data.preload else self.get_camera(opt,idx)
        if aug is not None:
            pose = self.augment_camera(opt,pose,aug,pose_cam=pose_cam)
        sample.update(
            pose=pose,
            intr=intr,
        )
        # load images and compute distance transform
        image = self.images[idx] if opt.data.preload else self.get_image(opt,idx)
        rgb,mask = self.preprocess_image(opt,image,aug=aug)
        dt = self.compute_dist_transform(opt,mask,intr=intr)
        sample.update(
            rgb_input_map=rgb,
            mask_input_map=mask,
            dt_input_map=dt,
        )
        # vectorize images (and randomly sample)
        rgb = rgb.permute(1,2,0).view(opt.H*opt.W,3)
        mask = mask.permute(1,2,0).view(opt.H*opt.W,1)
        dt = dt.permute(1,2,0).view(opt.H*opt.W,1)
        if self.split=="train" and opt.impl.rand_sample:
            ray_idx = torch.randperm(opt.H*opt.W)[:opt.impl.rand_sample]
            rgb,mask,dt = rgb[ray_idx],mask[ray_idx],dt[ray_idx]
            sample.update(ray_idx=ray_idx)
        sample.update(
            rgb_input=rgb,
            mask_input=mask,
            dt_input=dt,
        )
        # load GT point cloud (only for validation!)
        dpc = self.pointclouds[idx] if opt.data.preload else self.get_pointcloud(opt,idx)
        sample.update(dpc=dpc)
        return sample

    def get_image(self,opt,idx):
        c,m,i,v = self.list[idx]
        image_fname = "{0}/{1}/{2}/image/{3:04d}.png".format(self.path,c,m,v)
        mask_fname = "{0}/{1}/{2}/mask/{3:04d}.png".format(self.path,c,m,v)
        image = PIL.Image.open(image_fname).convert("RGB")
        mask = PIL.Image.open(mask_fname).split()[0]
        image = PIL.Image.merge("RGBA",list(image.split())+[mask])
        return image

    def preprocess_image(self,opt,image,aug=None):
        if aug is not None:
            image = self.apply_color_jitter(opt,image,aug.color_jitter)
            image = torchvision_F.hflip(image) if aug.flip else image
            image = image.rotate(aug.rot_angle,resample=PIL.Image.BICUBIC)
            image = self.square_crop(opt,image,crop_ratio=aug.crop_ratio)
        # torchvision_F.resize/torchvision_F.resized_crop will make masks really thick....
        image = image.resize((opt.W,opt.H))
        image = torchvision_F.to_tensor(image)
        rgb,mask = image[:3],image[3:]
        if opt.data.bgcolor:
            # replace background color using mask
            rgb = rgb*mask+opt.data.bgcolor*(1-mask)
        rgb = rgb*2-1
        return rgb,mask

    def get_camera(self,opt,idx):
        c,m,i,v = self.list[idx]
        cam_fname = "{0}/{1}/{2}/cameras.npz".format(self.path,c,m)
        cam = np.load(cam_fname)
        focal = 1.8660254037844388
        intr = torch.tensor([[focal*opt.W,0,opt.W/2],
                             [0,focal*opt.H,opt.H/2],
                             [0,0,1]])
        extr = torch.from_numpy(cam["world_mat_{}".format(v)][:3]).float()
        pose = camera.pose(R=extr[:,:3],t=extr[:,3])
        return intr,pose

    def augment_camera(self,opt,pose,aug,pose_cam=None):
        if aug.flip:
            raise NotImplementedError
        if aug.rot_angle:
            angle = torch.tensor(aug.rot_angle)*np.pi/180
            R = camera.angle_to_rotation_matrix(angle,axis="X") # in-place rotation
            rot_inplane = camera.pose(R=R)
            pose = camera.pose.compose([pose,camera.pose.invert(pose_cam),rot_inplane,pose_cam])
        return pose

    def get_pointcloud(self,opt,idx):
        c,m,i,v = self.list[idx]
        pc_fname = "{0}/{1}/{2}/pointcloud3.npz".format(self.path,c,m)
        npz = np.load(pc_fname)
        dpc = dict(
            points=torch.from_numpy(npz["points"]).float(),
            normals=torch.from_numpy(npz["normals"]).float(),
        )
        return dpc

    def square_crop(self,opt,image,crop_ratio=1.):
        # crop with random size (cropping out of boundary = padding)
        W,H = image.size
        H2,W2 = H*crop_ratio,W*crop_ratio
        top = max(0,H/2-H2/2)
        left = max(0,W/2-W2/2)
        image = torchvision_F.crop(image,top=int(top),left=int(left),height=int(H2),width=int(W2))
        return image

    def __len__(self):
        return len(self.list)
