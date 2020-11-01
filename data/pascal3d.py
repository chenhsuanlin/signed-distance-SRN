import numpy as np
import os,sys,time
import torch
import torch.nn.functional as torch_F
import torchvision
import torchvision.transforms.functional as torchvision_F
import PIL
import pickle
import scipy.io
import warnings
from easydict import EasyDict as edict

from . import base
import camera
import util

class Dataset(base.Dataset):

    def __init__(self,opt,split="train",subset=None):
        super().__init__(opt,split)
        self.cat = dict(
            car="car",
            chair="chair",
            plane="aeroplane"
        )[opt.data.pascal3d.cat]
        self.path = "data/PASCAL3D+_release1.1"
        self.path_mask = "data/pascal3d/mask/{}".format(self.cat)
        self.path_pc = "data/pascal3d/pointcloud/{}".format(self.cat)
        self.list = self.get_list(opt,split)
        if subset: self.list = self.list[:subset]
        # preload dataset
        if opt.data.preload:
            self.images = self.preload_threading(opt,self.get_image)
            self.pointclouds = self.preload_threading(opt,self.get_pointcloud,data_str="point clouds")

    def get_list(self,opt,split):
        list_fname = "data/pascal3d_imagenet_{}_{}.list".format(self.cat,split)
        list_all = [x.split() for x in open(list_fname).read().splitlines()]
        return list_all

    def __getitem__(self,idx):
        opt = self.opt
        name,mask_id = self.list[idx]
        if name=="n02690373_10374": # no mask data
            return self[np.random.randint(len(self))]
        sample = dict(idx=idx)
        aug = self.generate_augmentation(opt) if self.augment else None
        # load camera
        meta = self.get_metadata(opt,idx)
        pose_cam = camera.pose(t=[0,0,opt.camera.dist])
        assert(opt.camera.model=="orthographic")
        pose = self.cameras[idx] if opt.data.preload else self.get_camera(opt,idx,meta=meta)
        pose = camera.pose.compose([pose,pose_cam])
        if aug is not None:
            pose = self.augment_camera(opt,pose,aug,pose_cam=pose_cam)
        intr = False # there are no None tensors
        sample.update(
            pose=pose,
            intr=intr,
        )
        # load images and compute distance transform
        image = self.images[idx] if opt.data.preload else self.get_image(opt,idx)
        rgb,mask,crop_size = self.preprocess_image(opt,image,bbox=meta.bbox,aug=aug)
        meta.crop_size = crop_size
        dt = self.compute_dist_transform(opt,mask)
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
        dpc = self.pointclouds[idx] if opt.data.preload else self.get_pointcloud(opt,idx,meta=meta)
        sample.update(dpc=dpc)
        return sample

    def get_image(self,opt,idx):
        name,mask_id = self.list[idx]
        image_fname = "{0}/Images/{1}_imagenet/{2}.JPEG".format(self.path,self.cat,name)
        with warnings.catch_warnings(): # some images might contain corrupted EXIF data?
            warnings.simplefilter("ignore")
            image = PIL.Image.open(image_fname).convert("RGB")
        mask_fname = "{0}/{1}_{2}.png".format(self.path_mask,name,mask_id)
        mask = PIL.Image.open(mask_fname).convert("L")
        image = PIL.Image.merge("RGBA",(*image.split(),mask))
        return image

    def preprocess_image(self,opt,image,bbox,aug=None):
        if aug is not None:
            image = self.apply_color_jitter(opt,image,aug.color_jitter)
            image = torchvision_F.hflip(image) if aug.flip else image
            x1,y1,x2,y2 = bbox
            image = image.rotate(aug.rot_angle,center=((x1+x2)/2,(y1+y2)/2),resample=PIL.Image.BICUBIC)
            image = self.square_crop(opt,image,bbox=bbox,crop_ratio=aug.crop_ratio)
        else:
            image = self.square_crop(opt,image,bbox=bbox)
        # torchvision_F.resize/torchvision_F.resized_crop will make masks really thick....
        crop_size = image.size[0] # assume square
        image = image.resize((opt.W,opt.H))
        image = torchvision_F.to_tensor(image)
        rgb,mask = image[:3],image[3:]
        mask = (mask!=0).float()
        if opt.data.bgcolor:
            # replace background color using mask
            rgb = rgb*mask+opt.data.bgcolor*(1-mask)
        rgb = rgb*2-1
        return rgb,mask,crop_size

    def get_camera(self,opt,idx,meta=None):
        azim = torch.tensor(meta.cam.azim/180*np.pi).float()
        elev = torch.tensor(meta.cam.elev/180*np.pi).float()
        theta = torch.tensor(meta.cam.theta/180*np.pi).float()
        Rz = camera.angle_to_rotation_matrix(azim,axis="Z")
        Rx = camera.angle_to_rotation_matrix(elev-np.pi/2,axis="X")
        R2d = camera.angle_to_rotation_matrix(-theta,axis="Z")
        R = R2d@Rx@Rz
        R_trans = torch.tensor([[0,0,1],
                                [-1,0,0],
                                [0,1,0]],dtype=torch.float32)
        pose = camera.pose(R=R@R_trans)
        return pose

    def augment_camera(self,opt,pose,aug,pose_cam=None):
        if aug.flip:
            raise NotImplementedError
        if aug.rot_angle:
            angle = torch.tensor(aug.rot_angle)*np.pi/180
            R = camera.angle_to_rotation_matrix(-angle,axis="Z") # in-plane rotation
            rot_inplane = camera.pose(R=R)
            pose = camera.pose.compose([pose,camera.pose.invert(pose_cam),rot_inplane,pose_cam])
        return pose

    def get_pointcloud(self,opt,idx,meta=None):
        name,mask_id = self.list[idx]
        pc_fname = "{}/{:02d}.npy".format(self.path_pc,meta.cad_idx)
        pc = torch.from_numpy(np.load(pc_fname)).float()
        pc = torch.stack([pc[:,1],-pc[:,2],-pc[:,0]],dim=-1)
        # rescale to full image size
        pc *= meta.cam.viewport*meta.cam.focal/meta.cam.dist
        # rescale to canonical image size (cropped, scaled to [-1,1])
        pc *= 2./float(meta.crop_size)
        dpc = dict(
            points=pc,
            normals=torch.zeros_like(pc),
        )
        return dpc

    def square_crop(self,opt,image,bbox=None,crop_ratio=1.):
        # crop to canonical image size
        x1,y1,x2,y2 = bbox
        h,w = y2-y1,x2-x1
        yc,xc = (y1+y2)/2,(x1+x2)/2
        S = h*3 if opt.data.pascal3d.cat=="car" else max(h,w)*1.2
        # crop with random size (cropping out of boundary = padding)
        S2 = S*crop_ratio
        image = torchvision_F.crop(image,top=int(yc-S2/2),left=int(xc-S2/2),height=int(S2),width=int(S2))
        return image

    def get_metadata(self,opt,idx):
        name,mask_id = self.list[idx]
        meta_fname = "{}/Annotations/{}_imagenet/{}.mat".format(self.path,self.cat,name)
        meta = scipy.io.loadmat(meta_fname,squeeze_me=True)["record"]["objects"].item()
        multi = meta["viewpoint"].shape!=()
        viewpoint = meta["viewpoint"][int(mask_id)] if multi else meta["viewpoint"].item()
        cad_index = meta["cad_index"][int(mask_id)] if multi else meta["cad_index"].item()
        bbox = meta["bbox"][int(mask_id)] if multi else meta["bbox"].item()
        meta = edict(
            cam=edict(
                viewport=float(viewpoint["viewport"].item()),
                focal=float(viewpoint["focal"].item()),
                dist=float(viewpoint["distance"].item()),
                azim=float(viewpoint["azimuth"].item()),
                elev=float(viewpoint["elevation"].item()),
                theta=float(viewpoint["theta"].item()),
            ),
            cad_idx=int(cad_index),
            bbox=torch.from_numpy(bbox),
        )
        return meta

    def __len__(self):
        return len(self.list)
