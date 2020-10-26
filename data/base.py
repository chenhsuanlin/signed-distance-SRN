import numpy as np
import os,sys,time
import torch
import torchvision
import PIL
import tqdm
import threading,queue
import vigra
from easydict import EasyDict as edict

import util

class Dataset(torch.utils.data.Dataset):

    def __init__(self,opt,split):
        super().__init__()
        self.opt = opt
        self.split = split
        self.augment = split=="train" and opt.data.augment

    def setup_loader(self,opt,shuffle=False,drop_last=True):
        loader = torch.utils.data.DataLoader(self,
            batch_size=opt.batch_size,
            num_workers=opt.data.num_workers,
            shuffle=shuffle,
            drop_last=drop_last,
        )
        print("number of samples: {}".format(len(self)))
        return loader

    def get_list(self,opt):
        raise NotImplementedError

    def preload_worker(self,data_list,load_func,q,lock,idx_tqdm):
        while True:
            idx = q.get()
            data_list[idx] = load_func(self.opt,idx)
            with lock:
                idx_tqdm.update()
            q.task_done()

    def preload_threading(self,opt,load_func,data_str="images"):
        data_list = [None]*len(self)
        q = queue.Queue(maxsize=len(self))
        idx_tqdm = tqdm.tqdm(range(len(self)),desc="preloading {}".format(data_str),leave=False)
        for i in range(len(self)): q.put(i)
        lock = threading.Lock()
        for ti in range(opt.data.num_workers):
            t = threading.Thread(target=self.preload_worker,
                                 args=(data_list,load_func,q,lock,idx_tqdm),daemon=True)
            t.start()
        q.join()
        idx_tqdm.close()
        assert(all(map(lambda x: x is not None,data_list)))
        return data_list

    def __getitem__(self,idx):
        raise NotImplementedError

    def get_image(self,opt,idx):
        raise NotImplementedError

    def generate_augmentation(self,opt):
        brightness = opt.data.augment.brightness or 0.
        contrast = opt.data.augment.contrast or 0.
        saturation = opt.data.augment.saturation or 0.
        hue = opt.data.augment.hue or 0.
        color_jitter = torchvision.transforms.ColorJitter.get_params(
            brightness=(1-brightness,1+brightness),
            contrast=(1-contrast,1+contrast),
            saturation=(1-saturation,1+saturation),
            hue=(-hue,hue),
        )
        aug = edict(
            color_jitter=color_jitter,
            flip=np.random.randn()>0 if opt.data.augment.hflip else False,
            crop_ratio=1+(np.random.rand()*2-1)*opt.data.augment.crop_scale if opt.data.augment.crop_scale else 1,
            rot_angle=(np.random.rand()*2-1)*opt.data.augment.rotate if opt.data.augment.rotate else 0,
        )
        return aug

    def apply_color_jitter(self,opt,image,color_jitter):
        mode = image.mode
        if mode!="L":
            chan = image.split()
            rgb = PIL.Image.merge("RGB",chan[:3])
            rgb = color_jitter(rgb)
            rgb_chan = rgb.split()
            image = PIL.Image.merge(mode,rgb_chan+chan[3:])
        return image

    def compute_dist_transform(self,opt,mask,intr=None):
        assert(mask.shape[0]==1)
        mask = mask[0] # [H,W]
        mask_binary = mask!=0 # make sure only 0/1
        # use boundaryDistanceTransform instead of distanceTransform (for 0.5 pixel precision)
        bdt = vigra.filters.boundaryDistanceTransform(mask_binary.float().numpy())
        if opt.camera.model=="orthographic":
            # assume square images for now....
            assert(opt.H==opt.W)
            bdt *= 2./float(opt.H) # max distance from H (W) to 2
        elif opt.camera.model=="perspective":
            # assume same focal length for x/y for now....
            assert(intr[0,0]==intr[1,1])
            bdt /= float(intr[0,0])
        bdt = torch.from_numpy(bdt)[None]
        return bdt

    def __len__(self):
        return len(self.list)
