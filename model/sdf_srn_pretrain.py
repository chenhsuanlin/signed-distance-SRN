import numpy as np
import os,sys,time
import torch
import torch.nn.functional as torch_F
import tqdm
from easydict import EasyDict as edict

from . import implicit
import camera
import util
from util import log

# ============================ main engine for training and evaluation ============================

class Model(implicit.Model):

    def __init__(self,opt):
        super().__init__(opt)

    def load_dataset(self,opt): return

    def train(self,opt):
        # before training
        log.title("TRAINING START")
        self.timer = edict(start=time.time(),it_mean=None)
        self.ep = 0
        self.it = self.iter_start
        # training
        self.graph.train()
        loader = tqdm.trange(opt.max_iter,desc="training",leave=False)
        for it in loader:
            # train iteration
            var = edict()
            loss = self.train_iteration(opt,var,loader)
        # after train epoch
        self.save_checkpoint(opt,ep=1,it=self.it)
        # after training
        if opt.tb: self.tb.close()
        if opt.visdom: self.vis.close()
        log.title("TRAINING DONE")

    @torch.no_grad()
    def evaluate(self,opt,ep=None,training=False): return

    @torch.no_grad()
    def visualize(self,opt,var,step=0,split="train"): return

    def save_checkpoint(self,opt,ep=0,it=0,latest=False):
        util.save_checkpoint(opt,self,ep=ep,it=it,children=("generator",))
        if not latest:
            log.info("checkpoint saved: ({0}) {1}, epoch {2} (iteration {3})".format(opt.group,opt.name,ep,it))

# ============================ computation graph for forward/backprop ============================

class Graph(implicit.Graph):

    def __init__(self,opt):
        super().__init__(opt)

    def forward(self,opt,var,training=False):
        var.latent = torch.randn(opt.batch_size,opt.latent_dim,device=opt.device)*opt.latent_std
        var.impl_func = self.generator.forward(opt,var.latent) # [B,3,H,W]
        return var

    def compute_loss(self,opt,var,training=False):
        loss = edict()
        if opt.impl.occup:
            logit,occup_gt = self.get_sphere_occup_GT(opt,var.impl_func)
            loss.sphere = self.BCE_loss(logit,occup_gt)
        else:
            level,sdf_gt = self.get_sphere_sdf_GT(opt,var.impl_func)
            loss.sphere = self.MSE_loss(level,sdf_gt)
        return loss

    def get_sphere_occup_GT(self,opt,impl_func,N=10000):
        lower,upper = opt.impl.sdf_range
        points_3D = torch.rand(opt.batch_size,N,3,device=opt.device)
        points_3D = points_3D*(upper-lower)+lower
        logit = impl_func.forward(opt,points_3D)
        occup_gt = (points_3D.norm(dim=-1,keepdim=True)<opt.impl.pretrain_radius).float()
        return logit,occup_gt

    def get_sphere_sdf_GT(self,opt,impl_func,N=10000):
        lower,upper = opt.impl.sdf_range
        points_3D = torch.rand(opt.batch_size,N,3,device=opt.device)
        points_3D = points_3D*(upper-lower)+lower
        level = impl_func.forward(opt,points_3D)
        sdf_gt = points_3D.norm(dim=-1,keepdim=True)-opt.impl.pretrain_radius
        return level,sdf_gt
