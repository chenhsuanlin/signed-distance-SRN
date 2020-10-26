import numpy as np
import os,sys,time
import torch
import torchvision
import torchvision.transforms.functional as torchvision_F
import matplotlib.pyplot as plt
import PIL
import PIL.ImageDraw
import imageio
from easydict import EasyDict as edict

@torch.no_grad()
def tb_image(opt,tb,step,group,name,images,masks=None,num_vis=None,from_range=(0,1),poses=None,cmap="gray"):
    images = preprocess_vis_image(opt,images,masks=masks,from_range=from_range,cmap=cmap)
    num_H,num_W = num_vis or opt.tb.num_images
    images = images[:num_H*num_W]
    if poses is not None:
        rots = poses[:num_H*num_W,...,:3]
        images = torch.stack([draw_pose(opt,image,rot,size=20,width=2) for image,rot in zip(images,rots)],dim=0)
    image_grid = torchvision.utils.make_grid(images[:,:3],nrow=num_W,pad_value=1.)
    if images.shape[1]==4:
        mask_grid = torchvision.utils.make_grid(images[:,3:],nrow=num_W,pad_value=1.)[:1]
        image_grid = torch.cat([image_grid,mask_grid],dim=0)
    tag = "{0}/{1}".format(group,name)
    tb.add_image(tag,image_grid,step)

def preprocess_vis_image(opt,images,masks=None,from_range=(0,1),cmap="gray"):
    min,max = from_range
    images = (images-min)/(max-min)
    if masks is not None:
        images = torch.cat([images,masks],dim=1)
    images = images.clamp(min=0,max=1).cpu()
    if images.shape[1]==1:
        images = get_heatmap(opt,images[:,0].cpu(),cmap=cmap)
    return images

def dump_images(opt,idx,name,images,masks=None,from_range=(0,1),cmap="gray"):
    images = preprocess_vis_image(opt,images,masks=masks,from_range=from_range,cmap=cmap) # [B,3,H,W]
    images = images.cpu().permute(0,2,3,1).numpy() # [B,H,W,3]
    for i,img in zip(idx,images):
        fname = "{}/dump/{}_{}.png".format(opt.output_path,i,name)
        img_uint8 = (img*255).astype(np.uint8)
        imageio.imsave(fname,img_uint8)

def get_heatmap(opt,gray,cmap): # [N,H,W]
    color = plt.get_cmap(cmap)(gray.numpy())
    color = torch.from_numpy(color[...,:3]).permute(0,3,1,2).float() # [N,3,H,W]
    return color

def dump_meshes(opt,idx,name,meshes):
    for i,mesh in zip(idx,meshes):
        fname = "{}/dump/{}_{}.ply".format(opt.output_path,i,name)
        mesh.export(fname)

@torch.no_grad()
def vis_pointcloud(opt,vis,step,split,pred,GT=None):
    win_name = "{0}/{1}".format(opt.group,opt.name)
    pred,GT = pred.cpu().numpy(),GT.cpu().numpy()
    for i in range(opt.visdom.num_samples):
        # prediction
        data = [dict(
            type="scatter3d",
            x=[float(n) for n in points[i,:opt.visdom.num_points,0]],
            y=[float(n) for n in points[i,:opt.visdom.num_points,1]],
            z=[float(n) for n in points[i,:opt.visdom.num_points,2]],
            mode="markers",
            marker=dict(
                color=color,
                size=1,
            ),
        ) for points,color in zip([pred,GT],["blue","magenta"])]
        vis._send(dict(
            data=data,
            win="{0} #{1}".format(split,i),
            eid="{0}/{1}".format(opt.group,opt.name),
            layout=dict(
                title="{0} #{1} ({2})".format(split,i,step),
                autosize=True,
                margin=dict(l=30,r=30,b=30,t=30,),
                showlegend=False,
                yaxis=dict(
                    scaleanchor="x",
                    scaleratio=1,
                )
            ),
            opts=dict(title="{0} #{1} ({2})".format(win_name,i,step),),
        ))

@torch.no_grad()
def draw_pose(opt,image,rot_mtrx,size=15,width=1):
    mode = "RGBA" if image.shape[0]==4 else "RGB"
    image_pil = torchvision_F.to_pil_image(image.cpu()).convert("RGBA")
    draw_pil = PIL.Image.new("RGBA",image_pil.size,(0,0,0,0))
    draw = PIL.ImageDraw.Draw(draw_pil)
    center = (size,size)
    endpoint = [(size+size*p[0],size+size*p[1]) for p in rot_mtrx.t()]
    draw.line([center,endpoint[0]],fill=(255,0,0),width=width)
    draw.line([center,endpoint[1]],fill=(0,255,0),width=width)
    draw.line([center,endpoint[2]],fill=(0,0,255),width=width)
    image_pil.alpha_composite(draw_pil)
    image_drawn = torchvision_F.to_tensor(image_pil.convert(mode))
    return image_drawn
