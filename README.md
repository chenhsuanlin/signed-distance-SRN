## SDF-SRN: Learning Signed Distance 3D Object Reconstruction from Static Images
[Chen-Hsuan Lin](https://chenhsuanlin.bitbucket.io/),
[Chaoyang Wang](https://mightychaos.github.io/),
and [Simon Lucey](http://ci2cv.net/people/simon-lucey/)  
Advances in Neural Information Processing Systems (NeurIPS), 2020  

Project page: https://chenhsuanlin.bitbucket.io/signed-distance-SRN  
Paper: https://chenhsuanlin.bitbucket.io/signed-distance-SRN/paper.pdf  
arXiv preprint: https://arxiv.org/abs/2010.10505  

<p align="center"><img src="https://chenhsuanlin.bitbucket.io/signed-distance-SRN/teaser.png"></p>
<p align="center"><img src="images/teaser_mov.gif"></p>

We provide PyTorch code for both the ShapeNet and PASCAL3D+ experiments.

--------------------------------------

### Prerequisites

This code is developed with Python3 (`python3`). PyTorch 1.4+ is required.  
It is recommended to install the dependencies with `conda` by running
```bash
conda env create --file requirements.yaml python=3
```
This creates a conda environment named `sdfsrn-env`. Activate it with
```bash
conda activate sdfsrn-env
```
You may want to install with `virtualenv`; however, this repository depends on [VIGRA](http://ukoethe.github.io/vigra/) to compute the distance transforms, which does not seem to be pip installable.
Some workarounds would include (a) installing VIGRA from source, or (b) replacing the VIGRA distance transform function with `scipy.ndimage.distance_transform_edt` (significantly slower).

--------------------------------------

### Dataset

- #### ShapeNet
	Download the ShapeNet renderings of [Kato et al.](https://arxiv.org/abs/1711.07566) from the [DVR](https://github.com/autonomousvision/differentiable_volumetric_rendering) repository (33GB):  
	(this file is huge and takes a long time to fully unzip, so we extract only the 3 categories of interest in this work)
  ```bash
  wget https://s3.eu-central-1.amazonaws.com/avg-projects/differentiable_volumetric_rendering/data/NMR_Dataset.zip
  unzip NMR_Dataset.zip NMR_Dataset/02691156/* # airplane
  unzip NMR_Dataset.zip NMR_Dataset/02958343/* # car
  unzip NMR_Dataset.zip NMR_Dataset/03001627/* # chair
  rm NMR_Dataset.zip
  ```
  In the `data/NMR_Dataset` directory, download the post-processed surface point clouds:
  ```bash
  wget https://cmu.box.com/shared/static/yvencf3ts8itfgyuh5sap9q7dy5r1elg.gz
  tar -zxvf yvencf3ts8itfgyuh5sap9q7dy5r1elg.gz
  rm yvencf3ts8itfgyuh5sap9q7dy5r1elg.gz
  ```
  There should be a `pointcloud3.npz` within each shape directory, along with the original `pointcloud.npz`. You can check with
  ```bash
  ls NMR_Dataset/02691156/10155655850468db78d106ce0a280f87
  ```
- #### PASCAL3D+
  Download the [PASCAL3D+](https://cvgl.stanford.edu/projects/pascal3d.html) (v1.1) dataset under the `data` directory (7.7GB):
  ```bash
  wget ftp://cs.stanford.edu/cs/cvgl/PASCAL3D+_release1.1.zip
  unzip PASCAL3D+_release1.1.zip
  rm PASCAL3D+_release1.1.zip
  ```
  Also under the `data` directory, download the object masks and ground-truth point clouds for the 3 categories (23MB):
  ```bash
  wget https://cmu.box.com/shared/static/uyz0txthw0ufjwury0f3z3iuhqdbaet9.gz
  tar -zxvf uyz0txthw0ufjwury0f3z3iuhqdbaet9.gz
  rm uyz0txthw0ufjwury0f3z3iuhqdbaet9.gz
  ```

--------------------------------------

### Pretrained models
First, create a directory to store the pretrained models:
```bash
mkdir -p pretrained
```
Then under `pretrained`, download the pretrained model(s) by running the commands
```bash
# ShapeNet (trained on multi-view renderings, 615MB each)
wget https://cmu.box.com/shared/static/cgrzlaudm2ojs5l3nmmbbr7ovsvbqhtv.ckpt -O shapenet_airplane.ckpt  # airplane
wget https://cmu.box.com/shared/static/lclrhwae5xu6z7f2fc3qnkeon3q5ljfg.ckpt -O shapenet_car.ckpt  # car
wget https://cmu.box.com/shared/static/58dsppp8hq0yqj216tqm573or9porq2m.ckpt -O shapenet_chair.ckpt  # chair
# PASCAL3D+ (197MB each)
wget https://cmu.box.com/shared/static/gvslqtye7p0pzgaspmwvq7pggnmxsu3x.ckpt -O pascal3d_airplane.ckpt # airplane
wget https://cmu.box.com/shared/static/kh8mrrufol3u1mm6duaym5sygfd42d5p.ckpt -O pascal3d_car.ckpt # car
wget https://cmu.box.com/shared/static/ty0ywyeud1n1n9uu169xoag9m35me267.ckpt -O pascal3d_chair.ckpt # chair
```

--------------------------------------

### Compiling the CUDA libraries
The Chamfer distance function can be compiled by running `python3 setup.py install` under `external/chamfer3D`.
The source code is taken/modified from the [AtlasNet](https://github.com/ThibaultGROUEIX/AtlasNet) repository.  
When compiling CUDA code, you may need to modify `CUDA_PATH` accordingly.  

--------------------------------------

### Running the code

- #### Evaluating the downloaded pretrained models
  ```bash
  # ShapeNet (trained on multi-view renderings)
  python3 evaluate.py --model=sdf_srn --yaml=options/shapenet/sdf_srn.yaml --name=airplane_pretrained --data.shapenet.cat=plane --load=pretrained/shapenet_airplane.ckpt --tb= --visdom= --eval.vox_res=128
  python3 evaluate.py --model=sdf_srn --yaml=options/shapenet/sdf_srn.yaml --name=car_pretrained --data.shapenet.cat=car --load=pretrained/shapenet_car.ckpt --tb= --visdom= --eval.vox_res=128
  python3 evaluate.py --model=sdf_srn --yaml=options/shapenet/sdf_srn.yaml --name=chair_pretrained --data.shapenet.cat=chair --load=pretrained/shapenet_chair.ckpt --tb= --visdom= --eval.vox_res=128
  # PASCAL3D+
  python3 evaluate.py --model=sdf_srn --yaml=options/pascal3d/sdf_srn.yaml --name=airplane_pretrained --data.pascal3d.cat=plane --load=pretrained/pascal3d_airplane.ckpt --tb= --visdom= --eval.vox_res=128 --eval.icp
  python3 evaluate.py --model=sdf_srn --yaml=options/pascal3d/sdf_srn.yaml --name=car_pretrained --data.pascal3d.cat=car --load=pretrained/pascal3d_car.ckpt --tb= --visdom= --eval.vox_res=128 --eval.icp
  python3 evaluate.py --model=sdf_srn --yaml=options/pascal3d/sdf_srn.yaml --name=chair_pretrained --data.pascal3d.cat=chair --load=pretrained/pascal3d_chair.ckpt --tb= --visdom= --eval.vox_res=128 --eval.icp
  ```
  This will create the following files in the output directory (e.g. `output/sdf_srn_pascal3d/car_pretrained`):
  - `chamfer.txt`: the (bidirectional) Chamfer distance error for each example.
  - `dump/*_mesh.ply`: the resulting 3D meshes (from zero isosurface extraction with marching cubes).
  - `dump/*.png`: images including input/rendered RGB images, input/predicted masks, depth maps and surface normal maps.
  - `dump/vis.html`: a webpage to visualize all the images for convenience.
  
  The overall Chamfer distance error (the numbers reported in the paper) will also be shown on screen.  
  Note that it takes longer to evaluate the PASCAL3D+ models since we run ICP to pre-align the predictions to the ground-truth shapes.

- #### Training from scratch
  To train SDF-SRN, we first quickly pretrain the generator with a spherical SDF for 1000 iterations with:
  ```bash
  # ShapeNet
  python3 train.py --model=sdf_srn_pretrain --yaml=options/shapenet/sdf_srn_pretrain.yaml
  # PASCAL3D+
  python3 train.py --model=sdf_srn_pretrain --yaml=options/pascal3d/sdf_srn_pretrain.yaml
  ```
  This helps SDF-SRN converge to a feasible solution, otherwise it may get stuck in bad local minima.
  
  For the main training:
  ```bash
  # ShapeNet (~100K iterations for airplanes and cars, ~200K iterations for chairs)
  python3 train.py --model=sdf_srn --yaml=options/shapenet/sdf_srn.yaml --name=airplane --data.shapenet.cat=plane --max_epoch=24 --loss_weight.shape_silh=1
  python3 train.py --model=sdf_srn --yaml=options/shapenet/sdf_srn.yaml --name=car --data.shapenet.cat=car --max_epoch=27
  python3 train.py --model=sdf_srn --yaml=options/shapenet/sdf_srn.yaml --name=chair --data.shapenet.cat=chair --max_epoch=28
  # PASCAL3D+ (~30K iterations)
  python3 train.py --model=sdf_srn --yaml=options/pascal3d/sdf_srn.yaml --name=airplane --data.pascal3d.cat=plane --freq.eval=30 --freq.ckpt=30 --max_epoch=500
  python3 train.py --model=sdf_srn --yaml=options/pascal3d/sdf_srn.yaml --name=car --data.pascal3d.cat=car --freq.eval=10 --freq.ckpt=10 --max_epoch=170
  python3 train.py --model=sdf_srn --yaml=options/pascal3d/sdf_srn.yaml --name=chair --data.pascal3d.cat=chair --freq.eval=60 --freq.ckpt=60 --max_epoch=900
  ```
  The above command for ShapeNet runs single-view training on multi-view data.
  To train on single-view ShapeNet data (only 1 view is available per CAD model) with the reported settings, run
  ```bash
  # single-view ShapeNet chairs (~50K iterations)
  python3 train.py --model=sdf_srn --yaml=options/shapenet/sdf_srn.yaml --name=chair_1view_1kcad --data.shapenet.cat=chair --data.shapenet.train_view=1 --data.train_sub=1000 --data.augment.brightness=0.2 --data.augment.contrast=0.2 --data.augment.saturation=0.2 --data.augment.hue=0.5 --freq.eval=100 --freq.ckpt=100 --max_epoch=800
  ```
  This trains on a subset of 1000 chair CAD models with 1 viewpoint each while randomly jittering the colors.

  To evaluate the trained models:
  ```bash
  # ShapeNet
  python3 evaluate.py --model=sdf_srn --yaml=options/shapenet/sdf_srn.yaml --name=airplane --data.shapenet.cat=plane --tb= --visdom= --eval.vox_res=128 --resume
  python3 evaluate.py --model=sdf_srn --yaml=options/shapenet/sdf_srn.yaml --name=car --data.shapenet.cat=car --tb= --visdom= --eval.vox_res=128 --resume
  python3 evaluate.py --model=sdf_srn --yaml=options/shapenet/sdf_srn.yaml --name=chair --data.shapenet.cat=chair --tb= --visdom= --eval.vox_res=128 --resume
  # PASCAL3D+
  python3 evaluate.py --model=sdf_srn --yaml=options/pascal3d/sdf_srn.yaml --name=airplane --data.pascal3d.cat=plane --tb= --visdom= --eval.vox_res=128 --eval.icp --resume
  python3 evaluate.py --model=sdf_srn --yaml=options/pascal3d/sdf_srn.yaml --name=car --data.pascal3d.cat=car --tb= --visdom= --eval.vox_res=128 --eval.icp --resume
  python3 evaluate.py --model=sdf_srn --yaml=options/pascal3d/sdf_srn.yaml --name=chair --data.pascal3d.cat=chair --tb= --visdom= --eval.vox_res=128 --eval.icp --resume
  ```
  The expected output is similar to those described above (in the pretrained models section).

- #### Visualizing the results
  We have included code to visualize the training over TensorBoard. The TensorBoard events include the following:
  - **SCALARS**: the losses and bidirectional Chamfer distances (for both training and validation sets).  
  - **IMAGES**: visualization of the RGB/mask/depth/normal images.
  
  We also provide visualization of dense point clouds sampled on the zero isosurface in Visdom.

- #### General usage of the codebase
  The simplest command to run training is:
  ```bash
  python3 train.py --model=sdf_srn
  ```
  This will run `model/sdf_srn.py` as the main engine with `options/sdf_srn.yaml` as the main config file.
  Note that `sdf_srn` is hierarchically inherited from `implicit` and `base`, which makes the codebase customizable.  
  The complete configuration will be printed upon execution. To override specific options, add `--<key>=value` or `--<key1>.<key2>=value` (and so on) to the command line. The configuration will be loaded as the variable `opt` throughout the codebase.  
  If you want to reproduce the reported results, load preset configurations with the `yaml` option (details below).
  
  Some tips on using and understanding the codebase:
  - The computation graph for forward/backprop is stored in `var` throughout the codebase.
  - The losses are stored in `loss`. To add a new loss function, just implement it in `compute_loss()` and add its weight to `opt.loss_weight.<name>`. It will automatically be added to the overall loss and logged to Tensorboard.
  - To resume from a previous checkpoint, add `--resume=<epoch_number>`, or just `--resume` to resume from the latest checkpoint.
  - (to be continued....)
  
--------------------------------------

If you find our code useful for your research, please cite
```
@inproceedings{lin2020sdfsrn,
  title={SDF-SRN: Learning Signed Distance 3D Object Reconstruction from Static Images},
  author={Lin, Chen-Hsuan and Wang, Chaoyang and Lucey, Simon},
  booktitle={Advances in Neural Information Processing Systems ({NeurIPS})},
  year={2020}
}
```

Please contact me (chlin@cmu.edu) if you have any questions!
