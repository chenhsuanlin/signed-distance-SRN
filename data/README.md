### Creating the ground-truth 3D point clouds

- #### ShapeNet
  The surface point clouds for ShapeNet are readily available from the AtlasNet [repo](https://github.com/ThibaultGROUEIX/AtlasNet).
  Run [this script](https://github.com/ThibaultGROUEIX/AtlasNet/blob/master/dataset/download_shapenet_pointclouds.sh) to download and extract the point clouds of all object categories (19G).
  You should have the ShapeNet data (from the main README) downloaded.  
  Under this directory (`data`), run
  ```bash
  python3 shapenet_create_pointcloud.py
  ```
  A separate `pointcloud3.npz` file will be created for each CAD model in the corresponding folder in `NMR_Dataset`.
  You would need to modify the script, mostly in the first few lines:
  - `atlasnet_path`: where `ShapeNetV1PointCloud` was extracted.
  - `categories`: all ShapeNet category IDs to process.
  - `num_proc`: number of parallel processes.


- #### PASCAL3D+
  We use the [virtual scanner](https://github.com/wang-ps/O-CNN/tree/master/virtual_scanner) to sample point clouds from only the outer surface of the CAD models.  
  You can either follow the installation instructions from the official repo, or alternatively, you can follow the below steps to install via conda.
  
  First, create a new conda environment via
  ```bash
  conda env create --file vscan-requirements.yaml python=3
  ```
  This creates a conda environment named `vscan-env`. Activate it with
  ```bash
  conda activate vscan-env
  ```
  Under this directory (`data`), clone the repo containing the virtual scanner package and compile:
  ```bash
  git clone https://github.com/wang-ps/O-CNN
  cd O-CNN/virtual_scanner
  python3 setup.py install
  ```
  If all goes well, you should be able to import `ocnn` in Python. Check for errors by running `python3 -c "import ocnn"`.

  Next, run `make` to compile a simple parser that converts the virtual scanner data format to PLY files.
  An executable named `parse` will be created.
  
  Finally, run
  ```bash
  python3 pascal3d_create_pointcloud.py
  ```
  The corresponding virtual scanner output `*.points`, the converted PLY (`*.ply`) and Numpy (`*.npy`) files will be created for each CAD model in the corresponding folder.
  You would need to modify the script, mostly in the first few lines:
  - `root_original`: where the PASCAL3D+ CAD models (in OFF format) are stored.
  - `root_processed`: where the processed point clouds will be stored.
  - `categories`: all PASCAL3D+ category names to process.
  - `num_random_points`: number of random points to densely sample.
