### Creating the ground-truth 3D point clouds

We use the [virtual scanner](https://github.com/wang-ps/O-CNN/tree/master/virtual_scanner) to sample point clouds from only the outer surface of the CAD models.


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
  (coming soon!)
