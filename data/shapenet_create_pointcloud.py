import numpy as np
import os,sys
import multiprocessing as mp

atlasnet_path = "."
categories = ["02691156","02958343","03001627"]
num_proc = 24

def worker(q_in,c,m):
    while True:
        m = q_in.get()
        if m is None: break
        # load point cloud from AtlasNet
        points = np.load("{}/ShapeNetV1PointCloud/{}/{}.points.ply.npy".format(atlasnet_path,c,m))
        assert(points.shape==(30000,3))
        # fit shape into zero-centered unit cube
        center = (points.max(axis=0)+points.min(axis=0))/2
        size = (points.max(axis=0)-points.min(axis=0)).max()
        points = (points-center)/size
        # load point cloud data from NMR
        data = dict(np.load("NMR_Dataset/{}/{}/pointcloud.npz".format(c,m)))
        data["points"] = points
        np.savez("NMR_Dataset/{}/{}/pointcloud3.npz".format(c,m),**data)
        print(c,m)

for c in categories:
    M = sorted(os.listdir("{}/ShapeNetV1PointCloud/{}".format(atlasnet_path,c)))
    M = [m.split(".")[0] for m in M]
    q_in = mp.Queue()
    for m in M: q_in.put(m)
    for i in range(num_proc): q_in.put(None)
    processes = [mp.Process(target=worker,args=(q_in,c,m)) for _ in range(num_proc)]
    print("starting...")
    for p in processes: p.start()
    for p in processes: p.join()
