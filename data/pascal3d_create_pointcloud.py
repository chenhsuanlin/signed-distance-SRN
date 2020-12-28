import numpy as np
import os
from ocnn.virtualscanner import VirtualScanner
from ocnn.virtualscanner import DirectoryTreeScanner

root_original = "../PASCAL3D+_release1.1/CAD"
root_processed = "../pascal3d/pointcloud"

categories = ["car","chair","aeroplane"]

num_random_points = 50000

for c in categories:
    path_orig = "{}/{}".format(root_original,c)
    path_out = "{}/{}".format(root_processed,c)
    os.makedirs(path_out,exist_ok=True)
    os.system("rm {}/*".format(path_out))
    # run virtual scanner
    scanner = DirectoryTreeScanner(view_num=14,flags=False,normalize=False)
    scanner.scan_tree(input_base_folder=path_orig,
                      output_base_folder=path_out,
                      num_threads=48)
    # convert to ply and then to numpy
    raws = [f for f in sorted(os.listdir(path_out)) if f.endswith(".points")]
    plys = [f.split(".")[0]+".ply" for f in raws]
    npys = [f.split(".")[0]+".npy" for f in raws]
    for raw,ply,npy in zip(raws,plys,npys):
        # convert to ply
        os.system("./parse {0}/{1} {0}/{2}".format(path_out,raw,ply))
        array = []
        # convert to numpy
        with open("{}/{}".format(path_out,ply)) as file:
            for i,line in enumerate(file):
                if i<12: continue # assume 12 lines of header, may not be true for PLY in general
                array.append([float(n) for n in line.strip().split()])
        array = np.stack(array)
        array = array[:,:3] # discard surface normal, just keep xyz
        # randomly sample 50k points and save
        array = np.random.permutation(array)[:num_random_points]
        np.save("{}/{}".format(path_out,npy),array)

