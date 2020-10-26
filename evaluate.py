import numpy as np
import os,sys,time
import torch
import importlib

import options
from util import log

log.process(os.getpid())
log.title("[{}] (evaluate SDF-SRN)".format(sys.argv[0]))

opt_cmd = options.parse_arguments(sys.argv[1:])
opt = options.set(opt_cmd=opt_cmd)

with torch.cuda.device(opt.device):

    model = importlib.import_module("model.{}".format(opt.model))
    m = model.Model(opt)

    m.load_dataset(opt,eval_split="test" if opt.data.dataset=="shapenet" else \
                                  "val" if opt.data.dataset=="pascal3d" else None)
    m.build_networks(opt)
    m.restore_checkpoint(opt)
    m.setup_visualizer(opt)

    m.evaluate(opt)
