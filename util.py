import numpy as np
import os,sys,time
import shutil
import datetime
import torch
import torch.nn.functional as torch_F
import termcolor
import socket
import contextlib
from easydict import EasyDict as edict

# convert to colored strings
def red(message,**kwargs): return termcolor.colored(str(message),color="red",attrs=[k for k,v in kwargs.items() if v is True])
def green(message,**kwargs): return termcolor.colored(str(message),color="green",attrs=[k for k,v in kwargs.items() if v is True])
def blue(message,**kwargs): return termcolor.colored(str(message),color="blue",attrs=[k for k,v in kwargs.items() if v is True])
def cyan(message,**kwargs): return termcolor.colored(str(message),color="cyan",attrs=[k for k,v in kwargs.items() if v is True])
def yellow(message,**kwargs): return termcolor.colored(str(message),color="yellow",attrs=[k for k,v in kwargs.items() if v is True])
def magenta(message,**kwargs): return termcolor.colored(str(message),color="magenta",attrs=[k for k,v in kwargs.items() if v is True])
def grey(message,**kwargs): return termcolor.colored(str(message),color="grey",attrs=[k for k,v in kwargs.items() if v is True])

def get_time(sec):
    d = int(sec//(24*60*60))
    h = int(sec//(60*60)%24)
    m = int((sec//60)%60)
    s = int(sec%60)
    return d,h,m,s

class Log:
    def __init__(self): pass
    def process(self,pid):
        print(grey("Process ID: {}".format(pid),bold=True))
    def title(self,message):
        print(yellow(message,bold=True,underline=True))
    def info(self,message):
        print(magenta(message,bold=True))
    def options(self,opt,level=0):
        for key,value in sorted(opt.items()):
            if isinstance(value,(dict,edict)):
                print("   "*level+cyan("* ")+green(key)+":")
                self.options(value,level+1)
            else:
                print("   "*level+cyan("* ")+green(key)+":",yellow(value))
    def loss_train(self,opt,ep,lr,loss,timer):
        message = grey("[train] ",bold=True)
        message += "epoch {}/{}".format(cyan(ep,bold=True),opt.max_epoch)
        message += ", lr:{}".format(yellow("{:.2e}".format(lr),bold=True))
        message += ", loss:{}".format(red("{:.3e}".format(loss.all),bold=True))
        message += ", time:{}".format(blue("{0}-{1:02d}:{2:02d}:{3:02d}".format(*get_time(timer.elapsed)),bold=True))
        message += " (ETA:{})".format(blue("{0}-{1:02d}:{2:02d}:{3:02d}".format(*get_time(timer.arrival))))
        print(message)
    def loss_eval(self,opt,loss,chamfer=None):
        message = grey("[eval] ",bold=True)
        message += "loss:{}".format(red("{:.3e}".format(loss.all),bold=True))
        if chamfer is not None:
            message += ", chamfer:{}|{}".format(green("{:.4f}".format(chamfer[0]),bold=True),
                                                green("{:.4f}".format(chamfer[1]),bold=True))
        print(message)
log = Log()

def update_timer(opt,timer,ep,it_per_ep):
    momentum = 0.99
    timer.elapsed = time.time()-timer.start
    timer.it = timer.it_end-timer.it_start
    # compute speed with moving average
    timer.it_mean = timer.it_mean*momentum+timer.it*(1-momentum) if timer.it_mean is not None else timer.it
    timer.arrival = timer.it_mean*it_per_ep*(opt.max_epoch-ep)

# move tensors to device in-place
def move_to_device(X,device):
    if isinstance(X,dict):
        for k,v in X.items():
            X[k] = move_to_device(v,device)
    elif isinstance(X,list):
        for i,e in enumerate(X):
            X[i] = move_to_device(e,device)
    elif isinstance(X,tuple) and hasattr(X,"_fields"): # collections.namedtuple
        dd = X._asdict()
        dd = move_to_device(dd,device)
        return type(X)(**dd)
    elif isinstance(X,torch.Tensor):
        return X.to(device=device)
    return X

def to_dict(D,dict_type=dict):
    D = dict_type(D)
    for k,v in D.items():
        if isinstance(v,dict):
            D[k] = to_dict(v,dict_type)
    return D

def get_child_state_dict(state_dict,key):
    return { ".".join(k.split(".")[1:]): v for k,v in state_dict.items() if k.startswith("{}.".format(key)) }

def restore_checkpoint(opt,model,load_name=None,resume=False):
    assert((load_name is None)==(resume is not False)) # resume can be True/False or epoch numbers
    if resume:
        load_name = "{0}/latest.ckpt".format(opt.output_path) if resume==True else \
                    "{0}/checkpoint/ep{1}.ckpt".format(opt.output_path,opt.resume)
    checkpoint = torch.load(load_name,map_location=opt.device)
    # load individual (possibly partial) children modules
    for name,child in model.graph.named_children():
        child_state_dict = get_child_state_dict(checkpoint["graph"],name)
        if child_state_dict:
            print("restoring {}...".format(name))
            child.load_state_dict(child_state_dict)
    for key in model.__dict__:
        if key.split("_")[0] in ["optim","sched"] and key in checkpoint and resume:
            print("restoring {}...".format(key))
            getattr(model,key).load_state_dict(checkpoint[key])
    if resume:
        if resume is not True: assert(resume==checkpoint["epoch"])
        ep,it = checkpoint["epoch"],checkpoint["iter"]
        print("resuming from epoch {0} (iteration {1})".format(ep,it))
    else: ep,it = None,None
    return ep,it

def save_checkpoint(opt,model,ep,it,latest=False,children=None):
    os.makedirs("{0}/checkpoint".format(opt.output_path),exist_ok=True)
    if children is not None:
        graph_state_dict = { k: v for k,v in model.graph.state_dict().items() if k.startswith(children) }
    else: graph_state_dict = model.graph.state_dict()
    checkpoint = dict(
        epoch=ep,
        iter=it,
        graph=graph_state_dict,
    )
    for key in model.__dict__:
        if key.split("_")[0] in ["optim","sched"]:
            checkpoint.update({ key: getattr(model,key).state_dict() })
    torch.save(checkpoint,"{0}/latest.ckpt".format(opt.output_path))
    if not latest:
        shutil.copy("{0}/latest.ckpt".format(opt.output_path),
                    "{0}/checkpoint/ep{1}.ckpt".format(opt.output_path,ep))

def check_socket_open(hostname,port):
    s = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
    is_open = False
    try:
        s.bind((hostname,port))
    except socket.error:
        is_open = True
    finally:
        s.close()
    return is_open

def get_layer_dims(layers):
    # return a list of tuples (k_in,k_out)
    return list(zip(layers[:-1],layers[1:]))

@contextlib.contextmanager
def suppress(stdout=False,stderr=False):
    with open(os.devnull,"w") as devnull:
        if stdout: old_stdout,sys.stdout = sys.stdout,devnull
        if stderr: old_stderr,sys.stderr = sys.stderr,devnull
        try: yield
        finally:
            if stdout: sys.stdout = old_stdout
            if stderr: sys.stderr = old_stderr
