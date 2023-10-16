import os
import time
import torch
import numpy as np
import json
import _pickle
import math
from multiprocessing import Pool
import fcntl
import random
import csv
import pynvml,time

def getAvaliableDevice(gpu=[1,2,3,4,0,5],min_mem=18000,left=False):
# def getAvaliableDevice(gpu=[6],min_mem=10000,left=False):
    """
    :param gpu:
    :param min_mem:
    :param left:
    :return:
    """
    # return 0
    pynvml.nvmlInit()
    t=int(time.strftime("%H", time.localtime()))

    if t>=23 or t <8:
        left=False # do not leave any GPUs
    #else:
        #left=True

    min_num=3
    dic = {0: 5, 1: 0, 2: 1, 3: 2, 4: 3, 5: 4,-1: -1}  # just for 120 server
    ava_gpu = -1

    while ava_gpu == -1:
        avaliable=[]
        for i in gpu:
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            # handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
            utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
            # print((utilization.gpu))
            if (meminfo.free / 1024 ** 2)>min_mem:
                avaliable.append(dic[i])
            # elif i ==0 and (meminfo.free / 1024 ** 2)>16000:
            #     avaliable.append(dic[i])

            elif (meminfo.free / 1024 ** 2)>16000 and utilization.gpu<20:
                avaliable.append(dic[i])

        if len(avaliable)==0 or (left and len(avaliable)<=1):
            ava_gpu = -1
            time.sleep(5)
            continue
        ava_gpu= avaliable[0]
    return ava_gpu

# def write_shared_file(file_name,content):
#     nowtime=time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
#     content[0]=nowtime+" "+content[0]
#     with open(file_name,'a+') as f:
#         fcntl.flock(f,fcntl.LOCK_EX)
#         f.writelines(content)
#         fcntl.flock(f,fcntl.LOCK_UN)

def write_csv_file(file_name,content):
    nowtime=time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
    content["time"]=nowtime
    to_write_head = False
    if not os.path.exists(file_name):
        to_write_head=True
    with open(file_name,'a+') as f:
        writer=csv.DictWriter(f,content.keys())
        fcntl.flock(f,fcntl.LOCK_EX)
        if to_write_head:
            writer.writeheader()
        writer.writerow(content)
        # for key, value in content.items:
        #     writer.writerow([key, value])
        fcntl.flock(f,fcntl.LOCK_UN)

import pandas as pd

def write_excel_file(path_root,content):

    file=os.path.join(path_root, content["dataset"]+ str(content["net"])+ str(content["num"])+str(content["seed"])+ "shot.xlsx")

    if not os.path.exists(file):
        dff = pd.DataFrame(columns=["methods"])
        dff.to_excel(file)

    df=pd.read_excel(file, sheet_name='Sheet1')

    task=content['s'].upper()[0]+"2"+ content['t'].upper()[0]

    row=content["method"]
    if row not in df["methods"]:
        df.loc[row] = 0 # add a row 
    if task not in df.columns:
        df[task] = 0  # add a colum
    df.loc[row,task]=content["best_test_acc"]
    df.to_excel(file)


def get_para_num(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}

def setup_seed(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    # np.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def serialize(obj, path, in_json=False):
    if isinstance(obj, np.ndarray):
        np.save(path, obj)
    elif in_json:
        with open(path, "w") as file:
            json.dump(obj, file, indent=2)
    else:
        with open(path, 'wb') as file:
            _pickle.dump(obj, file)

def unserialize(path):
    suffix = os.path.basename(path).split(".")[-1]
    if suffix == "npy":
        return np.load(path)
    elif suffix == "json":
        with open(path, "r") as file:
            return json.load(file)
    else:
        with open(path, 'rb') as file:

            return _pickle.load(file)

def set_gpu(x):
    os.environ['CUDA_VISIBLE_DEVICES'] = x
    print('using gpu:', x)

def check_dir(path):
    '''
    Create directory if it does not exist.
        path:           Path of directory.
    '''
    if not os.path.exists(path):
        os.mkdir(path)

def uniform(size, tensor):
    bound = 1.0 / math.sqrt(size)
    if tensor is not None:
        tensor.data.uniform_(-bound, bound)

def count_accuracy(logits, label):
    pred = torch.argmax(logits, dim=1).view(-1)
    label = label.view(-1)
    accuracy = 100 * pred.eq(label).float().mean().item()
    return accuracy

class Timer():
    def __init__(self):
        self.o = time.time()

    def measure(self, p=1):
        x = (time.time() - self.o) / float(p)
        x = int(x)
        if x >= 3600:
            return '{:.1f}h'.format(x / 3600)
        if x >= 60:
            return '{}m'.format(round(x / 60))
        return '{}s'.format(x)
import datetime

def log(log_file_path, string):
    '''
    Write one line of log into screen and file.
        log_file_path: Path of log file.
        string:        String to write in log file.
    '''
    time=datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open(log_file_path, 'a+') as f:
        f.write(string+"  "+time + '\n')
        f.flush()
    print(string)

def store(st,writer,epoch=None):

    update_step=len(st["loss"])

    for step in range(update_step):
        writer.add_scalars("l_s_s",{"loss":st["loss"][step],
                                                  "stop_gate":st["stop_gates"][step],
                                                  "scores":st["scores"][step]
                                                  },step)

    for item in ["grads","input_gates","forget_gates"]:
        for step in range(update_step):
            d={}
            for index,v in enumerate(st[item][step]):
                d["layer"+str(index)]=v
            writer.add_scalars(item, d, step)
def interleave_offsets(batch, nu):
    groups = [batch // (nu + 1)] * (nu + 1)
    for x in range(batch - sum(groups)):
        groups[-x - 1] += 1
    offsets = [0]
    for g in groups:
        offsets.append(offsets[-1] + g)
    assert offsets[-1] == batch
    return offsets


def interleave(xy, batch):
    nu = len(xy) - 1
    offsets = interleave_offsets(batch, nu)
    xy = [[v[offsets[p]:offsets[p + 1]] for p in range(nu + 1)] for v in xy]
    for i in range(1, nu + 1):
        xy[0][i], xy[i][i] = xy[i][i], xy[0][i]
    return [torch.cat(v, dim=0) for v in xy]

def linear_rampup(current, rampup_length=0):
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current / rampup_length, 0.0, 1.0)
        return float(current)
def exp_rampup(current, rampup_length=0):
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(np.exp(2*current / rampup_length)-0.99, 0.0, 1.0)
        return float(current)

# import torch
import torch.nn as nn
class Bn_Controller:
    def __init__(self):
        """
        freeze_bn and unfreeze_bn must appear in pairs
        """
        self.backup = {}

    def freeze_bn(self, model):
        assert self.backup == {}
        for name, m in model.named_modules():
            if isinstance(m, nn.SyncBatchNorm) or isinstance(m, nn.BatchNorm2d):
                self.backup[name + '.running_mean'] = m.running_mean.data.clone()
                self.backup[name + '.running_var'] = m.running_var.data.clone()
                self.backup[name + '.num_batches_tracked'] = m.num_batches_tracked.data.clone()

    def unfreeze_bn(self, model):
        for name, m in model.named_modules():
            if isinstance(m, nn.SyncBatchNorm) or isinstance(m, nn.BatchNorm2d):
                m.running_mean.data = self.backup[name + '.running_mean']
                m.running_var.data = self.backup[name + '.running_var']
                m.num_batches_tracked.data = self.backup[name + '.num_batches_tracked']
        self.backup = {}
        
class EMA:
    """
    Implementation from https://fyubang.com/2019/06/01/ema/
    """

    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

    def load(self, ema_model):
        for name, param in ema_model.named_parameters():
            self.shadow[name] = param.data.clone()

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}

class Get_Scalar:
    def __init__(self, value):
        self.value = value

    def get_value(self, iter):
        return self.value

    def __call__(self, iter):
        return self.value
import torch.nn.functional as F
def ce_loss(logits, targets, use_hard_labels=True, reduction='none'):
    """
    wrapper for cross entropy loss in pytorch.
    
    Args
        logits: logit values, shape=[Batch size, # of classes]
        targets: integer or vector, shape=[Batch size] or [Batch size, # of classes]
        use_hard_labels: If True, targets have [Batch size] shape with int values. If False, the target is vector (default True)
    """
    if use_hard_labels:
        log_pred = F.log_softmax(logits, dim=-1)
        return F.nll_loss(log_pred, targets, reduction=reduction)
        # return F.cross_entropy(logits, targets, reduction=reduction) this is unstable
    else:
        assert logits.shape == targets.shape
        log_pred = F.log_softmax(logits, dim=-1)
        nll_loss = torch.sum(-targets * log_pred, dim=1)
        return nll_loss

from torch.optim.lr_scheduler import LambdaLR

def get_cosine_schedule_with_warmup(optimizer,
                                    num_training_steps,
                                    num_cycles=7. / 16.,
                                    num_warmup_steps=0,
                                    last_epoch=-1):
    '''
    Get cosine scheduler (LambdaLR).
    if warmup is needed, set num_warmup_steps (int) > 0.
    '''



    def _lr_lambda(current_step):
        '''
        _lr_lambda returns a multiplicative factor given an interger parameter epochs.
        Decaying criteria: last_epoch
        '''

        if current_step < num_warmup_steps:
            _lr = float(current_step) / float(max(1, num_warmup_steps))
        else:
            num_cos_steps = float(current_step - num_warmup_steps)
            num_cos_steps = num_cos_steps / float(max(1, num_training_steps - num_warmup_steps))
            _lr = max(0.0, math.cos(math.pi * num_cycles * num_cos_steps))
        return _lr
    # print(":LLLL")
    return torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda, last_epoch)
