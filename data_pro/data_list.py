import torch
import numpy as np
import random
from PIL import Image
from torch.utils.data import Dataset,Sampler
from collections import defaultdict
import os
import os.path
import cv2
import torchvision

def make_dataset(image_list, labels):
    if labels:
      len_ = len(image_list)
      images = [(image_list[i].strip(), labels[i, :]) for i in range(len_)]
    else:
      if len(image_list[0].split()) > 2:
        images = [(val.split()[0], np.array([int(la) for la in val.split()[1:]])) for val in image_list]
      else:
        images = [(val.split()[0], int(val.split()[1])) for val in image_list]
    return images


def rgb_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

def l_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('L')

class ImageList(Dataset):
    def __init__(self, image_list, root=None,labels=None, transform=None, target_transform=None, mode='RGB'):
        imgs = make_dataset(image_list, labels)
        # if len(imgs) == 0:
        #     raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
        #                        "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.imgs = imgs
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        if mode == 'RGB':
            self.loader = rgb_loader
        elif mode == 'L':
            self.loader = l_loader

        self.classwise_indices = defaultdict(list)
        self.idx2class = []
        for [_, target] in self.imgs:
            # print(target)
            self.idx2class.append(target)
        # self.idx2class=[target for i,[_,target] in enumerate(self.imgs)]
        for idx, classes in enumerate(self.idx2class):
            self.classwise_indices[classes].append(idx)

    def get_class(self, idx):
        return self.idx2class[idx]

    def __getitem__(self, index):
        # print("index",index)
        path, target = self.imgs[index]
        path=os.path.join(self.root,path)

        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def __len__(self):
        return len(self.imgs)

class ImageList_idx(Dataset):
    def __init__(self, image_list, root=None,labels=None, transform=None, target_transform=None, mode='RGB'):
        imgs = make_dataset(image_list, labels)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.imgs = imgs
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        if mode == 'RGB':
            self.loader = rgb_loader
        elif mode == 'L':
            self.loader = l_loader

        self.classwise_indices = defaultdict(list)
        self.idx2class = []
        for [_, target] in self.imgs:
            # print(target)
            self.idx2class.append(target)
        # self.idx2class=[target for i,[_,target] in enumerate(self.imgs)]
        for idx, classes in enumerate(self.idx2class):
            self.classwise_indices[classes].append(idx)

    # def get_psd_class(self,idx):
    #     return self.idx2class[idx]

    def set_psd_class(self,pred):
        self.idx2class=list(pred.reshape(-1)) # numpy to list    the pred is not shuffleed
        self.classwise_indices = defaultdict(list)
        for idx, classes in enumerate(self.idx2class):
            self.classwise_indices[classes].append(idx)

    def get_class(self, idx):
        return self.idx2class[idx]


    def __getitem__(self, index):
        path, target = self.imgs[index]

        path=os.path.join(self.root,path)

        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index

    def __len__(self):
        return len(self.imgs)

class PairBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, num_iterations=None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_iterations = num_iterations

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        random.shuffle(indices)
        for k in range(len(self)):
            if self.num_iterations is None:
                offset = k*self.batch_size
                batch_indices = indices[offset:offset+self.batch_size]
            else:
                batch_indices = random.sample(range(len(self.dataset)),
                                              self.batch_size)

            pair_indices = []
            for idx in batch_indices:
                y = self.dataset.get_class(idx)
                pair_indices.append(random.choice(self.dataset.classwise_indices[y]))
            # print(len(batch_indices + pair_indices))
            # print("1", batch_indices, "s", pair_indices)
            yield batch_indices + pair_indices

    def __len__(self):
        if self.num_iterations is None:
            return (len(self.dataset)) // self.batch_size
        else:
            return self.num_iterations