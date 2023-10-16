"""Data loading facilities for Omniglot experiment."""
import random
import os
from os.path import join

import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import Sampler
from torch.utils.data import Dataset
from torchvision.datasets.utils import list_dir, list_files
from torchvision import transforms
from PIL import Image


DEFAULT_TRANSFORM =transforms.Compose([transforms.Resize((100,100)),
                                     transforms.ToTensor(),
                                     transforms.Normalize(
                                          mean=(0.485, 0.456, 0.406),
                                          std=(0.229, 0.224, 0.225)
                                     )])

CAT2LABEL={'bike': 0, 'monitor': 1, 'laptop_computer': 2, 'mug': 3, 'calculator': 4, 'projector': 5, 'keyboard': 6, 'headphones': 7, 'back_pack': 8, 'mouse': 9}
###############################################################################
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

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
    def __init__(self, image_list, labels=None, transform=None,target_transform=None, mode='RGB'):
        imgs = make_dataset(image_list, labels)
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        if mode == 'RGB':
            self.loader = rgb_loader
        elif mode == 'L':
            self.loader = l_loader
        self.return_index=0
    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        if self.return_index==1:

            return img, target,index
        else:
            return img, target

    def __len__(self):
        return len(self.imgs)



