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

HOLD_OUT = 0

DEFAULT_TRANSFORM =transforms.Compose([transforms.Resize((100,100)),
                                     transforms.ToTensor(),
                                     transforms.Normalize(
                                          mean=(0.485, 0.456, 0.406),
                                          std=(0.229, 0.224, 0.225)
                                     )])

CAT2LABEL={'bike': 0, 'monitor': 1, 'laptop_computer': 2, 'mug': 3, 'calculator': 4, 'projector': 5, 'keyboard': 6, 'headphones': 7, 'back_pack': 8, 'mouse': 9}
###############################################################################

class DataContainer(object):

    """Data container class for Omniglot

    Arguments:
        root (str): root of dataset.
        num_pretrain_alphabets (int): number of alphabets to use for
            meta-training.
        num_classes (int): number of classes to enforce per task (optional).
        transform (func): transformation to apply to each input sample.
        seed (int): seed used to shuffle alphabets when creating train/val/test
            splits.
        **kwargs (dict): keyword arguments to pass to the
            torch.utils.data.DataLoader
    """

    folder = "./office_caltech_10"
    target_folder = "./office_caltech_10"

    def __init__(self, root="./data", pretrain_domains=[],target_domain=[],
                 num_classes=10, transform=DEFAULT_TRANSFORM,
                 seed=1, **kwargs):
        self.root = root
        self.pretrain_domains = pretrain_domains
        self.target_domain=target_domain
        self.transform = transform
        self.seed = seed
        self.kwargs = kwargs
        path = join(os.path.expanduser(self.root),  self.target_folder)
        domains = list_dir(path)
        if num_classes:
            domains = [a for a in domains
                         if len(list_dir(join(path, a))) >= num_classes]
            # assert self.num_pretrain_domains + TEST < len(domains), \
                # 'cannot create test set'

        random.seed(self.seed)
        train = self.pretrain_domains
        test = self.target_domain
        val = self.target_domain

        trainset = [office_caltech(root, [t], num_classes, HOLD_OUT,
                                transform=transform) for t in train]
        testset = [office_caltech(root, [v], num_classes, HOLD_OUT,
                               transform=transform) for v in test]
        valset = [office_caltech(root, [v], num_classes, HOLD_OUT,
                              transform=transform) for v in val]

        self.domains = domains
        self.domains_train = train
        self.domains_test = test
        self.domains_val = val

        self.data_train = trainset
        self.data_test = testset
        self.data_val = valset

    def get_loader(self, task, batch_size, iterations):
        """Returns a DataLoader for given configuration.

        Arguments:
            task (SubOmniglot): A SubOmniglot instance to pass to a
                DataLoader instance.
            batch_size (int): batch size in data loader.
            iterations (int): number of batches.
        """
        return DataLoader(task,
                          batch_size,
                          sampler=RandomSampler(task, iterations, batch_size),
                          **self.kwargs)
    def get_test_loader(self, task, batch_size):
        """Returns a DataLoader for given configuration.

        Arguments:
            task (SubOmniglot): A SubOmniglot instance to pass to a
                DataLoader instance.
            batch_size (int): batch size in data loader.
            iterations (int): number of batches.
        """
        return DataLoader(task,
                          batch_size,
                          # sampler=RandomSampler(task, iterations, batch_size),
                          **self.kwargs)

    def train(self, meta_batch_size, batch_size, iterations, return_idx=False):
        """Generator meta-train batch

        Arguments:
            meta_batch_size (int): number of tasks in batch.
            batch_size (int): number of samples in each batch in the inner
                (task) loop.
            iterations (int): number of training steps on each task.
            return_idx (int): return task ids (default=False).
        """
        n_tasks = len(self.data_train)

        if n_tasks == 1:
            tasks = zip([0] * meta_batch_size,
                        self.data_train * meta_batch_size)
        else:
            tasks = []
            task_ids = list(range(n_tasks))
            while True:
                random.shuffle(task_ids)
                tasks.extend([(i, self.data_train[i]) for i in task_ids])
                if len(tasks) >= meta_batch_size:
                    break
            tasks = tasks[:meta_batch_size]

        task_ids, task_data = zip(*tasks)
        task_data = [self.get_loader(t, batch_size, iterations)
                     for t in task_data]

        if return_idx:
            return list(zip(task_ids, task_data))
        return task_data

    def val(self, batch_size, iterations, return_idx=False):
        """Generator meta-validation batch

        Arguments:
            batch_size (int): number of samples in each batch in the inner
            (task) loop.
            iterations (int): number of training steps on each task.
            return_idx (int): return task ids (default=False).
        """
        n = len(self.data_train)

        tsk = [i+n for i in range(len(self.data_val))]
        tasks = [self.get_loader(d, batch_size, iterations)
                 for d in self.data_val]

        if return_idx:
            return list(zip(tsk, tasks))
        return tasks

    def test(self, batch_size):
        """Generator meta-test batch

        Arguments:
            batch_size (int): number of samples in each batch in the inner
                (task) loop.
            iterations (int): number of training steps on each task.
            return_idx (int): return task ids (default=False).
        """
        # n = len(self.data_train) + len(self.data_val)

        # tsk = [i+n for i in range(len(self.data_test))]
        # tasks = [self.get_loader(d, batch_size, 5)
        #          for d in self.data_test]
        self.data_test[0].train()
        tasks=DataLoader(self.data_test[0], batch_size=batch_size, shuffle=True,
                   num_workers=4, drop_last=True)
        # if return_idx:
        #     return list(zip(tsk, tasks))
        return tasks

    def test_warp(self, batch_size):
        """Generator meta-test batch

        Arguments:
            batch_size (int): number of samples in each batch in the inner
                (task) loop.
            iterations (int): number of training steps on each task.
            return_idx (int): return task ids (default=False).
        """
        # n = len(self.data_train) + len(self.data_val)

        # tsk = [i+n for i in range(len(self.data_test))]
        tasks = [self.get_loader(d, batch_size, 5)
                 for d in self.data_test]
        # self.data_test[0].train()
        # tasks=DataLoader(self.data_test[0], batch_size=batch_size, shuffle=True,
        #            num_workers=4, drop_last=True)
        # if return_idx:
        #     return list(zip(tsk, tasks))
        return tasks


class office_caltech(Dataset):

    """Data class for Omniglotamples a that subs specified number of alphabets.

    Arguments:
        root (str): root of the Omniglot dataset.
        alphabets (int): number of alphabets to use in the creation of the
            dataset.
        num_classes (int): number of classes to enforce per task (optional).
        hold_out (int): number of samples per character to hold for validation
            set (optional).
        transform (func): transformation to apply to each input sample.
        seed (int): seed used to shuffle alphabets when creating train/val/test
            splits.
    """

    folder = "office_caltech_10"
    target_folder = "office_caltech_10"

    def __init__(self, root, domains, num_classes=None, hold_out=None,
                 transform=None, seed=None):
        self.root = root
        self.domains = domains
        self.num_classes = num_classes
        self.hold_out = hold_out #rate for validation
        self.transform = transform
        self.target_transform = None
        self.seed = seed

        self.target_folder = join(self.root, self.target_folder)
        self._domains = [a for a in list_dir(self.target_folder)
                           if a in self.domains]
        self._categories= sum(
            [[join(a, c) for c in list_dir(join(self.target_folder, a))]
             for a in self._domains], [])

        # print(dict([(s.split("/")[1],index) for index,s in enumerate(self._categories)]))
        if seed:
            random.seed(seed)

        random.shuffle(self._categories)

        if self.num_classes:
            self._categories = self._categories[:num_classes]

        # self._categories_images = [
        #     [(image, idx) for image in
        #      list_files(join(self.target_folder, category), '.jpg')]
        #     for idx, category in enumerate(self._categories)
        # ]

        self._train_category_images = []
        self._val_category_images = []
        
        for idx, category in enumerate(self._categories):
            train_characters = []
            val_characters = []
            path_category=join(self.target_folder, category)
            for img_count, image in enumerate(
                    list_files(path_category, '.jpg')):
                image_path=join(path_category,image)
                # print(image_path)
                if hold_out and img_count < hold_out:
                    val_characters.append((image_path, CAT2LABEL[category.split("/")[1]]))
                else:
                    train_characters.append((image_path, CAT2LABEL[category.split("/")[1]]))
            self._train_category_images.append(train_characters)
            self._val_category_images.append(val_characters)
        #test
        # print(self._train_category_images)
        # for category in self._train_category_images:
        #     for (image,idx) in category:

        #         if "amazon" in image:
        #             print("image",image,idx)

        self._flat_train_character_images = sum(
            self._train_category_images, [])
        self._flat_val_character_images = sum(
            self._val_category_images, [])

        self._train = True
        self._set_images()
        # self.image_count=set()
        # self.count=0
        # print("creat")
    def __getitem__(self, index):
        
        path_img, label = self._flat_character_images[index]
        img = Image.open(path_img).convert('RGB')     # 0~255
        # print("img:",path_img,"label",label)
        # self.image_count.add(path_img)
        # self.count+=1
        # if index>800:
            # print(index)
        # print(self.count)
        # if self.count%30==0:
        #     print("domainS",self.domains,"len",len(self.image_count))
        if self.transform is not None:
            img = self.transform(img)   # 在这里做transform，转为tensor等等

        return img, label

    def __len__(self):
        return len(self._flat_character_images)

    def train(self):
        """Train mode"""
        self._train = True
        self._set_images()

    def eval(self):
        """Eval mode"""
        self._train = False
        self._set_images()

    def _set_images(self):
        """Set images"""
        if self._train:
            self._flat_character_images = self._flat_train_character_images
            
        else:
            self._flat_character_images = self._flat_val_character_images


class RandomSampler(Sampler):
    r"""Samples elements randomly with replacement (if iterations > data set).

    Arguments:
        data_source (Dataset): dataset to sample from
        iterations (int): number of samples to return on each call to __iter__
        batch_size (int): number of samples in each batch
    """

    def __init__(self, data_source, iterations, batch_size):
        self.data_source = data_source
        self.iterations = iterations
        self.batch_size = batch_size

    def __iter__(self):
        if self.data_source._train:
            # print("data",self.iterations * self.batch_size , len(self.data_source))
            assert self.iterations * self.batch_size < len(self.data_source)
            idx= torch.randperm(len(self.data_source))[0 : (self.iterations * self.batch_size) % len(self.data_source)]
            # idx = torch.randperm(self.iterations * self.batch_size) % len(
            #     self.data_source)
        else:
            idx = torch.randperm(len(self.data_source))
        # print("len",len(idx))
        return iter(idx.tolist())

    def __len__(self):  # pylint: disable=protected-access
        return self.iterations * self.batch_size if self.data_source._train \
            else len(self.data_source)
        # return len(self.data_source)
