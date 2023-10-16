import os
import torch
from torchvision import transforms
from data_pro.SSDA_data_list import Imagelists_VISDA, return_classlist
from data_pro.data_list import ImageList_idx,ImageList,PairBatchSampler

class ResizeImage():
    def __init__(self, size):
        if isinstance(size, int):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img):
        th, tw = self.size
        return img.resize((th, tw))


def return_dataset(args):
    base_path = '/data/maning/git/shot/data/SSDA_split/%s' % args.dataset

    if args.dataset in "office-home":
        # args.dataset='OfficeHomeDataset'
        root = '/data/maning/git/shot/data/OfficeHomeDataset/'
    else :

        root = '/data/maning/git/shot/data/%s/' % args.dataset
    image_set_file_s = \
        os.path.join(base_path,
                     'labeled_source_images_' +
                     args.s + '.txt')
    image_set_file_t = \
        os.path.join(base_path,
                     'labeled_target_images_' +
                     args.t + '_%d.txt' % (args.num))
    image_set_file_t_val = \
        os.path.join(base_path,
                     'validation_target_images_' +
                     args.t + '_3.txt')
    image_set_file_unl = \
        os.path.join(base_path,
                     'unlabeled_target_images_' +
                     args.t + '_%d.txt' % (args.num))

    if args.net == 'alexnet':
        crop_size = 227
    else:
        crop_size = 224
    data_transforms = {
        'train': transforms.Compose([
            ResizeImage(256),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            ResizeImage(256),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            ResizeImage(256),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    def split(train_r, source_path):
        with open(source_path, 'r') as f:
            data = f.readlines()
            train_len = int(len(data) * train_r)
            train, val = torch.utils.data.random_split(data, [train_len, len(data) - train_len])
        return train, val

    if args.dataset in "multi":
        source_train, source_val = split(train_r=0.95, source_path=image_set_file_s)
    else:

        source_train, source_val = split(train_r=0.90, source_path=image_set_file_s)

    print("source_train and val num", len(source_train), len(source_val))

    source_dataset = ImageList(source_train, root=root,
                                      transform=data_transforms['train'])
    source_val_dataset = ImageList(source_val, root=root,
                                          transform=data_transforms['val'])
    target_dataset = ImageList(open(image_set_file_t).readlines(), root=root,
                                      transform=data_transforms['val'])
    target_dataset_val = ImageList(open(image_set_file_t_val).readlines(), root=root,
                                          transform=data_transforms['val'])
    target_dataset_unl = ImageList_idx(open(image_set_file_unl).readlines(), root=root,
                                          transform=data_transforms['val'])

    target_dataset_test = ImageList(open(image_set_file_unl).readlines(), root=root,
                                           transform=data_transforms['test'])
    class_list = return_classlist(image_set_file_s)
    print("%d classes in this dataset" % len(class_list))
    # if args.net == 'alexnet':
    #     bs = 20
    # else:
    #     bs = 16
    bs=args.batch_size*2 # for KD term

    if args.skd_src==1:
        source_loader = torch.utils.data.DataLoader(source_dataset,batch_sampler=PairBatchSampler(source_dataset, args.batch_size),num_workers=args.worker)
    else:
        source_loader = torch.utils.data.DataLoader(source_dataset, batch_size=bs,
                                                num_workers=args.worker, shuffle=True,
                                                drop_last=False)

    source_val_loader = torch.utils.data.DataLoader(source_val_dataset, batch_size=bs,
                                                    num_workers=args.worker, shuffle=False,
                                                    drop_last=False)
    target_loader = \
        torch.utils.data.DataLoader(target_dataset,
                                    batch_size=min(bs, len(target_dataset)),
                                    num_workers=args.worker,
                                    shuffle=True, drop_last=True)
    target_loader_val = \
        torch.utils.data.DataLoader(target_dataset_val,
                                    batch_size=min(bs,
                                                   len(target_dataset_val)),
                                    num_workers=args.worker,
                                    shuffle=True, drop_last=False)
    if args.skd_src==1:
        target_loader_unl = torch.utils.data.DataLoader(target_dataset_unl,
                                                        batch_sampler=PairBatchSampler(target_dataset_unl, args.batch_size),num_workers=args.worker)
    else:
        target_loader_unl = torch.utils.data.DataLoader(target_dataset_unl,
                                    batch_size=bs * 1, num_workers=args.worker,
                                    shuffle=True, drop_last=True)

    target_loader_test = \
        torch.utils.data.DataLoader(target_dataset_test,
                                    batch_size=bs * 1, num_workers=args.worker,
                                    shuffle=False, drop_last=False)
    return source_loader, source_val_loader, target_loader, target_loader_unl, \
           target_loader_val, target_loader_test, target_dataset_unl


def return_dataset_test(args):
    base_path = './data/txt/%s' % args.dataset
    root = './data/%s/' % args.dataset
    image_set_file_s = os.path.join(base_path, args.source + '_all' + '.txt')
    image_set_file_test = os.path.join(base_path,
                                       'unlabeled_target_images_' +
                                       args.target + '_%d.txt' % (args.num))
    if args.net == 'alexnet':
        crop_size = 227
    else:
        crop_size = 224
    data_transforms = {
        'test': transforms.Compose([
            ResizeImage(256),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    target_dataset_unl = Imagelists_VISDA(image_set_file_test, root=root,
                                          transform=data_transforms['test'],
                                          test=True)
    class_list = return_classlist(image_set_file_s)
    print("%d classes in this dataset" % len(class_list))
    if args.net == 'alexnet':
        bs = 32
    else:
        bs = 24
    target_loader_unl = \
        torch.utils.data.DataLoader(target_dataset_unl,
                                    batch_size=bs * 2, num_workers=3,
                                    shuffle=False, drop_last=False)
    return target_loader_unl, class_list
