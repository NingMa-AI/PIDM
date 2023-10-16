from torchvision import transforms
import torchvision

from torch.utils.data import DataLoader
from usps import *
from data_pro.office_home import ImageList
import random
class PlaceCrop(object):
    """Crops the given PIL.Image at the particular index.
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (w, h), a square crop (size, size) is
            made.
    """

    def __init__(self, size, start_x, start_y):
        if isinstance(size, int):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.start_x = start_x
        self.start_y = start_y

    def __call__(self, img):
        """
        Args:
            img (PIL.Image): Image to be cropped.
        Returns:
            PIL.Image: Cropped image.
        """
        th, tw = self.size
        return img.crop((self.start_x, self.start_y, self.start_x + tw, self.start_y + th))

class ResizeImage():
    def __init__(self, size):
      if isinstance(size, int):
        self.size = (int(size), int(size))
      else:
        self.size = size
    def __call__(self, img):
      th, tw = self.size
      return img.resize((th, tw))

def image_train(resize_size=256, crop_size=224):

  return  transforms.Compose([
      transforms.Resize((resize_size, resize_size)),
      transforms.RandomCrop(crop_size),
      transforms.RandomHorizontalFlip(),
      transforms.ToTensor(),
      torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

def image_test(resize_size=256, crop_size=224):
  # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
  #                                  std=[0.229, 0.224, 0.225])
  # start_center = (resize_size - crop_size - 1) / 2
  return transforms.Compose([
      transforms.Resize((resize_size, resize_size)),
      transforms.CenterCrop(crop_size),
      # transforms.RandomHorizontalFlip(),
      transforms.ToTensor(),
      torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
  ])

def get_data_loader(args):
    dataset=args.dataset
    source_path =None
    target_path=None
    train_bs=args.batch_size
    dset_loaders = {}

    def split(train_r,source_path):
        with open(source_path,'r') as f:
            data=f.readlines()
            train_len=int(len(data)*train_r)
            train,val=torch.utils.data.random_split(data, [train_len, len(data) - train_len])
        return train,val

    if dataset == "digits":
        return get_data_loader(args)
    elif dataset == "office_home":
        source_path=os.path.join("./data/office-home/",args.s+".txt")
        target_path=os.path.join("./data/office-home/",args.t+".txt")
    elif dataset == "office-31":
        source_path=os.path.join("./data/office/",args.s+".txt")
        target_path=os.path.join("./data/office/",args.t+".txt")

    train_data,val_dada=split(train_r=0.9,source_path=source_path)
    # print("train-total:",len(train_data)+len(val_dada),"train_train",len(train_data),"train_val",len(val_dada))
    train_source = ImageList(train_data, transform=image_train())
    test_source = ImageList(val_dada, transform=image_test())
    train_target = ImageList(open(target_path).readlines(),transform=image_train())
    train_target.return_index = 1
    test_target = ImageList(open(target_path).readlines(), transform=image_test())
    test_target.return_index=1

    dset_loaders["source_tr"] = DataLoader(train_source, batch_size=train_bs, shuffle=True,
                                           num_workers=args.worker, drop_last=False)
    dset_loaders["source_te"] = DataLoader(test_source, batch_size=train_bs * 2, shuffle=True,
                                           num_workers=args.worker, drop_last=False)
    dset_loaders["target"] = DataLoader(train_target, batch_size=train_bs, shuffle=True,
                                        num_workers=args.worker, drop_last=False)
    dset_loaders["test"] = DataLoader(test_target, batch_size=train_bs * 2, shuffle=False,
                                  num_workers=args.worker, drop_last=False)
    return dset_loaders

def digit_load(args):
    train_bs = args.batch_size
    if args.s+"2"+args.t == 's2m':
        train_source = torchvision.datasets.SVHN('../data/digit/svhn/', split='train', download=True,
                                                 transform=transforms.Compose([
                                                     transforms.Resize(32),
                                                     transforms.ToTensor(),
                                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                                 ]))
        test_source = torchvision.datasets.SVHN('../data/digit/svhn/', split='test', download=True,
                                                transform=transforms.Compose([
                                                    transforms.Resize(32),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                                ]))
        train_target = torchvision.datasets.MNIST('../data/digit/mnist/', train=True, download=True,
                                                  transform=transforms.Compose([
                                                      transforms.Resize(32),
                                                      transforms.Lambda(lambda x: x.convert("RGB")),
                                                      transforms.ToTensor(),
                                                      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                                  ]))
        test_target = torchvision.datasets.MNIST('../data/digit/mnist/', train=False, download=True,
                                                 transform=transforms.Compose([
                                                     transforms.Resize(32),
                                                     transforms.Lambda(lambda x: x.convert("RGB")),
                                                     transforms.ToTensor(),
                                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                                 ]))
    elif args.s+"2"+args.t == 'u2m':
        train_source = USPS('../data/digit/usps/', train=True, download=True,
                            transform=transforms.Compose([
                                transforms.RandomCrop(28, padding=4),
                                transforms.RandomRotation(10),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))
                            ]))
        test_source = USPS('../data/digit/usps/', train=False, download=True,
                           transform=transforms.Compose([
                               transforms.RandomCrop(28, padding=4),
                               transforms.RandomRotation(10),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5,), (0.5,))
                           ]))
        train_target = torchvision.datasets.MNIST('../data/digit/mnist/', train=True, download=True,
                                                  transform=transforms.Compose([
                                                      transforms.ToTensor(),
                                                      transforms.Normalize((0.5,), (0.5,))
                                                  ]))
        test_target = torchvision.datasets.MNIST('../data/digit/mnist/', train=False, download=True,
                                                 transform=transforms.Compose([
                                                     transforms.ToTensor(),
                                                     transforms.Normalize((0.5,), (0.5,))
                                                 ]))
        # train_target=train_source
        # test_target=test_source
    elif args.s+"2"+args.t == 'm2u':
        train_source = torchvision.datasets.MNIST('../data/digit/mnist/', train=True, download=True,
                                                  transform=transforms.Compose([
                                                      transforms.ToTensor(),
                                                      transforms.Normalize((0.5,), (0.5,))
                                                  ]))
        test_source = torchvision.datasets.MNIST('../data/digit/mnist/', train=False, download=True,
                                                 transform=transforms.Compose([
                                                     transforms.ToTensor(),
                                                     transforms.Normalize((0.5,), (0.5,))
                                                 ]))

        train_target = USPS('../data/digit/usps/', train=True, download=True,
                            transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))
                            ]))
        test_target = USPS('../data/digit/usps/', train=False, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.5,), (0.5,))
                           ]))
    if args.s+"2"+args.t == 'm2s':
        train_target = torchvision.datasets.SVHN('../data/digit/svhn/', split='train', download=True,
                                                 transform=transforms.Compose([
                                                     transforms.Resize(32),
                                                     transforms.ToTensor(),
                                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                                 ]))
        test_target = torchvision.datasets.SVHN('../data/digit/svhn/', split='test', download=True,
                                                transform=transforms.Compose([
                                                    transforms.Resize(32),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                                ]))
        train_source= torchvision.datasets.MNIST('../data/digit/mnist/', train=True, download=True,
                                                  transform=transforms.Compose([
                                                      transforms.Resize(32),
                                                      transforms.Lambda(lambda x: x.convert("RGB")),
                                                      transforms.ToTensor(),
                                                      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                                  ]))
        test_source = torchvision.datasets.MNIST('../data/digit/mnist/', train=False, download=True,
                                                 transform=transforms.Compose([
                                                     transforms.Resize(32),
                                                     transforms.Lambda(lambda x: x.convert("RGB")),
                                                     transforms.ToTensor(),
                                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                                 ]))
    dset_loaders = {}
    dset_loaders["source_tr"] = DataLoader(train_source, batch_size=train_bs, shuffle=True,
                                           num_workers=args.worker, drop_last=False)
    dset_loaders["source_te"] = DataLoader(test_source, batch_size=train_bs * 2, shuffle=True,
                                           num_workers=args.worker, drop_last=False)
    dset_loaders["target"] = DataLoader(train_target, batch_size=train_bs, shuffle=True,
                                        num_workers=args.worker, drop_last=False)
    dset_loaders["test"] = DataLoader(test_target, batch_size=train_bs * 2, shuffle=False,
                                      num_workers=args.worker, drop_last=False)
    return dset_loaders

# class ImageList_idx(Dataset):
#     def __init__(self, image_list, labels=None, transform=None, target_transform=None, mode='RGB'):
#         imgs = make_dataset(image_list, labels)
#         if len(imgs) == 0:
#             raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
#                                "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))
#
#         self.imgs = imgs
#         self.transform = transform
#         self.target_transform = target_transform
#         if mode == 'RGB':
#             self.loader = rgb_loader
#         elif mode == 'L':
#             self.loader = l_loader
#
#     def __getitem__(self, index):
#         path, target = self.imgs[index]
#         img = self.loader(path)
#         if self.transform is not None:
#             img = self.transform(img)
#         if self.target_transform is not None:
#             target = self.target_transform(target)
#
#         return img, target, index
#
#     def __len__(self):
#         return len(self.imgs)