import os
import torch
from torchvision import transforms
from data_pro.SSDA_data_list import Imagelists_VISDA, return_classlist
from data_pro.data_list import ImageList_idx,ImageList
import collections
import torch
import sys
project_root="."
class ResizeImage():
    def __init__(self, size):
        if isinstance(size, int):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img):
        th, tw = self.size
        return img.resize((th, tw))

class TransformTwice:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, inp):
        out1 = self.transform(inp)
        out2 = self.transform(inp)
        return out1, out2

def apply_train_dropout(m):
    if type(m) == torch.nn.Dropout:
        # print("find droput")
        m.train()
def return_dataloader_by_UPS(args,netF,netC,unlabeled_data_loader):
    netF.eval()
    netC.eval()
    base_path = project_root+"/data/SSDA_split/%s" % args.dataset

    if args.dataset in "office-home":
        # args.dataset='OfficeHomeDataset'
        root = project_root+"/data/OfficeHomeDataset/"
    else:
        root = project_root+"/data/%s/" % args.dataset

    image_set_file_t = \
        os.path.join(base_path,
                     'labeled_target_images_' +
                     args.t + '_%d.txt' % (args.num))
    image_set_file_t_val = \
        os.path.join(base_path,
                     'validation_target_images_' +
                     args.t + '_3.txt')
    # if args.uda==1:
    #     image_set_file_unl = \
    #         os.path.join(base_path,
    #                      'labeled_source_images_' +
    #                      args.t+".txt")
    # else:
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
    netF.apply(apply_train_dropout)# MC dropout
    netC.apply(apply_train_dropout)

    bs = args.batch_size
    unlabel_target_list = open(image_set_file_unl).readlines()
    target_list = open(image_set_file_t).readlines()

    target_dataset_unl = Imagelists_VISDA(unlabel_target_list, root=root,
                                          transform=TransformTwice(data_transforms['val']))
    target_dataset_unl.return_index=True
    target_loader_unl = torch.utils.data.DataLoader(target_dataset_unl,
                                                    batch_size=bs * 1, num_workers=args.worker,
                                                    shuffle=False, drop_last=False)
    start_test = True
    with torch.no_grad():
        all_index=[]
        for step, ((inputs1,inputs2), labels,index) in enumerate(target_loader_unl):
            # print(step,index)
             # = data[0]
            # labels = data[1]
            inputs1,inputs2 = inputs1.cuda(),inputs2.cuda()
            # labels = labels.cuda()  # 2020 07 06
            # inputs = inputs
            output=[]
            batch_s=inputs2.shape[0]
            repeat=5
            for i in range(repeat):
                outputs1 = netC(netF(inputs1)).cpu()
                outputs2 = netC(netF(inputs2)).cpu()
                output.append(outputs1)
                output.append(outputs2)
            output=torch.cat(output,dim=0).view(2*repeat,batch_s,-1)
            # print("std",torch.std(output,dim=0))

            outputs=torch.mean(output,dim=0)
            # print("outputs",outputs.shape)
            # print("mean",outputs)
            # outputs,margin_logits = netC(netF(inputs),labels)
            # labels = labels.cpu()  # 2020 07 06
            # step=torch.tensor(step).int().reshape(-1)
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                all_index= index
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
                all_index= torch.cat((all_index, index), 0)
            # print(type(index))
            # all_index.append(index)

    # print("len",len(all_index),len(all_output))
    _, pred_label = torch.max(all_output, 1)

    # _, predict = torch.max(all_output, 1)
    # res = torch.squeeze(predict).float() == all_label
    # accuracy = torch.sum(res).item() / float(all_label.size()[0])

    entropy = Entropy(torch.nn.Softmax(dim=1)(all_output))
    # mean_ent = torch.mean(entropy).cpu().data.item()

    # entropy for each class
    class_to_mean_entropy = list(0. for i in range(args.class_num))
    # correct = list(0. for i in range(args.class_num))
    total_add = list(0. for i in range(args.class_num))
    class2index=collections.defaultdict(list)
    classforentropy=collections.defaultdict(list)
    index2all_index=collections.defaultdict(dict)

    max_num_per_class=args.max_num_per_class
    # print(all_index.shape,pred_label.shape)
    for label in range(args.class_num):
        class_to_mean_entropy[label] = torch.mean(entropy[pred_label == label]).item()
        # classforentropy[label]=list(entropy[pred_label == label].numpy())

        index2all_index[label]=dict(list(zip(range(len(all_index[pred_label == label])),all_index[pred_label == label].numpy())))
        # print("index2all_label",index2all_index[label])
        classentropy=entropy[pred_label == label]
        if len(classentropy)<max_num_per_class:
            indexes=range(len(classentropy))
        else:
            preds, indexes = torch.topk(classentropy, max_num_per_class, largest=False)
            indexes=indexes.numpy()
        # print("entroy_len",len(entropy[pred_label == label]),"index_len",len(all_index[pred_label == label]))
        # print("indexes",indexes)
        for ind in indexes:
            class2index[label].append(index2all_index[label][ind]) # local index to global index

    # for index in all_index:
    #      label=pred_label[index].item()
    #      if entropy[index] < class_to_mean_entropy[label]*0.5:
    #          class2index[label].append(index)
    #
    # # print(class2index.items())
    #
    #
    # # assert
    # # for index , label in enumerate(all_label):
    # #     print(label,unlabel_target_list[index])
    #
    total=0
    acc=0
    line2remove=[]

    if args.uda == 1:
        target_list=[]

    for psudo_label, indexes in class2index.items():
        for index in indexes:
            line= unlabel_target_list[index]
            psudo_line=line.split(" ")[0] + " " + str(psudo_label)
            target_list.append(psudo_line)
            line2remove.append(line)
            total_add[int(line.split(" ")[1])] += 1
            # if total_add[int(line.split(" ")[1])] >= max_num_per_class:
            #     break
            # print(line.split(" ")[1],str(psudo_label))
            if int(line.split(" ")[1])==psudo_label:
                acc+=1
            total+=1
    # print("acc of psudo label",acc*1.0/total)
    # print("added number for each class ",sum(total_add),total_add)
    before_remove=len(unlabel_target_list)
    
    for line in line2remove :
        unlabel_target_list.remove(line)
        # print("remove",line)
    print("removed data from unlalbed dataset",before_remove," to ",len(unlabel_target_list))
    # print(target_list)


    target_dataset = Imagelists_VISDA(target_list, root=root,
                                      transform=data_transforms['val'])
    # print("target len", len(unlabel_target_list))
    target_dataset_unl = Imagelists_VISDA(unlabel_target_list, root=root,
                                          transform=data_transforms['val'])

    target_dataset_unl.return_index=True

    target_loader = \
        torch.utils.data.DataLoader(target_dataset,
                                    batch_size=min(bs, len(target_dataset)),
                                    # batch_size=bs,
                                    # pin_memory=True,
                                    num_workers=args.worker,
                                    shuffle=True, drop_last=True)
    target_loader_unl = \
        torch.utils.data.DataLoader(target_dataset_unl,
                                    # pin_memory=True,
                                    batch_size=bs * 1, num_workers=args.worker,
                                    shuffle=True, drop_last=True)
    netF.train()
    netC.train()
    return target_loader, target_loader_unl,acc*1.0/total

def return_dataset(args):
    base_path = project_root + "/data/SSDA_split/%s" % args.dataset

    if args.dataset in "office-home":
        # args.dataset='OfficeHomeDataset'
        root = project_root + "/data/OfficeHomeDataset/"
    else:
        root = project_root + "/data/%s/" % args.dataset

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

    source_dataset = Imagelists_VISDA(source_train, root=root,
                                      transform=data_transforms['train'])
    source_val_dataset = Imagelists_VISDA(source_val, root=root,
                                          transform=data_transforms['val'])
    target_dataset = Imagelists_VISDA(open(image_set_file_t).readlines(), root=root,
                                      transform=data_transforms['val'])
    target_dataset_val = Imagelists_VISDA(open(image_set_file_t_val).readlines(), root=root,
                                          transform=data_transforms['val'])
    target_dataset_unl = Imagelists_VISDA(open(image_set_file_unl).readlines(), root=root,
                                          transform=data_transforms['val'])
    target_dataset_unl.return_index = True

    target_dataset_test = Imagelists_VISDA(open(image_set_file_unl).readlines(), root=root,
                                           transform=data_transforms['test'])
    class_list = return_classlist(image_set_file_s)
    print("%d classes in this dataset" % len(class_list))
    # if args.net == 'alexnet':
    #     bs = 20
    # else:
    #     bs = 16
    bs=args.batch_size
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
    target_loader_unl = \
        torch.utils.data.DataLoader(target_dataset_unl,
                                    batch_size=bs * 1, num_workers=args.worker,
                                    shuffle=True, drop_last=True)
    target_loader_test = \
        torch.utils.data.DataLoader(target_dataset_test,
                                    batch_size=bs * 1, num_workers=args.worker,
                                    shuffle=False, drop_last=False)
    return source_loader, source_val_loader, target_loader, target_loader_unl, \
           target_loader_val, target_loader_test, class_list


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

def Entropy(input_):
    bs = input_.size(0)
    epsilon = 1e-5
    entropy = -input_ * torch.log(input_ + epsilon)
    entropy = torch.sum(entropy, dim=1)
    return entropy